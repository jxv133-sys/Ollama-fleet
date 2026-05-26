from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from ollama_fleet.agents import coder as coder_module
from ollama_fleet.agents import critic as critic_module
from ollama_fleet.agents import planner as planner_module
from ollama_fleet.agents import synthesizer as synthesizer_module
from ollama_fleet.agents import tester as tester_module
from ollama_fleet.agents.executor import AgentExecutor
from ollama_fleet.agents.schemas import AgentType, CoderOutput, CriticOutput, PlannerOutput, SynthesizerOutput, TesterOutput
from ollama_fleet.ollama.client import OllamaClient
from ollama_fleet.config import FleetSettings
from ollama_fleet.db.database import Database
from ollama_fleet.memory.episodic import EpisodicEntry
from ollama_fleet.memory.memory_system import MemorySystem
from ollama_fleet.orchestrator.escalation import EscalationManager
from ollama_fleet.orchestrator.job_manager import JobManager
from ollama_fleet.scheduler.task_scheduler import ScheduledTask, TaskScheduler
from ollama_fleet.validation.validator import ValidationLayer
from ollama_fleet.ui.event_bus import UIEventBus
from ollama_fleet.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)

TERMINAL_STATES = {"completed", "failed", "cancelled"}


class Orchestrator:
    def __init__(
        self,
        db: Database,
        settings: FleetSettings,
        ui_bus: UIEventBus | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.job_manager = JobManager(db)
        self.scheduler = TaskScheduler(db)
        self.executor = AgentExecutor(OllamaClient(settings.ollama.base_url), settings)
        self.validation = ValidationLayer()
        self.memory_system = MemorySystem(db, settings, self.executor.client)
        self.revision_counts: dict[str, int] = {}
        self.revision_issues: dict[str, list[dict[str, Any]]] = {}
        self.previous_coder_outputs: dict[str, list[tuple[str, str]]] = {}
        self.ui_bus = ui_bus or UIEventBus()
        self._stall_task: asyncio.Task | None = None

    async def submit_job(self, goal: str, config: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        await self._recover_running_tasks()

        workspace_manager = WorkspaceManager.create_workspace(
            job_id=job_id,
            goal=goal,
            config=config,
            base_path=self.settings.workspace.base_path,
        )

        await self.job_manager.create_job(
            goal=goal,
            config=config,
            workspace_path=str(workspace_manager.root),
            job_id=job_id,
        )
        self._publish_event(
            {
                "type": "workspace_created",
                "workspace_path": str(workspace_manager.root),
            }
        )
        self._publish_event(
            {
                "type": "job_state_changed",
                "job_id": job_id,
                "new_state": "submitted",
                "goal": goal,
            }
        )
        # Allow UI to render job submission
        await asyncio.sleep(0.3)
        
        workspace_manager.append_execution_history(
            {"event": "job_submitted", "job_id": job_id, "goal": goal}
        )
        escalation_manager = EscalationManager(self.db, workspace_manager)

        try:
            planner_output = await self._run_planner(goal)
        except Exception as exc:
            logger.exception("Planner failed for job %s", job_id)
            await self.job_manager.update_job_state(job_id, "failed")
            await escalation_manager.write_escalation(
                task_id="planner",
                job_id=job_id,
                reason=str(exc),
                retry_count=0,
            )
            self._publish_event(
                {
                    "type": "job_state_changed",
                    "job_id": job_id,
                    "new_state": "failed",
                }
            )
            return job_id
        finally:
            # start stall detector if not already running (safe no-op if loop not running)
            if getattr(self, "_stall_task", None) is None or (self._stall_task is not None and self._stall_task.done()):
                try:
                    self._stall_task = asyncio.create_task(self._check_stall())
                except RuntimeError:
                    # If there's no running event loop yet, skip; it will be created on next submit
                    self._stall_task = None

        tasks = self._create_tasks_from_planner(planner_output, job_id)
        await self.scheduler.enqueue_tasks(tasks)
        
        # Publish planner output event for dashboard
        self._publish_event({
            "type": "agent_output",
            "agent_type": AgentType.PLANNER.value,
            "output": {
                "tasks_created": len(tasks),
                "milestones": planner_output.milestones,
                "architecture": planner_output.architecture_notes[:100] if planner_output.architecture_notes else "",
            }
        })
        
        self._publish_event({
            "type": "agent_log",
            "message": f"[PLANNER] Created {len(tasks)} tasks ready for execution"
        })
        
        # Save planner output to workspace
        planner_output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_type": AgentType.PLANNER.value,
            "tasks_created": len(tasks),
            "milestones": planner_output.milestones,
            "architecture_notes": planner_output.architecture_notes,
            "tasks": [
                {
                    "task_id": task["task_id"],
                    "title": task["title"],
                    "description": task["description"],
                    "agent_type": task["agent_type"],
                    "dependencies": task["dependencies"],
                    "priority": task["priority"],
                }
                for task in tasks
            ],
        }
        planner_count = len(list(workspace_manager.root.glob("agent_outputs/planner_*.json"))) + 1
        workspace_manager.write_file(
            f"agent_outputs/planner_{planner_count}.json",
            json.dumps(planner_output_data, indent=2),
        )
        
        # Allow UI to display planner tasks
        await asyncio.sleep(0.3)
        
        await self.dispatch_loop(job_id, workspace_manager, escalation_manager)
        return job_id

    async def dispatch_loop(
        self,
        job_id: str,
        workspace_manager: WorkspaceManager,
        escalation_manager: EscalationManager,
    ) -> None:
        semaphore = asyncio.Semaphore(self.settings.scheduler.max_concurrent_tasks)

        async def _dispatch_with_semaphore(task: ScheduledTask) -> None:
            async with semaphore:
                await self._dispatch_task(task, workspace_manager, escalation_manager)

        while True:
            await self.scheduler.resolve_dependencies(job_id)
            await self.scheduler.recover_stalled_tasks(job_id, self.settings.scheduler.stall_timeout)
            ready_tasks = await self.scheduler.get_ready_tasks(job_id)
            if not ready_tasks:
                active = await self.scheduler.count_active_tasks(job_id)
                if active == 0:
                    failed = await self.scheduler.count_failed_tasks(job_id)
                    completed_coder_tasks = await self.scheduler.count_completed_tasks_by_agent(
                        job_id,
                        AgentType.CODER.value,
                    )
                    final_state = "failed" if failed or completed_coder_tasks == 0 else "completed"
                    if completed_coder_tasks == 0 and failed == 0:
                        await escalation_manager.write_escalation(
                            task_id="planner",
                            job_id=job_id,
                            reason="Planner produced no completed coder tasks; no files were created.",
                            retry_count=0,
                        )
                        self._publish_event(
                            {
                                "type": "agent_log",
                                "message": "Job failed: planner produced no completed coder tasks, so no files were created.",
                                "level": "error",
                            }
                        )
                    await self.job_manager.update_job_state(job_id, final_state)
                    self._publish_event(
                        {
                            "type": "job_state_changed",
                            "job_id": job_id,
                            "new_state": final_state,
                        }
                    )
                    break
                await asyncio.sleep(0.1)
                continue

            # Dispatch all ready tasks concurrently up to max_concurrent_tasks
            dispatch_tasks = [
                asyncio.create_task(_dispatch_with_semaphore(task))
                for task in ready_tasks
            ]
            await asyncio.gather(*dispatch_tasks, return_exceptions=True)

            if await self.scheduler.count_failed_tasks(job_id) > 0:
                await self.job_manager.update_job_state(job_id, "failed")
                self._publish_event(
                    {
                        "type": "job_state_changed",
                        "job_id": job_id,
                        "new_state": "failed",
                    }
                )
                return

    async def _dispatch_task(
        self,
        task: ScheduledTask,
        workspace: WorkspaceManager,
        escalation_manager: EscalationManager,
    ) -> None:
        await self.scheduler.transition(task.task_id, "running")
        self._publish_event(
            {
                "type": "task_state_changed",
                "task_id": task.task_id,
                "agent_type": task.agent_type,
                "new_state": "running",
            }
        )
        # Brief pause to allow UI to display task start
        await asyncio.sleep(0.2)

        try:
            if task.agent_type == AgentType.CODER.value:
                await self._run_coder_task(task, workspace, escalation_manager)
            elif task.agent_type == AgentType.TESTER.value:
                await self._run_tester_task(task)
            elif task.agent_type == AgentType.SYNTHESIZER.value:
                await self._run_synthesizer_task(task)
            else:
                await self.scheduler.transition(task.task_id, "completed")
                self._publish_event(
                    {
                        "type": "task_state_changed",
                        "task_id": task.task_id,
                        "agent_type": task.agent_type,
                        "new_state": "completed",
                    }
                )
        except Exception as exc:
            logger.exception("Task %s failed during %s execution", task.task_id, task.agent_type)
            await self.scheduler.transition(task.task_id, "failed", reason=str(exc))
            await escalation_manager.write_escalation(
                task_id=task.task_id,
                job_id=task.job_id,
                reason=str(exc),
                retry_count=task.retry_count,
            )
            self._publish_event(
                {
                    "type": "task_state_changed",
                    "task_id": task.task_id,
                    "agent_type": task.agent_type,
                    "new_state": "failed",
                    "reason": str(exc),
                }
            )
            self._publish_event(
                {
                    "type": "agent_log",
                    "message": f"Task {task.task_id} failed: {exc}",
                    "level": "error",
                }
            )
        
        # Brief pause after task completion to allow UI to update
        await asyncio.sleep(0.1)

    async def _recover_running_tasks(self) -> None:
        rows = await self.db.fetchall(
            "SELECT task_id FROM tasks WHERE state = 'running'",
        )
        for row in rows:
            task_id = row[0]
            await self.scheduler.transition(task_id, "pending")
            self._publish_event(
                {
                    "type": "agent_log",
                    "message": f"Recovered stalled task {task_id} to pending state.",
                }
            )

    def _publish_event(self, event: dict[str, Any]) -> None:
        try:
            self.ui_bus.publish(event)
        except Exception:
            pass

    async def _run_planner(self, goal: str) -> PlannerOutput:
        output = await self.executor.execute(
            {
                "task_id": "planner",
                "goal": goal,
                "description": "",
            },
            AgentType.PLANNER,
            extra_context={"architecture_notes": ""},
        )
        if not isinstance(output, PlannerOutput):
            raise RuntimeError("Planner did not return a PlannerOutput")
        return output

    @staticmethod
    def _create_tasks_from_planner(planner_output: PlannerOutput, job_id: str) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        task_id_map = {
            task.task_id: f"{job_id}:{task.task_id}"
            for task in planner_output.tasks
        }
        known_task_ids = set(task_id_map.values())
        tasks = []
        for task in planner_output.tasks:
            dependencies = [
                mapped_dependency
                for dependency in task.dependencies
                if (mapped_dependency := task_id_map.get(dependency, dependency)) in known_task_ids
            ]
            state = "pending" if not dependencies else "blocked"
            tasks.append(
                {
                    "task_id": task_id_map[task.task_id],
                    "job_id": job_id,
                    "title": task.title,
                    "description": task.description,
                    "agent_type": task.agent_type,
                    "state": state,
                    "priority": task.priority,
                    "retry_count": 0,
                    "dependencies": dependencies,
                    "created_at": now,
                    "updated_at": now,
                    "version": 0,
                }
            )
        return tasks

    async def _run_coder_task(
        self,
        task: ScheduledTask,
        workspace_manager: WorkspaceManager,
        escalation_manager: EscalationManager,
    ) -> None:
        active_files = [
            str(path.relative_to(workspace_manager.root))
            for path in workspace_manager.root.rglob("*.py")
            if path.is_file()
        ]
        file_contents = {
            path: (workspace_manager.root / path).read_text(encoding="utf-8")
            for path in active_files
            if (workspace_manager.root / path).exists()
        }

        extra_context = {
            "active_files": active_files,
            "episodic_summaries": [],
        }
        if task.task_id in self.revision_issues:
            extra_context["critic_issues"] = self.revision_issues[task.task_id]

        memory_context = await self.memory_system.assemble_context(
            task_description=task.description,
            job_id=task.job_id,
            active_files=active_files,
            file_contents=file_contents,
        )
        extra_context["active_files"] = memory_context.active_files
        extra_context["episodic_summaries"] = memory_context.episodic_summaries

        coder_output = await self.executor.execute(
            {
                "task_id": task.task_id,
                "goal": "",
                "description": task.description,
            },
            AgentType.CODER,
            extra_context=extra_context,
        )

        # Publish coder output event for dashboard
        self._publish_event({
            "type": "agent_output",
            "agent_type": AgentType.CODER.value,
            "output": {
                "file_count": len(coder_output.file_modifications) if isinstance(coder_output, CoderOutput) else 0,
                "confidence": coder_output.confidence_score if isinstance(coder_output, CoderOutput) else 0,
                "summary": coder_output.summary if isinstance(coder_output, CoderOutput) else "",
            }
        })

        if not isinstance(coder_output, CoderOutput):
            await self.scheduler.transition(task.task_id, "failed")
            self._publish_event(
                {
                    "type": "task_state_changed",
                    "task_id": task.task_id,
                    "agent_type": task.agent_type,
                    "new_state": "failed",
                }
            )
            return

        current_modifications = [
            (change.file_path, change.content) for change in coder_output.file_modifications
        ]
        if self.previous_coder_outputs.get(task.task_id) == current_modifications:
            await escalation_manager.write_escalation(
                task_id=task.task_id,
                job_id=task.job_id,
                reason="Identical coder output detected",
                retry_count=self.revision_counts.get(task.task_id, 0),
            )
            await self.scheduler.transition(task.task_id, "failed")
            self._publish_event(
                {
                    "type": "escalation_added",
                    "escalation": {
                        "task_id": task.task_id,
                        "job_id": task.job_id,
                        "reason": "Identical coder output detected",
                        "retry_count": self.revision_counts.get(task.task_id, 0),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )
            return

        self.previous_coder_outputs[task.task_id] = current_modifications

        if not coder_output.file_modifications:
            await self.scheduler.transition(task.task_id, "failed", reason="Coder produced no file modifications")
            self._publish_event(
                {
                    "type": "task_state_changed",
                    "task_id": task.task_id,
                    "agent_type": task.agent_type,
                    "new_state": "failed",
                    "reason": "Coder produced no file modifications",
                }
            )
            return

        modified_files: list[str] = []
        for change in coder_output.file_modifications:
            workspace_manager.write_file(change.file_path, change.content)
            modified_files.append(change.file_path)
            # Publish file written event
            self._publish_event({
                "type": "file_written",
                "path": change.file_path,
                "job_id": task.job_id,
            })

        validation_result = self.validation.validate(modified_files, workspace_manager)
        self._publish_event(
            {"type": "validation_result", "validation_result": vars(validation_result)}
        )
        
        # Save coder output to workspace
        coder_output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_type": AgentType.CODER.value,
            "task_id": task.task_id,
            "summary": coder_output.summary,
            "confidence_score": coder_output.confidence_score,
            "file_modifications": [
                {
                    "file_path": mod.file_path,
                    "content_preview": mod.content[:200] if len(mod.content) > 200 else mod.content,
                    "content_length": len(mod.content),
                }
                for mod in coder_output.file_modifications
            ],
            "files_created": modified_files,
            "validation": {
                "syntax_ok": validation_result.syntax_ok,
                "linter_available": validation_result.linter_available,
                "lint_results": len(validation_result.lint_results),
            },
        }
        coder_count = len(list(workspace_manager.root.glob("agent_outputs/coder_*.json"))) + 1
        workspace_manager.write_file(
            f"agent_outputs/coder_{coder_count}.json",
            json.dumps(coder_output_data, indent=2),
        )
        
        if not validation_result.syntax_ok:
            await self.scheduler.transition(task.task_id, "pending")
            self._publish_event(
                {
                    "type": "task_state_changed",
                    "task_id": task.task_id,
                    "agent_type": task.agent_type,
                    "new_state": "pending",
                }
            )
            return

        critic_output = await self.executor.execute(
            {
                "task_id": task.task_id,
                "goal": "",
                "description": task.description,
            },
            AgentType.CRITIC,
            extra_context={
                "modified_files": modified_files,
                "file_contents": {
                    path: (workspace_manager.root / path).read_text(encoding="utf-8")
                    for path in modified_files
                },
                "lint_results": [vars(issue) for issue in validation_result.lint_results],
                "critic_issues": self.revision_issues.get(task.task_id, []),
            },
        )

        # Publish critic output event for dashboard
        self._publish_event({
            "type": "agent_output",
            "agent_type": AgentType.CRITIC.value,
            "output": {
                "approved": critic_output.approved if isinstance(critic_output, CriticOutput) else False,
                "issues_found": len(critic_output.issues) if isinstance(critic_output, CriticOutput) else 0,
                "assessment": critic_output.overall_assessment if isinstance(critic_output, CriticOutput) else "",
            }
        })
        
        # Save critic output to workspace
        if isinstance(critic_output, CriticOutput):
            critic_output_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_type": AgentType.CRITIC.value,
                "task_id": task.task_id,
                "approved": critic_output.approved,
                "overall_assessment": critic_output.overall_assessment,
                "issues": [
                    {
                        "severity": issue.severity,
                        "message": issue.description,
                        "line": issue.line_number,
                        "file": issue.file_path,
                        "suggested_fix": issue.suggested_fix,
                    }
                    for issue in critic_output.issues
                ],
            }
            critic_count = len(list(workspace_manager.root.glob("agent_outputs/critic_*.json"))) + 1
            workspace_manager.write_file(
                f"agent_outputs/critic_{critic_count}.json",
                json.dumps(critic_output_data, indent=2),
            )

        if isinstance(critic_output, CriticOutput) and not critic_output.approved:
            await self._handle_critic_output(task, critic_output, escalation_manager)
            return

        if coder_output.confidence_score < 0.4:
            self._publish_event(
                {
                    "type": "agent_log",
                    "message": f"Low confidence detected for {task.task_id}: {coder_output.confidence_score:.2f}",
                }
            )

        await self.memory_system.save_episodic(
            EpisodicEntry(
                job_id=task.job_id,
                task_id=task.task_id,
                agent_type=task.agent_type,
                outcome="completed",
                files_modified=modified_files,
                summary_text=coder_output.summary,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

        await self.scheduler.transition(task.task_id, "completed")
        self._publish_event(
            {
                "type": "task_state_changed",
                "task_id": task.task_id,
                "agent_type": task.agent_type,
                "new_state": "completed",
            }
        )

    async def _handle_critic_output(
        self,
        task: ScheduledTask,
        critic_output: CriticOutput,
        escalation_manager: EscalationManager,
    ) -> None:
        count = self.revision_counts.get(task.task_id, 0) + 1
        self.revision_counts[task.task_id] = count
        self.revision_issues[task.task_id] = [vars(issue) for issue in critic_output.issues]
        if count >= self.settings.scheduler.max_critique_revision_loops:
            await escalation_manager.write_escalation(
                task_id=task.task_id,
                job_id=task.job_id,
                reason="Critic revision loop exceeded",
                retry_count=count,
            )
            await self.scheduler.transition(task.task_id, "failed")
            self._publish_event(
                {
                    "type": "task_state_changed",
                    "task_id": task.task_id,
                    "agent_type": task.agent_type,
                    "new_state": "failed",
                }
            )
            self._publish_event(
                {
                    "type": "escalation_added",
                    "escalation": {
                        "task_id": task.task_id,
                        "job_id": task.job_id,
                        "reason": "Critic revision loop exceeded",
                        "retry_count": count,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )
            return

        await self.scheduler.transition(task.task_id, "pending")
        self._publish_event(
            {
                "type": "task_state_changed",
                "task_id": task.task_id,
                "agent_type": task.agent_type,
                "new_state": "pending",
            }
        )
        self._publish_event(
            {
                "type": "agent_log",
                "message": f"Task {task.task_id} requires revision loop {count}",
            }
        )

    async def _run_tester_task(self, task: ScheduledTask) -> None:
        # Gather actual workspace state for the tester
        job = await self.job_manager.get_job(task.job_id)
        workspace_state = ""
        test_results = ""
        if job is not None:
            try:
                ws = WorkspaceManager(job.workspace_path)
                py_files = [
                    str(p.relative_to(ws.root))
                    for p in ws.root.rglob("*.py")
                    if p.is_file()
                ]
                workspace_state = "\n".join(py_files)
                # Run tests and capture output
                from ollama_fleet.tools.shell_tools import ShellTools
                shell = ShellTools(ws.root)
                run_result = shell.run_tests(timeout=self.settings.tools.command_timeout)
                test_results = run_result.get("stdout", "") + run_result.get("stderr", "")
            except Exception:
                pass

        tester_output = await self.executor.execute(
            {
                "task_id": task.task_id,
                "goal": "",
                "description": task.description,
            },
            AgentType.TESTER,
            extra_context={"workspace_state": workspace_state, "test_results": test_results},
        )
        if isinstance(tester_output, TesterOutput):
            self._publish_event(
                {
                    "type": "agent_output",
                    "agent_type": AgentType.TESTER.value,
                    "output": {
                        "tests_passed": tester_output.tests_passed,
                        "tests_failed": tester_output.tests_failed,
                        "ready_for_review": tester_output.ready_for_review,
                    },
                }
            )
        await self.scheduler.transition(task.task_id, "completed")
        self._publish_event(
            {
                "type": "task_state_changed",
                "task_id": task.task_id,
                "agent_type": task.agent_type,
                "new_state": "completed",
            }
        )

    async def _run_synthesizer_task(self, task: ScheduledTask) -> None:
        # Gather actual job context for the synthesizer
        job = await self.job_manager.get_job(task.job_id)
        goal = ""
        completed_summaries: list[str] = []
        files_produced: list[str] = []
        if job is not None:
            goal = job.goal
            try:
                ws = WorkspaceManager(job.workspace_path)
                files_produced = [
                    str(p.relative_to(ws.root))
                    for p in ws.root.rglob("*")
                    if p.is_file() and not p.name.startswith(".")
                ]
                # Pull episodic summaries for context
                memory_ctx = await self.memory_system.assemble_context(
                    task_description=task.description,
                    job_id=task.job_id,
                )
                completed_summaries = memory_ctx.episodic_summaries
            except Exception:
                pass

        synthesizer_output = await self.executor.execute(
            {
                "task_id": task.task_id,
                "goal": goal,
                "description": task.description,
            },
            AgentType.SYNTHESIZER,
            extra_context={
                "goal": goal,
                "completed_summaries": completed_summaries,
                "files_produced": files_produced,
            },
        )
        if isinstance(synthesizer_output, SynthesizerOutput):
            self._publish_event(
                {
                    "type": "agent_output",
                    "agent_type": AgentType.SYNTHESIZER.value,
                    "output": {
                        "summary": synthesizer_output.summary,
                        "files_produced": synthesizer_output.files_produced,
                    },
                }
            )
        await self.scheduler.transition(task.task_id, "completed")
        self._publish_event(
            {
                "type": "task_state_changed",
                "task_id": task.task_id,
                "agent_type": task.agent_type,
                "new_state": "completed",
            }
        )

    async def _check_stall(self) -> None:
        """Background coroutine: detect stalled jobs and escalate them.

        Checks the most-recent `updated_at` timestamp for non-terminal tasks
        grouped by job and marks jobs as `failed` if no progress has been made
        for longer than `scheduler.stall_timeout` seconds.
        """
        while True:
            try:
                await asyncio.sleep(60)
                # Find running tasks across jobs and check per-task progress
                rows = await self.db.fetchall(
                    "SELECT task_id, job_id, agent_type, updated_at, retry_count FROM tasks WHERE state = 'running'"
                )
                now_ts = datetime.now(timezone.utc).timestamp()
                for task_id, job_id, agent_type, updated_at, retry_count in rows:
                    if updated_at is None:
                        continue
                    try:
                        updated_ts = datetime.fromisoformat(updated_at).timestamp()
                    except Exception:
                        continue
                    elapsed = now_ts - updated_ts
                    stall_threshold = float(self.settings.scheduler.stall_timeout)
                    if elapsed <= stall_threshold:
                        continue

                    # If the task has already retried too many times, fail and escalate
                    if int(retry_count or 0) >= int(self.settings.scheduler.retry_limit):
                        # mark task failed and write escalation
                        await self.scheduler.transition(task_id, "failed", reason="Stalled and retry limit exceeded")
                        # derive workspace for metadata and write escalation
                        job = await self.job_manager.get_job(job_id)
                        if job is not None:
                            try:
                                workspace_manager = WorkspaceManager(job.workspace_path)
                                esc_mgr = EscalationManager(self.db, workspace_manager)
                                await esc_mgr.write_escalation(
                                    task_id=task_id,
                                    job_id=job_id,
                                    reason="Stalled task exceeded retry limit",
                                    retry_count=int(retry_count or 0),
                                )
                            except Exception:
                                # best-effort: continue even if workspace unavailable
                                pass
                        self._publish_event({
                            "type": "task_state_changed",
                            "task_id": task_id,
                            "agent_type": agent_type,
                            "new_state": "failed",
                        })
                        self._publish_event({
                            "type": "agent_log",
                            "message": f"Task {task_id} failed: stalled and retry limit exceeded.",
                        })
                        continue

                    # Otherwise increment retry and requeue the task to pending
                    try:
                        new_retry = await self.scheduler.increment_retry(task_id)
                        await self.scheduler.transition(task_id, "pending", reason="Stalled; requeued for retry")
                        await self.db.execute_and_commit(
                            "INSERT INTO escalations (task_id, job_id, reason, retry_count, timestamp, dismissed) VALUES (?, ?, ?, ?, ?, 0)",
                            [task_id, job_id, "Stalled task requeued for retry", new_retry, datetime.now(timezone.utc).isoformat()],
                        )
                        # append execution history if workspace exists
                        job = await self.job_manager.get_job(job_id)
                        if job is not None:
                            try:
                                workspace_manager = WorkspaceManager(job.workspace_path)
                                workspace_manager.append_execution_history({
                                    "event": "stalled_task_requeued",
                                    "task_id": task_id,
                                    "elapsed_seconds": elapsed,
                                    "retry_count": new_retry,
                                })
                            except Exception:
                                pass

                        self._publish_event({
                            "type": "task_state_changed",
                            "task_id": task_id,
                            "agent_type": agent_type,
                            "new_state": "pending",
                        })
                        self._publish_event({
                            "type": "agent_log",
                            "message": f"Task {task_id} requeued after stall; retry {new_retry}.",
                        })
                    except Exception:
                        logger.exception("Failed handling stalled task %s", task_id)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in stall detection loop")
