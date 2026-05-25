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
from ollama_fleet.scheduler.dependency_resolver import DependencyResolver
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
        self.dependency_resolver = DependencyResolver(db)
        self.executor = AgentExecutor(OllamaClient(settings.ollama.base_url), settings)
        self.validation = ValidationLayer()
        self.memory_system = MemorySystem(db, settings, self.executor.client)
        self.revision_counts: dict[str, int] = {}
        self.revision_issues: dict[str, list[dict[str, Any]]] = {}
        self.previous_coder_outputs: dict[str, list[tuple[str, str]]] = {}
        self.ui_bus = ui_bus or UIEventBus()

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
                "type": "job_state_changed",
                "job_id": job_id,
                "new_state": "submitted",
            }
        )
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

        tasks = self._create_tasks_from_planner(planner_output, job_id)
        await self.scheduler.enqueue_tasks(tasks)
        await self.dispatch_loop(job_id, workspace_manager, escalation_manager)
        return job_id

    async def dispatch_loop(
        self,
        job_id: str,
        workspace_manager: WorkspaceManager,
        escalation_manager: EscalationManager,
    ) -> None:
        while True:
            await self.scheduler.resolve_dependencies(job_id)
            await self.scheduler.recover_stalled_tasks(job_id, self.settings.scheduler.stall_timeout)
            ready_tasks = await self.scheduler.get_ready_tasks(job_id)
            if not ready_tasks:
                active = await self.scheduler.count_active_tasks(job_id)
                if active == 0:
                    await self.job_manager.update_job_state(job_id, "completed")
                    self._publish_event(
                        {
                            "type": "job_state_changed",
                            "job_id": job_id,
                            "new_state": "completed",
                        }
                    )
                    break
                await asyncio.sleep(0.1)
                continue

            for task in ready_tasks:
                await self._dispatch_task(task, workspace_manager, escalation_manager)
                if await self.scheduler.count_active_tasks(job_id) == 0:
                    break

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

    def _create_tasks_from_planner(self, planner_output: PlannerOutput, job_id: str) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        tasks = []
        for task in planner_output.tasks:
            state = "pending" if not task.dependencies else "blocked"
            tasks.append(
                {
                    "task_id": task.task_id,
                    "job_id": job_id,
                    "title": task.title,
                    "description": task.description,
                    "agent_type": task.agent_type,
                    "state": state,
                    "priority": task.priority,
                    "retry_count": 0,
                    "dependencies": task.dependencies,
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

        modified_files: list[str] = []
        for change in coder_output.file_modifications:
            workspace_manager.write_file(change.file_path, change.content)
            modified_files.append(change.file_path)

        validation_result = self.validation.validate(modified_files, workspace_manager)
        self._publish_event(
            {"type": "validation_result", "validation_result": vars(validation_result)}
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
        await self.executor.execute(
            {
                "task_id": task.task_id,
                "goal": "",
                "description": task.description,
            },
            AgentType.TESTER,
            extra_context={"workspace_state": "", "test_results": ""},
        )
        await self.scheduler.transition(task.task_id, "completed")

    async def _run_synthesizer_task(self, task: ScheduledTask) -> None:
        await self.executor.execute(
            {
                "task_id": task.task_id,
                "goal": "",
                "description": task.description,
            },
            AgentType.SYNTHESIZER,
            extra_context={
                "goal": "",
                "completed_summaries": [],
                "files_produced": [],
            },
        )
        await self.scheduler.transition(task.task_id, "completed")
