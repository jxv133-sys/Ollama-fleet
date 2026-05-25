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
from ollama_fleet.orchestrator.escalation import EscalationManager
from ollama_fleet.orchestrator.job_manager import JobManager
from ollama_fleet.scheduler.task_scheduler import ScheduledTask, TaskScheduler
from ollama_fleet.validation.validator import ValidationLayer
from ollama_fleet.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)

TERMINAL_STATES = {"completed", "failed", "cancelled"}


class Orchestrator:
    def __init__(self, db: Database, settings: FleetSettings) -> None:
        self.db = db
        self.settings = settings
        self.job_manager = JobManager(db)
        self.scheduler = TaskScheduler(db)
        self.executor = AgentExecutor(OllamaClient(settings.ollama.base_url), settings)
        self.validation = ValidationLayer()
        self.revision_counts: dict[str, int] = {}
        self.revision_issues: dict[str, list[dict[str, Any]]] = {}

    async def submit_job(self, goal: str, config: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
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
            ready_tasks = await self.scheduler.get_ready_tasks(job_id)
            if not ready_tasks:
                active = await self.scheduler.count_active_tasks(job_id)
                if active == 0:
                    await self.job_manager.update_job_state(job_id, "completed")
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

        if task.agent_type == AgentType.CODER.value:
            await self._run_coder_task(task, workspace, escalation_manager)
        elif task.agent_type == AgentType.TESTER.value:
            await self._run_tester_task(task)
        elif task.agent_type == AgentType.SYNTHESIZER.value:
            await self._run_synthesizer_task(task)
        else:
            await self.scheduler.transition(task.task_id, "completed")

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
            tasks.append(
                {
                    "task_id": task.task_id,
                    "job_id": job_id,
                    "title": task.title,
                    "description": task.description,
                    "agent_type": task.agent_type,
                    "state": "pending",
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
        extra_context: dict[str, Any] = {}
        if task.task_id in self.revision_issues:
            extra_context["critic_issues"] = self.revision_issues[task.task_id]

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
            return

        modified_files = []
        for change in coder_output.file_modifications:
            workspace_manager.write_file(change.file_path, change.content)
            modified_files.append(change.file_path)

        validation_result = self.validation.validate(modified_files, workspace_manager)
        if not validation_result.syntax_ok:
            await self.scheduler.transition(task.task_id, "pending")
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
            },
        )

        if isinstance(critic_output, CriticOutput) and not critic_output.approved:
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
                return
            await self.scheduler.transition(task.task_id, "pending")
            return

        await self.scheduler.transition(task.task_id, "completed")

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
