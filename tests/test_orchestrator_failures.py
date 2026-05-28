"""Tests for orchestrator task failure handling."""

from __future__ import annotations

from typing import Any

import pytest

from ollama_fleet.orchestrator.orchestrator import Orchestrator
from ollama_fleet.scheduler.task_scheduler import ScheduledTask


class FakeScheduler:
    def __init__(self) -> None:
        self.transitions: list[tuple[str, str, str | None]] = []

    async def transition(
        self,
        task_id: str,
        new_state: str,
        reason: str | None = None,
    ) -> bool:
        self.transitions.append((task_id, new_state, reason))
        return True


class FakeEscalationManager:
    def __init__(self) -> None:
        self.escalations: list[dict[str, Any]] = []

    async def write_escalation(
        self,
        task_id: str,
        job_id: str,
        reason: str,
        retry_count: int,
    ) -> None:
        self.escalations.append(
            {
                "task_id": task_id,
                "job_id": job_id,
                "reason": reason,
                "retry_count": retry_count,
            }
        )


class FakeEventBus:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_dispatch_task_marks_agent_exception_failed() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.scheduler = FakeScheduler()
    orchestrator.ui_bus = FakeEventBus()

    async def failing_coder_task(
        task: ScheduledTask,
        workspace: object,
        escalation_manager: FakeEscalationManager,
    ) -> None:
        raise RuntimeError("model failed to load")

    orchestrator._run_coder_task = failing_coder_task
    escalation_manager = FakeEscalationManager()
    task = ScheduledTask(
        task_id="job-1:task_001",
        job_id="job-1",
        title="Build",
        description="Build code",
        agent_type="coder",
        state="pending",
        priority=5,
        retry_count=2,
        dependencies=[],
        created_at="",
        updated_at="",
        version=0,
    )

    await orchestrator._dispatch_task(task, object(), escalation_manager)

    assert orchestrator.scheduler.transitions[0] == ("job-1:task_001", "running", None)
    assert orchestrator.scheduler.transitions[1] == (
        "job-1:task_001",
        "failed",
        "model failed to load",
    )
    assert escalation_manager.escalations == [
        {
            "task_id": "job-1:task_001",
            "job_id": "job-1",
            "reason": "model failed to load",
            "retry_count": 2,
        }
    ]
    assert orchestrator.ui_bus.events[-2]["new_state"] == "failed"
    assert orchestrator.ui_bus.events[-1]["level"] == "error"

