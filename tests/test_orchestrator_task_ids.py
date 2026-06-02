"""Tests for orchestrator task ID preparation."""

from __future__ import annotations

from ollama_fleet.agents.schemas import PlannerOutput, PlannerTask
from ollama_fleet.orchestrator.orchestrator import Orchestrator


def test_create_tasks_from_planner_qualifies_task_ids_and_dependencies() -> None:
    planner_output = PlannerOutput(
        tasks=[
            PlannerTask(
                task_id="task_001",
                title="Build",
                description="Build the thing",
                agent_type="coder",
                dependencies=[],
                priority=5,
            ),
            PlannerTask(
                task_id="task_002",
                title="Test",
                description="Test the thing",
                agent_type="tester",
                dependencies=["task_001"],
                priority=4,
            ),
        ],
        milestones=[],
        architecture_notes="",
    )

    tasks = Orchestrator._create_tasks_from_planner(planner_output, "job-123")

    assert tasks[0]["task_id"] == "job-123:task_001"
    assert tasks[0]["dependencies"] == []
    assert tasks[0]["state"] == "pending"
    assert tasks[1]["task_id"] == "job-123:task_002"
    assert tasks[1]["dependencies"] == ["job-123:task_001"]
    assert tasks[1]["state"] == "blocked"


def test_create_tasks_from_planner_drops_unknown_dependencies() -> None:
    planner_output = PlannerOutput(
        tasks=[
            PlannerTask(
                task_id="task_002",
                title="Build",
                description="Build the thing",
                agent_type="coder",
                dependencies=["task_001", "task_003"],
                priority=5,
            ),
        ],
        milestones=[],
        architecture_notes="",
    )

    tasks = Orchestrator._create_tasks_from_planner(planner_output, "job-123")

    assert tasks[0]["task_id"] == "job-123:task_002"
    assert tasks[0]["dependencies"] == []
    assert tasks[0]["state"] == "pending"


def test_create_tasks_from_planner_renames_duplicate_task_ids() -> None:
    planner_output = PlannerOutput(
        tasks=[
            PlannerTask(
                task_id="task_001",
                title="Build",
                description="Build the thing",
                agent_type="coder",
                dependencies=[],
                priority=5,
            ),
            PlannerTask(
                task_id="task_001",
                title="Build duplicate",
                description="Build another thing",
                agent_type="coder",
                dependencies=["task_001"],
                priority=5,
            ),
        ],
        milestones=[],
        architecture_notes="",
    )

    tasks = Orchestrator._create_tasks_from_planner(planner_output, "job-123")

    assert tasks[0]["task_id"] == "job-123:task_001"
    assert tasks[1]["task_id"] == "job-123:task_001_2"
    assert tasks[1]["dependencies"] == ["job-123:task_001"]
    assert tasks[1]["state"] == "blocked"
