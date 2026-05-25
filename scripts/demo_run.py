#!/usr/bin/env python3
"""Run a demo job using a DummyExecutor to avoid calling external models.

Creates a small workspace under `demo_workspaces/` and runs the orchestrator
through Planner -> Coder -> Critic -> Tester flow using deterministic
responses.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ollama_fleet.agents.schemas import (
    AgentType,
    PlannerOutput,
    PlannerTask,
    CoderOutput,
    FileModification,
    CriticOutput,
    CriticIssue,
    TesterOutput,
)
from ollama_fleet.config import FleetSettings
from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.orchestrator import Orchestrator


class DummyExecutor:
    def __init__(self, settings: FleetSettings | None = None):
        self.settings = settings
        self.client = None

    async def execute(self, task: dict[str, Any], agent_type: AgentType, extra_context: dict[str, Any] | None = None):
        # Simple deterministic behavior for demo purposes
        if agent_type == AgentType.PLANNER:
            t = PlannerTask(task_id="task-1", title="Create demo module", description="Add demo.py with a greet() function", agent_type="coder", dependencies=[], priority=5)
            return PlannerOutput(tasks=[t], milestones=["demo-ready"], architecture_notes="demo arch")

        if agent_type == AgentType.CODER:
            fm = FileModification(file_path="src/demo.py", operation="create", content="def greet(name):\n    return f\"Hello, {name}!\"\n")
            return CoderOutput(file_modifications=[fm], summary="Added demo.greet", confidence_score=0.95)

        if agent_type == AgentType.CRITIC:
            # approve by default
            return CriticOutput(approved=True, issues=[], overall_assessment="Looks good")

        if agent_type == AgentType.TESTER:
            return TesterOutput(tests_passed=1, tests_failed=0, failures=[], ready_for_review=True)

        raise RuntimeError("Unsupported agent type in DummyExecutor")


async def main() -> int:
    db_path = Path("demo_fleet.db")
    if db_path.exists():
        db_path.unlink()

    settings = FleetSettings()
    # use a demo workspace base under the repo; perform a safe model re-validate
    sd = settings.model_dump()
    sd["workspace"]["base_path"] = "demo_workspaces"
    settings = FleetSettings.model_validate(sd)

    db = Database(db_path)
    await db.connect()

    orch = Orchestrator(db, settings)
    orch.executor = DummyExecutor(settings)

    job_id = await orch.submit_job(goal="Create a demo project", config={"source": "demo"})
    print("Submitted job:", job_id)

    # show workspace path and demo file
    job = await orch.job_manager.get_job(job_id)
    if job:
        ws = Path(job.workspace_path)
        demo_file = ws / "src" / "demo.py"
        print("Workspace:", ws)
        if demo_file.exists():
            print("Demo file content:\n", demo_file.read_text(encoding="utf-8"))

    await db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
