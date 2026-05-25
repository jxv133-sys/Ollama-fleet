"""Tests for job manager CRUD operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.job_manager import JobManager


@pytest.mark.asyncio
async def test_job_manager_create_and_fetch() -> None:
    async with Database(":memory:") as db:
        schema_path = (
            Path(__file__).parent.parent
            / "ollama_fleet"
            / "db"
            / "migrations"
            / "001_initial.sql"
        )
        sql = schema_path.read_text(encoding="utf-8")
        await db._conn.executescript(sql)

        job_manager = JobManager(db)
        job_id = await job_manager.create_job(
            goal="Complete a job record",
            config={"alpha": True},
            workspace_path="/tmp/workspace",
        )
        record = await job_manager.get_job(job_id)

        assert record is not None
        assert record.job_id == job_id
        assert record.goal == "Complete a job record"
        assert record.config["alpha"] is True

        state_list = await job_manager.list_jobs_by_state("submitted")
        assert any(job.job_id == job_id for job in state_list)

        updated = await job_manager.update_job_state(job_id, "running")
        assert updated is True
        record = await job_manager.get_job(job_id)
        assert record is not None
        assert record.state == "running"
