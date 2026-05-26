"""Tests for task scheduler state transitions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from ollama_fleet.db.database import Database
from ollama_fleet.scheduler.task_scheduler import TaskScheduler

MIGRATION_SQL_PATH = (
    Path(__file__).parent.parent
    / "ollama_fleet"
    / "db"
    / "migrations"
    / "001_initial.sql"
)


def _write_schema_sql() -> str:
    return MIGRATION_SQL_PATH.read_text(encoding="utf-8")


async def _insert_job(conn: aiosqlite.Connection, job_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await conn.execute(
        """
        INSERT INTO jobs (job_id, goal, state, created_at, updated_at, config_json, workspace_path, version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (job_id, "goal", "submitted", now, now, "{}", "/tmp/workspace", 0),
    )
    await conn.commit()


async def _create_task(conn: aiosqlite.Connection, task_id: str, job_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await conn.execute(
        """
        INSERT INTO tasks (
            task_id, job_id, title, description, agent_type, state,
            priority, retry_count, dependencies, created_at, updated_at, version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            job_id,
            "task",
            "desc",
            "coder",
            "pending",
            5,
            0,
            "[]",
            now,
            now,
            0,
        ),
    )
    await conn.commit()


@pytest.mark.asyncio
async def test_transition_updates_state_and_version() -> None:
    async with Database(":memory:") as db:
        await db._conn.executescript(_write_schema_sql())
        job_id = str(uuid.uuid4())
        await _insert_job(db._conn, job_id)
        task_id = str(uuid.uuid4())
        await _create_task(db._conn, task_id, job_id)

        scheduler = TaskScheduler(db)
        changed = await scheduler.transition(task_id, "running")
        assert changed is True

        row = await db.fetchone("SELECT state, version FROM tasks WHERE task_id = ?", (task_id,))
        assert row[0] == "running"
        assert row[1] == 1


@pytest.mark.asyncio
async def test_cancel_task_from_pending() -> None:
    async with Database(":memory:") as db:
        await db._conn.executescript(_write_schema_sql())
        job_id = str(uuid.uuid4())
        await _insert_job(db._conn, job_id)
        task_id = str(uuid.uuid4())
        await _create_task(db._conn, task_id, job_id)

        scheduler = TaskScheduler(db)
        cancelled = await scheduler.cancel_task(task_id)
        assert cancelled is True
        row = await db.fetchone("SELECT state FROM tasks WHERE task_id = ?", (task_id,))
        assert row[0] == "cancelled"


@pytest.mark.asyncio
async def test_count_failed_tasks() -> None:
    async with Database(":memory:") as db:
        await db._conn.executescript(_write_schema_sql())
        job_id = str(uuid.uuid4())
        await _insert_job(db._conn, job_id)
        failed_task_id = str(uuid.uuid4())
        pending_task_id = str(uuid.uuid4())
        await _create_task(db._conn, failed_task_id, job_id)
        await _create_task(db._conn, pending_task_id, job_id)

        scheduler = TaskScheduler(db)
        await scheduler.transition(failed_task_id, "failed", reason="boom")

        assert await scheduler.count_failed_tasks(job_id) == 1


@pytest.mark.asyncio
async def test_count_completed_tasks_by_agent() -> None:
    async with Database(":memory:") as db:
        await db._conn.executescript(_write_schema_sql())
        job_id = str(uuid.uuid4())
        await _insert_job(db._conn, job_id)
        coder_task_id = str(uuid.uuid4())
        synth_task_id = str(uuid.uuid4())
        await _create_task(db._conn, coder_task_id, job_id)
        await _create_task(db._conn, synth_task_id, job_id)
        await db.execute_and_commit(
            "UPDATE tasks SET agent_type = ? WHERE task_id = ?",
            ("synthesizer", synth_task_id),
        )

        scheduler = TaskScheduler(db)
        await scheduler.transition(coder_task_id, "completed")
        await scheduler.transition(synth_task_id, "completed")

        assert await scheduler.count_completed_tasks_by_agent(job_id, "coder") == 1
        assert await scheduler.count_completed_tasks_by_agent(job_id, "synthesizer") == 1
