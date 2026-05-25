"""Property-based tests for the database layer.

Property 5: Task State Persistence Round-Trip
  Validates: Requirements 2.1

For any task created with any valid state value from the set
{pending, running, completed, failed, blocked, cancelled},
writing the task to SQLite and reading it back SHALL produce a record
with the identical state value.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIGRATION_SQL_PATH = (
    Path(__file__).parent.parent
    / "ollama_fleet"
    / "db"
    / "migrations"
    / "001_initial.sql"
)

VALID_TASK_STATES = ["pending", "running", "completed", "failed", "blocked", "cancelled"]
VALID_AGENT_TYPES = ["planner", "coder", "critic", "tester", "synthesizer"]


async def _create_schema(conn: aiosqlite.Connection) -> None:
    """Apply the initial migration SQL to an in-memory database."""
    sql = MIGRATION_SQL_PATH.read_text(encoding="utf-8")
    await conn.executescript(sql)
    await conn.commit()


async def _insert_job(conn: aiosqlite.Connection, job_id: str) -> None:
    """Insert a minimal job record so the tasks FK constraint is satisfied."""
    now = datetime.now(timezone.utc).isoformat()
    await conn.execute(
        """
        INSERT INTO jobs (job_id, goal, state, created_at, updated_at, config_json, workspace_path, version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (job_id, "test goal", "submitted", now, now, "{}", "/tmp/workspace", 0),
    )
    await conn.commit()


async def _insert_task(
    conn: aiosqlite.Connection,
    task_id: str,
    job_id: str,
    state: str,
) -> None:
    """Insert a task record with the given state."""
    now = datetime.now(timezone.utc).isoformat()
    await conn.execute(
        """
        INSERT INTO tasks (
            task_id, job_id, title, description, agent_type, state,
            priority, retry_count, dependencies, created_at, updated_at, version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            job_id,
            "Test Task",
            "A test task description",
            "coder",
            state,
            5,
            0,
            "[]",
            now,
            now,
            0,
        ),
    )
    await conn.commit()


async def _read_task_state(conn: aiosqlite.Connection, task_id: str) -> str:
    """Read back the state of a task by task_id."""
    async with conn.execute(
        "SELECT state FROM tasks WHERE task_id = ?", (task_id,)
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None, f"Task {task_id!r} not found in database"
    return row[0]


# ---------------------------------------------------------------------------
# Property 5: Task State Persistence Round-Trip
# Validates: Requirements 2.1
# ---------------------------------------------------------------------------


@given(state=st.sampled_from(VALID_TASK_STATES))
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_task_state_persistence_round_trip(state: str) -> None:
    """**Validates: Requirements 2.1**

    For any valid task state, writing a task record to SQLite and reading it
    back SHALL produce a record with the identical state value.
    """
    task_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    async with aiosqlite.connect(":memory:") as conn:
        await _create_schema(conn)
        await _insert_job(conn, job_id)
        await _insert_task(conn, task_id, job_id, state)

        retrieved_state = await _read_task_state(conn, task_id)

    assert retrieved_state == state, (
        f"State round-trip failed: wrote {state!r}, read back {retrieved_state!r}"
    )
