from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ollama_fleet.db.database import Database


@dataclass
class ScheduledTask:
    task_id: str
    job_id: str
    title: str
    description: str
    agent_type: str
    state: str
    priority: int
    retry_count: int
    dependencies: list[str]
    created_at: str
    updated_at: str
    version: int


class TaskScheduler:
    """Scheduler helper for task queue and atomic state transitions."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def enqueue_tasks(self, tasks: list[dict[str, Any]]) -> None:
        for task in tasks:
            await self._db.execute_and_commit(
                """
                INSERT INTO tasks (
                    task_id, job_id, title, description, agent_type, state,
                    priority, retry_count, dependencies, created_at, updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    task["task_id"],
                    task["job_id"],
                    task["title"],
                    task["description"],
                    task["agent_type"],
                    task.get("state", "pending"),
                    task.get("priority", 5),
                    task.get("retry_count", 0),
                    json.dumps(task.get("dependencies", [])),
                    task.get("created_at"),
                    task.get("updated_at"),
                    task.get("version", 0),
                ],
            )

    async def get_ready_tasks(self, job_id: str) -> list[ScheduledTask]:
        rows = await self._db.fetchall(
            "SELECT task_id, job_id, title, description, agent_type, state, priority, retry_count, dependencies, created_at, updated_at, version FROM tasks WHERE job_id = ? AND state = 'pending'",
            (job_id,),
        )
        return [
            ScheduledTask(
                task_id=row[0],
                job_id=row[1],
                title=row[2],
                description=row[3],
                agent_type=row[4],
                state=row[5],
                priority=row[6],
                retry_count=row[7],
                dependencies=json.loads(row[8]),
                created_at=row[9],
                updated_at=row[10],
                version=row[11],
            )
            for row in rows
        ]

    async def transition(
        self,
        task_id: str,
        new_state: str,
        reason: str | None = None,
    ) -> bool:
        row = await self._db.fetchone(
            "SELECT state, version FROM tasks WHERE task_id = ?",
            (task_id,),
        )
        if row is None:
            return False

        current_version = row[1]
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute("BEGIN IMMEDIATE")
        cursor = await self._db.execute(
            """
            UPDATE tasks
            SET state = ?, version = version + 1, updated_at = ?
            WHERE task_id = ? AND version = ?
            """,
            (new_state, now, task_id, current_version),
        )
        await self._db.commit()
        return cursor.rowcount == 1

    async def increment_retry(self, task_id: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute("BEGIN IMMEDIATE")
        await self._db.execute(
            "UPDATE tasks SET retry_count = retry_count + 1, updated_at = ? WHERE task_id = ?",
            (now, task_id),
        )
        await self._db.commit()
        row = await self._db.fetchone(
            "SELECT retry_count FROM tasks WHERE task_id = ?",
            (task_id,),
        )
        return int(row[0]) if row is not None else 0

    async def count_active_tasks(self, job_id: str) -> int:
        row = await self._db.fetchone(
            "SELECT COUNT(*) FROM tasks WHERE job_id = ? AND state IN ('pending', 'running')",
            (job_id,),
        )
        return int(row[0]) if row is not None else 0

    async def cancel_task(self, task_id: str) -> bool:
        await self._db.execute("BEGIN IMMEDIATE")
        cursor = await self._db.execute(
            """
            UPDATE tasks
            SET state = 'cancelled', updated_at = ?
            WHERE task_id = ? AND state IN ('pending', 'blocked')
            """,
            (datetime.now(timezone.utc).isoformat(), task_id),
        )
        await self._db.commit()
        return cursor.rowcount == 1
