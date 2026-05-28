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
            state = task.get("state", "pending")
            if task.get("dependencies"):
                state = "blocked"
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
                    state,
                    task.get("priority", 5),
                    task.get("retry_count", 0),
                    json.dumps(task.get("dependencies", [])),
                    task.get("created_at"),
                    task.get("updated_at"),
                    task.get("version", 0),
                ],
            )

    async def get_ready_tasks(self, job_id: str) -> list[ScheduledTask]:
        await self.resolve_dependencies(job_id)
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

    async def resolve_dependencies(self, job_id: str) -> None:
        blocked = await self._db.fetchall(
            "SELECT task_id, dependencies FROM tasks WHERE job_id = ? AND state = 'blocked'",
            (job_id,),
        )
        for task_id, raw_dependencies in blocked:
            dependencies = json.loads(raw_dependencies or "[]")
            if not dependencies:
                continue

            dep_states: list[str] = []
            failed_dependency: str | None = None
            for dependency_id in dependencies:
                dep = await self._db.fetchone(
                    "SELECT state FROM tasks WHERE task_id = ?",
                    (dependency_id,),
                )
                if dep is None:
                    failed_dependency = dependency_id
                    break
                dep_state = dep[0]
                dep_states.append(dep_state)
                if dep_state in ("failed", "cancelled"):
                    failed_dependency = dependency_id
                    break

            now = datetime.now(timezone.utc).isoformat()
            if failed_dependency is not None:
                await self._db.execute("BEGIN IMMEDIATE")
                await self._db.execute(
                    """
                    UPDATE tasks
                    SET state = 'failed', failure_reason = ?, updated_at = ?, version = version + 1
                    WHERE task_id = ? AND state = 'blocked'
                    """,
                    (
                        f"dependency {failed_dependency} missing, failed, or cancelled",
                        now,
                        task_id,
                    ),
                )
                await self._db.commit()
                continue

            if dep_states and all(state == "completed" for state in dep_states):
                await self._db.execute("BEGIN IMMEDIATE")
                await self._db.execute(
                    """
                    UPDATE tasks
                    SET state = 'pending', updated_at = ?, version = version + 1
                    WHERE task_id = ? AND state = 'blocked'
                    """,
                    (now, task_id),
                )
                await self._db.commit()

    async def recover_stalled_tasks(self, job_id: str, timeout_seconds: float) -> None:
        rows = await self._db.fetchall(
            "SELECT task_id, updated_at FROM tasks WHERE job_id = ? AND state = 'running'",
            (job_id,),
        )
        cutoff = datetime.now(timezone.utc).timestamp() - timeout_seconds
        for task_id, updated_at in rows:
            try:
                updated_ts = datetime.fromisoformat(updated_at).timestamp()
            except ValueError:
                updated_ts = 0.0
            if updated_ts < cutoff:
                await self._db.execute("BEGIN IMMEDIATE")
                await self._db.execute(
                    """
                    UPDATE tasks
                    SET state = 'pending', dispatched_at = NULL, updated_at = ?, version = version + 1
                    WHERE task_id = ? AND state = 'running'
                    """,
                    (datetime.now(timezone.utc).isoformat(), task_id),
                )
                await self._db.commit()

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
        set_clauses = ["state = ?", "version = version + 1", "updated_at = ?"]
        params: list[str] = [new_state, now]

        if new_state == "running":
            set_clauses.append("dispatched_at = ?")
            params.append(now)
        else:
            set_clauses.append("dispatched_at = dispatched_at")

        if new_state in ("completed", "failed", "cancelled"):
            set_clauses.append("completed_at = ?")
            params.append(now)
        else:
            set_clauses.append("completed_at = completed_at")

        if reason is not None:
            set_clauses.append("failure_reason = ?")
            params.append(reason)
        else:
            set_clauses.append("failure_reason = failure_reason")

        params.extend([task_id, current_version])
        sql = f"""
            UPDATE tasks
            SET {', '.join(set_clauses)}
            WHERE task_id = ? AND version = ?
            """
        cursor = await self._db.execute(sql, params)
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

    async def count_failed_tasks(self, job_id: str) -> int:
        row = await self._db.fetchone(
            "SELECT COUNT(*) FROM tasks WHERE job_id = ? AND state = 'failed'",
            (job_id,),
        )
        return int(row[0]) if row is not None else 0

    async def count_completed_tasks_by_agent(self, job_id: str, agent_type: str) -> int:
        row = await self._db.fetchone(
            "SELECT COUNT(*) FROM tasks WHERE job_id = ? AND agent_type = ? AND state = 'completed'",
            (job_id, agent_type),
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
