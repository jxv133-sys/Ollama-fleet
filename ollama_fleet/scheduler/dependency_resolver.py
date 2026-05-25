from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ollama_fleet.db.database import Database


class DependencyResolver:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def resolve(self, job_id: str) -> None:
        blocked_tasks = await self._db.fetchall(
            "SELECT task_id, dependencies FROM tasks WHERE job_id = ? AND state = 'blocked'",
            (job_id,),
        )

        for task_id, raw_dependencies in blocked_tasks:
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
                    continue
                dep_state = dep[0]
                dep_states.append(dep_state)
                if dep_state in ("failed", "cancelled"):
                    failed_dependency = dependency_id
                    break

            if failed_dependency is not None:
                await self._db.execute("BEGIN IMMEDIATE")
                await self._db.execute(
                    """
                    UPDATE tasks
                    SET state = 'failed', failure_reason = ?, updated_at = ?, version = version + 1
                    WHERE task_id = ? AND state = 'blocked'
                    """,
                    (
                        f"dependency {failed_dependency} failed or cancelled",
                        datetime.now(timezone.utc).isoformat(),
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
                    (datetime.now(timezone.utc).isoformat(), task_id),
                )
                await self._db.commit()
