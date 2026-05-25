from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ollama_fleet.db.database import Database


@dataclass
class JobRecord:
    job_id: str
    goal: str
    state: str
    created_at: str
    updated_at: str
    config: dict[str, Any]
    workspace_path: str
    version: int


class JobManager:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def create_job(
        self,
        goal: str,
        config: dict[str, Any],
        workspace_path: str,
        job_id: str | None = None,
    ) -> str:
        job_id = job_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute_and_commit(
            """
            INSERT INTO jobs (
                job_id, goal, state, created_at, updated_at, config_json, workspace_path, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                job_id,
                goal,
                "submitted",
                now,
                now,
                json.dumps(config),
                workspace_path,
                0,
            ],
        )
        return job_id

    async def get_job(self, job_id: str) -> JobRecord | None:
        row = await self._db.fetchone(
            "SELECT job_id, goal, state, created_at, updated_at, config_json, workspace_path, version FROM jobs WHERE job_id = ?",
            (job_id,),
        )
        if row is None:
            return None
        return JobRecord(
            job_id=row[0],
            goal=row[1],
            state=row[2],
            created_at=row[3],
            updated_at=row[4],
            config=json.loads(row[5]),
            workspace_path=row[6],
            version=row[7],
        )

    async def update_job_state(self, job_id: str, new_state: str) -> bool:
        await self._db.execute("BEGIN IMMEDIATE")
        cursor = await self._db.execute(
            "UPDATE jobs SET state = ?, updated_at = ?, version = version + 1 WHERE job_id = ?",
            (new_state, datetime.now(timezone.utc).isoformat(), job_id),
        )
        await self._db.commit()
        return cursor.rowcount == 1

    async def list_jobs_by_state(self, state: str) -> list[JobRecord]:
        rows = await self._db.fetchall(
            "SELECT job_id, goal, state, created_at, updated_at, config_json, workspace_path, version FROM jobs WHERE state = ?",
            (state,),
        )
        return [
            JobRecord(
                job_id=row[0],
                goal=row[1],
                state=row[2],
                created_at=row[3],
                updated_at=row[4],
                config=json.loads(row[5]),
                workspace_path=row[6],
                version=row[7],
            )
            for row in rows
        ]
