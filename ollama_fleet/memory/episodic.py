from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ollama_fleet.db.database import Database


@dataclass
class EpisodicEntry:
    job_id: str
    task_id: str
    agent_type: str
    outcome: str
    files_modified: list[str]
    summary_text: str
    timestamp: str


class EpisodicMemory:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, entry: EpisodicEntry) -> None:
        await self._db.execute_and_commit(
            """
            INSERT INTO episodic_memory (
                job_id, task_id, agent_type, outcome,
                files_modified, summary_text, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                entry.job_id,
                entry.task_id,
                entry.agent_type,
                entry.outcome,
                json.dumps(entry.files_modified),
                entry.summary_text,
                entry.timestamp,
            ],
        )

    async def get_recent(self, job_id: str, limit: int = 5) -> list[EpisodicEntry]:
        rows = await self._db.fetchall(
            """
            SELECT job_id, task_id, agent_type, outcome, files_modified, summary_text, timestamp
            FROM episodic_memory
            WHERE job_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [job_id, limit],
        )
        return [
            EpisodicEntry(
                job_id=row[0],
                task_id=row[1],
                agent_type=row[2],
                outcome=row[3],
                files_modified=json.loads(row[4]),
                summary_text=row[5],
                timestamp=row[6],
            )
            for row in rows
        ]
