from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ollama_fleet.db.database import Database


@dataclass
class LongTermEntry:
    job_id: str
    summary_text: str
    source: str
    metadata: dict[str, Any]
    inserted_at: str


class LongTermMemory:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, entry: LongTermEntry) -> None:
        await self._db.execute_and_commit(
            """
            INSERT INTO long_term_memory (
                job_id, summary_text, source, metadata_json, inserted_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                entry.job_id,
                entry.summary_text,
                entry.source,
                json.dumps(entry.metadata),
                entry.inserted_at,
            ],
        )

    async def search(self, job_id: str, query: str) -> list[LongTermEntry]:
        rows = await self._db.fetchall(
            """
            SELECT job_id, summary_text, source, metadata_json, inserted_at
            FROM long_term_memory
            WHERE job_id = ? AND summary_text LIKE ? COLLATE NOCASE
            ORDER BY inserted_at DESC
            """,
            [job_id, f"%{query}%"],
        )
        return [
            LongTermEntry(
                job_id=row[0],
                summary_text=row[1],
                source=row[2],
                metadata=json.loads(row[3]),
                inserted_at=row[4],
            )
            for row in rows
        ]
