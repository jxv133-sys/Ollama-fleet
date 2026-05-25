from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ollama_fleet.db.database import Database
from ollama_fleet.workspace.manager import WorkspaceManager


class EscalationManager:
    def __init__(self, db: Database, workspace: WorkspaceManager) -> None:
        self._db = db
        self._workspace = workspace

    async def write_escalation(
        self,
        task_id: str,
        job_id: str,
        reason: str,
        retry_count: int,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "task_id": task_id,
            "job_id": job_id,
            "reason": reason,
            "retry_count": retry_count,
            "timestamp": timestamp,
        }
        self._append_metadata(record)
        await self._db.execute_and_commit(
            """
            INSERT INTO escalations (task_id, job_id, reason, retry_count, timestamp, dismissed)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            [task_id, job_id, reason, retry_count, timestamp],
        )

    def _append_metadata(self, record: dict[str, Any]) -> None:
        target = self._workspace.root / "metadata" / "escalations.json"
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
        else:
            existing = []
        existing.append(record)
        target.write_text(json.dumps(existing, indent=2), encoding="utf-8")
