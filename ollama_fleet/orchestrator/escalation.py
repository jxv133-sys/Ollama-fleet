from __future__ import annotations

import json
import tempfile
import os
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
        """Atomically append an escalation record to the metadata file.
        
        Uses atomic write pattern (temp file + rename) to prevent
        data loss from concurrent writes.
        """
        target = self._workspace.root / "metadata" / "escalations.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        
        existing = []
        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, OSError):
                # If file is corrupted/unreadable, start fresh
                existing = []
        
        existing.append(record)
        
        # Atomic write using temp file + rename
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=target.parent,
                suffix=".json",
                prefix=".escalations_tmp_",
            )
            temp_file.write(json.dumps(existing, indent=2))
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            os.replace(str(temp_file.name), str(target))
        except (OSError, IOError) as exc:
            if temp_file is not None:
                try:
                    temp_file.close()
                except OSError:
                    pass
                try:
                    if Path(temp_file.name).exists():
                        Path(temp_file.name).unlink()
                except OSError:
                    pass
            # Log the error but don't crash - escalation is already in the database
            import logging
            logging.warning(f"Failed to write escalation metadata: {exc}")
