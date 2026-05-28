from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WorkspaceError(Exception):
    pass


class WorkspaceCreationError(WorkspaceError):
    pass


class AtomicWriteError(WorkspaceError):
    pass


class PathTraversalError(WorkspaceError):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Path traversal outside workspace root: {path}")

    def to_dict(self) -> dict[str, str]:
        return {"error_type": "path_traversal", "path": self.path}


class WorkspaceManager:
    """Manage a per-job workspace with atomic file writes and safe paths."""

    SUBDIRS = [
        "src",
        "tests",
        "logs",
        "agent_outputs",
        "validation",
        "summaries",
        "metadata",
    ]

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    @classmethod
    def create_workspace(
        cls,
        job_id: str,
        goal: str,
        config: dict[str, Any],
        base_path: str | Path = "./workspaces",
    ) -> "WorkspaceManager":
        root = Path(base_path).resolve() / job_id
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise WorkspaceCreationError(
                f"Workspace for job_id '{job_id}' already exists"
            ) from exc
        manager = cls(root)
        for name in cls.SUBDIRS:
            (root / name).mkdir(parents=True, exist_ok=True)

        metadata = {
            "job_id": job_id,
            "goal": goal,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
        }

        try:
            manager.write_file("metadata/job.json", json.dumps(metadata, indent=2))
        except WorkspaceError as exc:
            raise WorkspaceCreationError(
                f"Failed to create workspace metadata for job_id '{job_id}': {exc}"
            ) from exc

        return manager

    def _validate_path(self, rel_path: str | Path) -> Path:
        candidate = Path(rel_path)
        if candidate.is_absolute():
            candidate = candidate.relative_to(candidate.anchor)

        resolved = (self.root / candidate).resolve()
        root_resolved = self.root.resolve()
        if root_resolved not in (*resolved.parents, resolved):
            raise PathTraversalError(str(rel_path))
        return resolved

    def write_file(self, rel_path: str | Path, content: str) -> None:
        target_path = self._validate_path(rel_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=target_path.parent,
                prefix=f".{target_path.name}.",
            ) as fh:
                temp_file = Path(fh.name)
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp_file, target_path)
        except Exception as exc:
            if temp_file is not None and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise AtomicWriteError(
                f"Failed to write {rel_path}: {exc}"
            ) from exc

    def append_execution_history(self, event: dict[str, Any]) -> None:
        log_path = self._validate_path("logs/execution_history.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({**event, "timestamp": datetime.now(timezone.utc).isoformat()})
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            raise WorkspaceError(f"Failed to append execution history: {exc}") from exc
