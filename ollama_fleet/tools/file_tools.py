from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import Any

from ollama_fleet.workspace.manager import PathTraversalError, WorkspaceManager


class FileTools:
    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    def _validate_path(self, rel_path: str | Path) -> Path:
        manager = WorkspaceManager(self.workspace_root)
        return manager._validate_path(rel_path)

    def read_file(self, rel_path: str | Path) -> dict[str, Any] | str:
        try:
            target = self._validate_path(rel_path)
        except PathTraversalError as exc:
            return exc.to_dict()

        if not target.exists():
            return {"error_type": "file_not_found", "path": str(rel_path)}

        try:
            return target.read_text(encoding="utf-8")
        except PermissionError:
            return {"error_type": "permission_denied", "path": str(rel_path)}

    def write_file(self, rel_path: str | Path, content: str) -> dict[str, Any] | None:
        try:
            manager = WorkspaceManager(self.workspace_root)
            manager.write_file(rel_path, content)
            return None
        except PathTraversalError as exc:
            return exc.to_dict()
        except Exception as exc:
            return {"error_type": "write_failed", "message": str(exc)}

    def list_files(self, rel_path: str | Path = ".") -> dict[str, Any] | list[str]:
        try:
            target = self._validate_path(rel_path)
        except PathTraversalError as exc:
            return exc.to_dict()

        if not target.exists():
            return {"error_type": "file_not_found", "path": str(rel_path)}
        if not target.is_dir():
            return {"error_type": "not_a_directory", "path": str(rel_path)}

        results: list[str] = []
        for path in sorted(target.rglob("*")):
            if path.is_file():
                results.append(str(path.relative_to(self.workspace_root)))
        return results

    def search_code(self, rel_path: str | Path, pattern: str) -> dict[str, Any] | list[dict[str, Any]]:
        try:
            target = self._validate_path(rel_path)
        except PathTraversalError as exc:
            return exc.to_dict()

        if not target.exists():
            return {"error_type": "file_not_found", "path": str(rel_path)}
        if not target.is_dir():
            return {"error_type": "not_a_directory", "path": str(rel_path)}

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return {"error_type": "invalid_pattern", "message": str(exc)}

        matches: list[dict[str, Any]] = []
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(
                        {
                            "path": str(path.relative_to(self.workspace_root)),
                            "line_number": line_number,
                            "line": line,
                        }
                    )
        return matches
