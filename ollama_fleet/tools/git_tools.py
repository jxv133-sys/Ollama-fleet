from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class GitTools:
    def __init__(self, workspace_root: str | Path, git_enabled: bool = True) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.git_enabled = git_enabled

    def _unavailable(self) -> dict[str, Any]:
        return {"error_type": "tool_unavailable", "message": "Git support is disabled"}

    def git_diff(self) -> dict[str, Any]:
        if not self.git_enabled:
            return self._unavailable()

        try:
            completed = subprocess.run(
                ["git", "diff", "--no-ext-diff", "--stat"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                check=False,
            )
            return {
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "exit_code": completed.returncode,
            }
        except FileNotFoundError:
            return {"error_type": "tool_unavailable", "message": "Git binary not found"}

    def git_commit(self, message: str) -> dict[str, Any]:
        if not self.git_enabled:
            return self._unavailable()

        try:
            add = subprocess.run(
                ["git", "add", "-A"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if add.returncode != 0:
                return {
                    "error_type": "git_error",
                    "message": add.stderr.strip(),
                }

            commit = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                check=False,
            )
            return {
                "stdout": commit.stdout,
                "stderr": commit.stderr,
                "exit_code": commit.returncode,
            }
        except FileNotFoundError:
            return {"error_type": "tool_unavailable", "message": "Git binary not found"}
