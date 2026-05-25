from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

SHELL_METACHARACTERS = set(";&|><$`(){}[]*?!~")


class ShellToolError(Exception):
    pass


class ShellTools:
    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    def _validate_args(self, args: list[str]) -> dict[str, Any] | None:
        for arg in args:
            if any(ch in arg for ch in SHELL_METACHARACTERS):
                return {
                    "error_type": "validation_error",
                    "message": f"Argument {shlex.quote(arg)} contains forbidden metacharacters",
                }
        return None

    def run_command(self, args: list[str], timeout: float) -> dict[str, Any]:
        validation_error = self._validate_args(args)
        if validation_error is not None:
            return validation_error

        try:
            completed = subprocess.run(
                args,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return {
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "exit_code": completed.returncode,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "error_type": "timeout",
                "timeout_seconds": timeout,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            }

    def run_tests(self, timeout: float = 120.0) -> dict[str, Any]:
        result = self.run_command(["python3", "-m", "pytest", "-q"], timeout)
        if "error_type" in result:
            return result

        stdout = result["stdout"]
        stderr = result["stderr"]
        passed = 0
        failed = 0
        failures: list[dict[str, str]] = []

        for line in stdout.splitlines():
            if line.strip().endswith(" passed"):
                passed += 1
            if line.strip().endswith(" failed"):
                failed_count = line.strip().split()[0]
                if failed_count.isdigit():
                    failed += int(failed_count)

        if failed > 0:
            failures.append({"message": stderr.strip()})

        return {
            "pass_count": passed,
            "fail_count": failed,
            "stdout": stdout,
            "stderr": stderr,
            "failures": failures,
        }
