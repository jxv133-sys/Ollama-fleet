from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ollama_fleet.workspace.manager import WorkspaceManager


@dataclass
class SyntaxValidationError:
    file_path: str
    message: str
    line: int | None
    column: int | None


@dataclass
class LintIssue:
    file_path: str
    line_number: int
    code: str
    message: str
    severity: str


@dataclass
class ValidationResult:
    syntax_ok: bool
    lint_results: list[LintIssue]
    linter_available: bool
    timestamp: str


class ValidationLayer:
    def validate(
        self,
        modified_files: list[str],
        workspace: WorkspaceManager,
    ) -> ValidationResult:
        timestamp = datetime.now(timezone.utc).isoformat()
        syntax_ok = True
        lint_results: list[LintIssue] = []
        linter_available = True
        syntax_errors: list[SyntaxValidationError] = []

        for rel_path in modified_files:
            target = workspace.root / rel_path
            if not target.exists() or not target.is_file():
                continue

            if target.suffix == ".py":
                source = target.read_text(encoding="utf-8")
                try:
                    ast.parse(source, filename=str(target))
                except SyntaxError as exc:
                    syntax_ok = False
                    syntax_errors.append(
                        SyntaxValidationError(
                            file_path=str(rel_path),
                            message=str(exc),
                            line=exc.lineno,
                            column=exc.offset,
                        )
                    )

        command = ["ruff", "check", "--format", "json"] + modified_files
        try:
            proc = subprocess.run(
                command,
                cwd=workspace.root,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            linter_available = False
            proc = None

        if linter_available and proc is not None and proc.stdout:
            try:
                parsed = json.loads(proc.stdout)
                # ruff --format json outputs a flat list of objects with keys:
                # filename, row, col, code, message, fix, url
                for item in parsed:
                    lint_results.append(
                        LintIssue(
                            file_path=item.get("filename", ""),
                            line_number=item.get("row", 0),
                            code=item.get("code", ""),
                            message=item.get("message", ""),
                            severity="warning",
                        )
                    )
            except json.JSONDecodeError:
                linter_available = False

        result = ValidationResult(
            syntax_ok=syntax_ok,
            lint_results=lint_results,
            linter_available=linter_available,
            timestamp=timestamp,
        )
        self._write_result(workspace, result, syntax_errors)
        return result

    def _write_result(
        self,
        workspace: WorkspaceManager,
        result: ValidationResult,
        syntax_errors: list[SyntaxValidationError],
    ) -> None:
        filename = workspace.root / "validation" / f"validation_{result.timestamp}.json"
        payload: dict[str, Any] = {
            "syntax_ok": result.syntax_ok,
            "linter_available": result.linter_available,
            "timestamp": result.timestamp,
            "lint_results": [vars(issue) for issue in result.lint_results],
            "syntax_errors": [vars(error) for error in syntax_errors],
        }
        filename.write_text(json.dumps(payload, indent=2), encoding="utf-8")
