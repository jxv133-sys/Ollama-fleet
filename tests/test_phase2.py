"""Phase 2 tests for validation, critic prompts, and escalation handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.escalation import EscalationManager
from ollama_fleet.validation.validator import ValidationLayer
from ollama_fleet.workspace.manager import WorkspaceManager


def test_validation_syntax_failure_reports_false(tmp_path: Path) -> None:
    workspace = WorkspaceManager.create_workspace(
        job_id="job-validation",
        goal="Validate syntax failure",
        config={},
        base_path=tmp_path,
    )
    bad_path = workspace.root / "src" / "bad.py"
    bad_path.write_text("def broken(:\n    pass\n", encoding="utf-8")

    result = ValidationLayer().validate(["src/bad.py"], workspace)

    assert result.syntax_ok is False
    assert isinstance(result.lint_results, list)
    assert result.timestamp


def test_critic_prompt_includes_issue_details() -> None:
    from ollama_fleet.agents.critic import build_critic_prompt

    issues = [
        {
            "file_path": "src/foo.py",
            "line_number": 10,
            "severity": "major",
            "description": "Missing return statement.",
            "suggested_fix": "Add an explicit return value.",
        }
    ]
    prompt = build_critic_prompt(
        modified_files=["src/foo.py"],
        file_contents={"src/foo.py": "def foo():\n    pass\n"},
        lint_results=[],
        critic_issues=issues,
    )

    assert "Missing return statement." in prompt
    assert "src/foo.py" in prompt
    assert "Add an explicit return value." in prompt


@pytest.mark.asyncio
async def test_escalation_record_field_completeness(tmp_path: Path) -> None:
    async with Database(":memory:") as db:
        schema_path = (
            Path(__file__).parent.parent
            / "ollama_fleet"
            / "db"
            / "migrations"
            / "001_initial.sql"
        )
        sql = schema_path.read_text(encoding="utf-8")
        await db._conn.executescript(sql)
        workspace = WorkspaceManager.create_workspace(
            job_id="job-escalation",
            goal="Write escalation record",
            config={},
            base_path=tmp_path,
        )

        manager = EscalationManager(db, workspace)
        await manager.write_escalation(
            task_id="task-1",
            job_id="job-escalation",
            reason="critique loop exceeded",
            retry_count=3,
        )

        metadata_file = workspace.root / "metadata" / "escalations.json"
        assert metadata_file.exists()
        data = json.loads(metadata_file.read_text(encoding="utf-8"))
        assert data[0]["task_id"] == "task-1"
        assert data[0]["job_id"] == "job-escalation"
        assert data[0]["reason"] == "critique loop exceeded"
        assert data[0]["retry_count"] == 3
        assert data[0]["timestamp"]

        cursor = await db.connection.execute(
            "SELECT task_id, job_id, reason, retry_count, timestamp FROM escalations"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "task-1"
        assert row[1] == "job-escalation"
        assert row[2] == "critique loop exceeded"
        assert row[3] == 3
        assert row[4]
