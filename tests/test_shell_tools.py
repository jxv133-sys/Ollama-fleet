"""Tests for shell tools command execution and validation."""

from __future__ import annotations

from pathlib import Path

from ollama_fleet.tools.shell_tools import ShellTools


def test_run_command_rejects_metacharacters(tmp_path: Path) -> None:
    tools = ShellTools(tmp_path)
    result = tools.run_command(["echo", "hello; rm -rf /"], timeout=5.0)
    assert result["error_type"] == "validation_error"


def test_run_command_executes_simple_command(tmp_path: Path) -> None:
    tools = ShellTools(tmp_path)
    result = tools.run_command(["echo", "hi"], timeout=5.0)
    assert result["exit_code"] == 0
    assert "hi" in result["stdout"]
