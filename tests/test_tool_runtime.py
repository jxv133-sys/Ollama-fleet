"""Tests for the tool runtime dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from ollama_fleet.tools.runtime import ToolRuntime
from ollama_fleet.workspace.manager import WorkspaceManager


@pytest.mark.asyncio
async def test_tool_runtime_file_read_write(tmp_path: Path) -> None:
    manager = WorkspaceManager.create_workspace(
        job_id="job-runtime",
        goal="Test runtime tool dispatcher",
        config={},
        base_path=tmp_path,
    )
    runtime = ToolRuntime(str(manager.root), tools_config={"git_enabled": False})

    result = await runtime.invoke(
        "file_tools",
        {"action": "write_file", "path": "src/runtime.txt", "content": "runtime"},
        task_id="task-1",
    )
    assert result is None or result.get("error_type") is None

    read_result = await runtime.invoke(
        "file_tools",
        {"action": "read_file", "path": "src/runtime.txt"},
        task_id="task-1",
    )
    assert read_result == "runtime"


@pytest.mark.asyncio
async def test_tool_runtime_unknown_tool(tmp_path: Path) -> None:
    runtime = ToolRuntime(str(tmp_path), tools_config={"git_enabled": False})
    result = await runtime.invoke("not_a_tool", {"action": "noop"}, task_id="task-2")
    assert result["error_type"] == "tool_not_found"
