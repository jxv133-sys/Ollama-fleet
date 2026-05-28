"""Tests for file tools path validation and search."""

from __future__ import annotations

from pathlib import Path

from ollama_fleet.tools.file_tools import FileTools
from ollama_fleet.workspace.manager import WorkspaceManager


def test_file_tools_write_read_list_search(tmp_path: Path) -> None:
    manager = WorkspaceManager.create_workspace(
        job_id="job-file-tools",
        goal="Validate file tools",
        config={},
        base_path=tmp_path,
    )
    tools = FileTools(manager.root)

    error = tools.write_file("src/notes.txt", "hello world")
    assert error is None

    content = tools.read_file("src/notes.txt")
    assert content == "hello world"

    listing = tools.list_files("src")
    assert "src/notes.txt" in listing

    matches = tools.search_code("src", "hello")
    assert matches[0]["path"] == "src/notes.txt"
    assert matches[0]["line_number"] == 1
