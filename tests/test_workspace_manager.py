"""Tests for the workspace manager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ollama_fleet.workspace.manager import WorkspaceManager, PathTraversalError


def test_create_workspace_creates_directories(tmp_path: Path) -> None:
    manager = WorkspaceManager.create_workspace(
        job_id="job-123",
        goal="Build a minimal pipeline",
        config={"foo": "bar"},
        base_path=tmp_path,
    )

    assert manager.root.exists()
    assert (manager.root / "src").is_dir()
    assert (manager.root / "logs").is_dir()
    metadata = json.loads((manager.root / "metadata" / "job.json").read_text(encoding="utf-8"))
    assert metadata["job_id"] == "job-123"
    assert metadata["goal"] == "Build a minimal pipeline"


def test_write_file_atomic_and_path_validation(tmp_path: Path) -> None:
    manager = WorkspaceManager.create_workspace(
        job_id="job-456",
        goal="Test atomic write",
        config={},
        base_path=tmp_path,
    )

    manager.write_file("src/hello.txt", "hello world")
    assert (manager.root / "src" / "hello.txt").read_text(encoding="utf-8") == "hello world"

    with pytest.raises(PathTraversalError):
        manager.write_file("../outside.txt", "nope")
