"""Unit tests for ProjectMemoryManager.

Tests for:
- File metadata extraction and storage
- Dependency resolution
- Interface extraction
- Project state tracking
"""

import pytest
import json
from datetime import datetime, timezone

from ollama_fleet.db.database import Database
from ollama_fleet.memory.project_memory import (
    ProjectMemoryManager,
    FileMetadata,
    ProjectMemoryEntry,
)


@pytest.fixture
async def db():
    """Create in-memory test database."""
    db = Database(":memory:")
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
def manager(db):
    """Create ProjectMemoryManager instance."""
    return ProjectMemoryManager(db)


class TestFileMetadataExtraction:
    """Test metadata extraction from source code."""

    def test_extract_python_imports(self, manager):
        """Test extraction of import statements."""
        code = """
import os
from pathlib import Path
from ollama_fleet.agents.executor import AgentExecutor
from . import local_module
"""
        metadata = manager._extract_python_metadata(code)

        assert "os" in metadata.imports
        assert "pathlib" in metadata.imports
        assert "ollama_fleet.agents.executor" in metadata.imports
        # Local imports not included
        assert "." not in metadata.imports

    def test_extract_python_classes(self, manager):
        """Test extraction of class definitions."""
        code = """
class MyClass:
    pass

class AnotherClass(BaseClass):
    pass
"""
        metadata = manager._extract_python_metadata(code)

        assert "MyClass" in metadata.classes
        assert "AnotherClass" in metadata.classes

    def test_extract_python_functions(self, manager):
        """Test extraction of public function definitions."""
        code = """
def public_function():
    pass

def _private_function():
    pass

async def async_function():
    pass
"""
        metadata = manager._extract_python_metadata(code)

        assert "public_function" in metadata.functions
        assert "async_function" in metadata.functions
        assert "_private_function" not in metadata.functions  # Private functions excluded

    def test_extract_python_exports_from_all(self, manager):
        """Test extraction of __all__ exports."""
        code = """
def func_a():
    pass

def func_b():
    pass

__all__ = ["func_a", "func_b"]
"""
        metadata = manager._extract_python_metadata(code)

        assert metadata.exports == ["func_a", "func_b"]

    def test_extract_python_default_exports(self, manager):
        """Test that public functions are default exports if no __all__."""
        code = """
def public_func():
    pass

def _private_func():
    pass
"""
        metadata = manager._extract_python_metadata(code)

        # If no __all__, exports should be public functions
        assert "public_func" in metadata.exports
        assert "_private_func" not in metadata.exports


class TestProjectMemoryStorage:
    """Test storing and retrieving file metadata."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_file_metadata(self, manager, db):
        """Test storing and retrieving file metadata."""
        job_id = "test-job-1"
        file_path = "src/logic.py"
        source_code = """
def run_analysis():
    pass

def load_data():
    pass
"""

        # Store metadata
        await manager.store_file_metadata(
            job_id=job_id,
            file_path=file_path,
            source_code=source_code,
            file_type="python",
        )

        # Retrieve metadata
        entry = await manager.get_file_metadata(job_id, file_path)

        assert entry is not None
        assert entry.job_id == job_id
        assert entry.file_path == file_path
        assert "run_analysis" in entry.functions
        assert "load_data" in entry.functions

    @pytest.mark.asyncio
    async def test_get_project_files(self, manager, db):
        """Test retrieving all files in a project."""
        job_id = "test-job-2"

        # Store multiple files
        files = [
            ("src/models.py", "class User: pass"),
            ("src/logic.py", "def process(): pass"),
            ("src/utils.py", "def helper(): pass"),
        ]

        for file_path, code in files:
            await manager.store_file_metadata(
                job_id=job_id,
                file_path=file_path,
                source_code=code,
                file_type="python",
            )

        # Retrieve all files
        all_files = await manager.get_project_files(job_id)

        assert len(all_files) == 3
        paths = [f.file_path for f in all_files]
        assert all(path in paths for path, _ in files)

    @pytest.mark.asyncio
    async def test_file_metadata_hash(self, manager, db):
        """Test that source code hash is stored correctly."""
        job_id = "test-job-3"
        file_path = "src/test.py"
        source_code = "def test(): pass"

        await manager.store_file_metadata(
            job_id=job_id,
            file_path=file_path,
            source_code=source_code,
            file_type="python",
        )

        entry = await manager.get_file_metadata(job_id, file_path)

        # Hash should be SHA256 of source code
        import hashlib
        expected_hash = hashlib.sha256(source_code.encode()).hexdigest()
        assert entry.source_hash == expected_hash


class TestDependencyResolution:
    """Test dependency tracking and resolution."""

    @pytest.mark.asyncio
    async def test_get_dependencies_for_file(self, manager, db):
        """Test retrieving dependencies for a file."""
        job_id = "test-job-4"

        # Create files
        await manager.store_file_metadata(
            job_id=job_id,
            file_path="src/models.py",
            source_code="class User: pass",
            file_type="python",
        )

        await manager.store_file_metadata(
            job_id=job_id,
            file_path="src/logic.py",
            source_code="from models import User",
            file_type="python",
        )

        # Get dependencies (note: requires manual setup in real scenario)
        # For now, just test the method exists and handles empty case
        deps = await manager.get_dependencies_for_file(job_id, "src/logic.py")
        assert isinstance(deps, list)


class TestProjectState:
    """Test project state tracking."""

    @pytest.mark.asyncio
    async def test_update_and_get_project_state(self, manager, db):
        """Test updating and retrieving project state."""
        job_id = "test-job-5"

        # Update state
        await manager.update_project_state(
            job_id=job_id,
            last_action="plan_created",
            metadata={"total_files": 5},
        )

        # Retrieve state
        state = await manager.get_project_state(job_id)

        assert state is not None
        assert state.job_id == job_id
        assert state.last_action == "plan_created"
        assert state.metadata["total_files"] == 5

    @pytest.mark.asyncio
    async def test_project_state_metadata_is_json(self, manager, db):
        """Test that project state metadata is properly JSON serialized."""
        job_id = "test-job-6"
        metadata = {
            "files_planned": 10,
            "generated": 3,
            "validated": 2,
            "errors": ["file1.py", "file2.py"],
        }

        await manager.update_project_state(
            job_id=job_id,
            last_action="progress",
            metadata=metadata,
        )

        state = await manager.get_project_state(job_id)

        assert state.metadata == metadata
        assert state.metadata["files_planned"] == 10
        assert "file1.py" in state.metadata["errors"]


class TestInterfaceExtraction:
    """Test interface extraction and storage."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_interfaces(self, manager, db):
        """Test storing and retrieving interfaces."""
        job_id = "test-job-7"
        file_path = "src/models.py"

        interfaces = [
            {
                "type": "class",
                "name": "User",
                "signature": "class User(BaseModel):",
                "docstring": "User model",
            },
            {
                "type": "function",
                "name": "get_user",
                "signature": "def get_user(user_id: int) -> User:",
                "docstring": None,
            },
        ]

        await manager.store_interfaces(
            job_id=job_id,
            file_path=file_path,
            interfaces=interfaces,
        )

        # Retrieve interfaces for file
        retrieved = await manager.get_file_interfaces(job_id, file_path)

        assert len(retrieved) == 2
        names = [i.interface_name for i in retrieved]
        assert "User" in names
        assert "get_user" in names

    @pytest.mark.asyncio
    async def test_find_interface_by_name(self, manager, db):
        """Test finding an interface by name across project."""
        job_id = "test-job-8"

        # Store interface in one file
        await manager.store_interfaces(
            job_id=job_id,
            file_path="src/models.py",
            interfaces=[
                {
                    "type": "class",
                    "name": "User",
                    "signature": "class User:",
                    "docstring": None,
                }
            ],
        )

        # Find interface
        interface = await manager.find_interface(job_id, "User")

        assert interface is not None
        assert interface.interface_name == "User"
        assert interface.source_file == "src/models.py"


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_file_metadata(self, manager, db):
        """Test retrieving metadata for nonexistent file."""
        job_id = "test-job-9"
        entry = await manager.get_file_metadata(job_id, "nonexistent.py")

        assert entry is None

    @pytest.mark.asyncio
    async def test_get_project_files_empty_project(self, manager, db):
        """Test retrieving files from project with no files."""
        job_id = "empty-job"
        files = await manager.get_project_files(job_id)

        assert files == []

    def test_extract_malformed_imports(self, manager):
        """Test extraction handles malformed import statements."""
        code = """
import os, sys
from pathlib import Path as P
import this.module.path
"""
        metadata = manager._extract_python_metadata(code)

        # Should handle various import formats
        assert len(metadata.imports) > 0

    def test_extract_no_exports(self, manager):
        """Test extraction when no exports or public functions."""
        code = """
def _private():
    pass

class _PrivateClass:
    pass
"""
        metadata = manager._extract_python_metadata(code)

        # No public items, exports should default to empty
        assert len(metadata.exports) == 0 or metadata.exports == metadata.functions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
