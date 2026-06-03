"""Project Memory Manager - Structured project state index.

Maintains a database index of:
- File paths, exports, imports, classes, functions
- Project interfaces (classes, functions, signatures)
- Project state snapshot (files generated, validated, failed)

This replaces hallucination-prone episodic memory with facts extracted from actual code.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ollama_fleet.db.database import Database

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Structured metadata for a single file."""

    file_path: str
    file_type: str  # 'python', 'json', etc.
    exports: list[str]
    imports: list[str]
    classes: list[str]
    functions: list[str]
    dependencies: list[str]


@dataclass
class ProjectMemoryEntry:
    """Full project memory entry with timestamps."""

    id: int
    job_id: str
    file_path: str
    file_type: str
    exports: list[str]
    imports: list[str]
    classes: list[str]
    functions: list[str]
    dependencies: list[str]
    last_updated: str
    source_hash: str


@dataclass
class ProjectInterface:
    """Extracted public interface (function or class)."""

    id: int
    job_id: str
    source_file: str
    interface_type: str  # 'class', 'function'
    interface_name: str
    signature: str
    docstring: str | None
    exports_from: str


@dataclass
class ProjectState:
    """High-level project state snapshot."""

    id: int
    job_id: str
    total_files: int
    generated_files: int
    validated_files: int
    failed_files: int
    last_action: str
    last_action_time: str
    metadata: dict[str, Any]


class ProjectMemoryManager:
    """Manages project memory database operations.

    Core responsibility:
    - Store file-level metadata after generation
    - Query project structure for context building
    - Track project state for orchestrator decision-making
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ================================================================
    # File Metadata Operations
    # ================================================================

    async def store_file_metadata(
        self,
        job_id: str,
        file_path: str,
        source_code: str,
        file_type: str = "python",
    ) -> None:
        """Extract and store file metadata after successful generation.

        Args:
            job_id: Job identifier
            file_path: Path of the generated file
            source_code: Complete source code
            file_type: File type ('python', 'json', etc.)
        """
        metadata = self._extract_metadata(source_code, file_type)
        source_hash = hashlib.sha256(source_code.encode()).hexdigest()
        now = datetime.now(timezone.utc).isoformat()

        await self._db.execute(
            """
            INSERT OR REPLACE INTO project_memory
            (job_id, file_path, file_type, exports, imports, classes, functions, dependencies, last_updated, source_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                file_path,
                file_type,
                json.dumps(metadata.exports),
                json.dumps(metadata.imports),
                json.dumps(metadata.classes),
                json.dumps(metadata.functions),
                json.dumps(metadata.dependencies),
                now,
                source_hash,
            ),
        )

    async def get_file_metadata(
        self,
        job_id: str,
        file_path: str,
    ) -> ProjectMemoryEntry | None:
        """Retrieve stored metadata for a file.

        Args:
            job_id: Job identifier
            file_path: Path of the file

        Returns:
            ProjectMemoryEntry if found, None otherwise
        """
        row = await self._db.fetchone(
            """
            SELECT id, job_id, file_path, file_type, exports, imports, classes,
                   functions, dependencies, last_updated, source_hash
            FROM project_memory
            WHERE job_id = ? AND file_path = ?
            """,
            (job_id, file_path),
        )
        if row is None:
            return None

        return ProjectMemoryEntry(
            id=row[0],
            job_id=row[1],
            file_path=row[2],
            file_type=row[3],
            exports=json.loads(row[4]),
            imports=json.loads(row[5]),
            classes=json.loads(row[6]),
            functions=json.loads(row[7]),
            dependencies=json.loads(row[8]),
            last_updated=row[9],
            source_hash=row[10],
        )

    async def get_project_files(self, job_id: str) -> list[ProjectMemoryEntry]:
        """Get metadata for all files in a project.

        Args:
            job_id: Job identifier

        Returns:
            List of ProjectMemoryEntry objects
        """
        rows = await self._db.fetchall(
            """
            SELECT id, job_id, file_path, file_type, exports, imports, classes,
                   functions, dependencies, last_updated, source_hash
            FROM project_memory
            WHERE job_id = ?
            ORDER BY last_updated DESC
            """,
            (job_id,),
        )

        return [
            ProjectMemoryEntry(
                id=row[0],
                job_id=row[1],
                file_path=row[2],
                file_type=row[3],
                exports=json.loads(row[4]),
                imports=json.loads(row[5]),
                classes=json.loads(row[6]),
                functions=json.loads(row[7]),
                dependencies=json.loads(row[8]),
                last_updated=row[9],
                source_hash=row[10],
            )
            for row in rows
        ]

    # ================================================================
    # Dependency Resolution
    # ================================================================

    async def get_dependencies_for_file(
        self,
        job_id: str,
        file_path: str,
    ) -> list[ProjectMemoryEntry]:
        """Get metadata for all dependencies of a file.

        Args:
            job_id: Job identifier
            file_path: File path to resolve dependencies for

        Returns:
            List of ProjectMemoryEntry objects for dependencies
        """
        entry = await self.get_file_metadata(job_id, file_path)
        if entry is None:
            return []

        dependencies = []
        for dep_path in entry.dependencies:
            dep_entry = await self.get_file_metadata(job_id, dep_path)
            if dep_entry is not None:
                dependencies.append(dep_entry)

        return dependencies

    async def get_dependents_for_file(
        self,
        job_id: str,
        file_path: str,
    ) -> list[ProjectMemoryEntry]:
        """Get metadata for all files that depend on this file.

        Args:
            job_id: Job identifier
            file_path: File path to find dependents for

        Returns:
            List of ProjectMemoryEntry objects that import this file
        """
        all_files = await self.get_project_files(job_id)
        dependents = []
        for entry in all_files:
            if file_path in entry.dependencies:
                dependents.append(entry)
        return dependents

    # ================================================================
    # Interface Extraction & Lookup
    # ================================================================

    async def store_interfaces(
        self,
        job_id: str,
        file_path: str,
        interfaces: list[dict[str, str]],
    ) -> None:
        """Store extracted interfaces (functions, classes) for a file.

        Args:
            job_id: Job identifier
            file_path: File path
            interfaces: List of interface dicts with keys:
                - type: 'class' or 'function'
                - name: interface name
                - signature: function sig or class def
                - docstring: optional docstring
        """
        for interface in interfaces:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO project_interfaces
                (job_id, source_file, interface_type, interface_name, signature, docstring, exports_from)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    file_path,
                    interface["type"],
                    interface["name"],
                    interface["signature"],
                    interface.get("docstring"),
                    file_path,
                ),
            )

    async def find_interface(
        self,
        job_id: str,
        interface_name: str,
    ) -> ProjectInterface | None:
        """Find an interface by name across the project.

        Args:
            job_id: Job identifier
            interface_name: Name of the interface (class or function)

        Returns:
            ProjectInterface if found, None otherwise
        """
        row = await self._db.fetchone(
            """
            SELECT id, job_id, source_file, interface_type, interface_name, signature, docstring, exports_from
            FROM project_interfaces
            WHERE job_id = ? AND interface_name = ?
            LIMIT 1
            """,
            (job_id, interface_name),
        )
        if row is None:
            return None

        return ProjectInterface(
            id=row[0],
            job_id=row[1],
            source_file=row[2],
            interface_type=row[3],
            interface_name=row[4],
            signature=row[5],
            docstring=row[6],
            exports_from=row[7],
        )

    async def get_file_interfaces(
        self,
        job_id: str,
        file_path: str,
    ) -> list[ProjectInterface]:
        """Get all interfaces exported by a file.

        Args:
            job_id: Job identifier
            file_path: File path

        Returns:
            List of ProjectInterface objects
        """
        rows = await self._db.fetchall(
            """
            SELECT id, job_id, source_file, interface_type, interface_name, signature, docstring, exports_from
            FROM project_interfaces
            WHERE job_id = ? AND source_file = ?
            ORDER BY interface_type, interface_name
            """,
            (job_id, file_path),
        )

        return [
            ProjectInterface(
                id=row[0],
                job_id=row[1],
                source_file=row[2],
                interface_type=row[3],
                interface_name=row[4],
                signature=row[5],
                docstring=row[6],
                exports_from=row[7],
            )
            for row in rows
        ]

    # ================================================================
    # Project State
    # ================================================================

    async def update_project_state(
        self,
        job_id: str,
        last_action: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update project state snapshot.

        Args:
            job_id: Job identifier
            last_action: Description of last action taken
            metadata: Additional state metadata
        """
        now = datetime.now(timezone.utc).isoformat()
        metadata = metadata or {}

        # Count files
        all_files = await self.get_project_files(job_id)
        total_files = len(all_files)

        await self._db.execute(
            """
            INSERT OR REPLACE INTO project_state
            (job_id, total_files, generated_files, validated_files, failed_files, last_action, last_action_time, metadata_json)
            VALUES (?, ?, 0, 0, 0, ?, ?, ?)
            """,
            (job_id, total_files, last_action, now, json.dumps(metadata)),
        )

    async def get_project_state(self, job_id: str) -> ProjectState | None:
        """Retrieve project state snapshot.

        Args:
            job_id: Job identifier

        Returns:
            ProjectState if exists, None otherwise
        """
        row = await self._db.fetchone(
            """
            SELECT id, job_id, total_files, generated_files, validated_files, failed_files, last_action, last_action_time, metadata_json
            FROM project_state
            WHERE job_id = ?
            """,
            (job_id,),
        )
        if row is None:
            return None

        return ProjectState(
            id=row[0],
            job_id=row[1],
            total_files=row[2],
            generated_files=row[3],
            validated_files=row[4],
            failed_files=row[5],
            last_action=row[6],
            last_action_time=row[7],
            metadata=json.loads(row[8]),
        )

    # ================================================================
    # Private Methods - Metadata Extraction
    # ================================================================

    def _extract_metadata(
        self,
        source_code: str,
        file_type: str,
    ) -> FileMetadata:
        """Extract metadata from source code.

        For Python: parse imports, exports, class/function definitions.
        For other formats: return empty lists.

        Args:
            source_code: Complete source code
            file_type: File type

        Returns:
            FileMetadata object
        """
        if file_type == "python":
            return self._extract_python_metadata(source_code)
        else:
            # For non-Python files, return minimal metadata
            return FileMetadata(
                file_path="",
                file_type=file_type,
                exports=[],
                imports=[],
                classes=[],
                functions=[],
                dependencies=[],
            )

    def _extract_python_metadata(self, source_code: str) -> FileMetadata:
        """Extract metadata from Python source code.

        Parse:
        - Imports (from X import Y, import X)
        - Class definitions
        - Function definitions
        - __all__ exports (if present)

        Args:
            source_code: Python source code

        Returns:
            FileMetadata object
        """
        imports = []
        classes = []
        functions = []
        exports = []

        lines = source_code.split("\n")

        for line in lines:
            line = line.strip()

            # Parse imports
            if line.startswith("from ") and " import " in line:
                # from X import Y
                parts = line.split(" import ")
                if len(parts) == 2:
                    module = parts[0].replace("from ", "").strip()
                    if not module.startswith("."):
                        imports.append(module)
            elif line.startswith("import "):
                # import X
                module = line.replace("import ", "").split(",")[0].strip()
                imports.append(module)

            # Parse class definitions
            elif line.startswith("class "):
                class_name = line.replace("class ", "").split("(")[0].split(":")[0].strip()
                if class_name:
                    classes.append(class_name)

            # Parse function definitions
            elif line.startswith("def "):
                func_name = line.replace("def ", "").split("(")[0].strip()
                if func_name and not func_name.startswith("_"):
                    functions.append(func_name)

            # Parse __all__
            elif "__all__" in line and "=" in line:
                try:
                    all_str = line.split("=", 1)[1].strip()
                    if all_str.startswith("[") and all_str.endswith("]"):
                        exports = json.loads(all_str)
                except (json.JSONDecodeError, ValueError):
                    pass

        return FileMetadata(
            file_path="",
            file_type="python",
            exports=exports or functions,  # Default exports to public functions
            imports=imports,
            classes=classes,
            functions=functions,
            dependencies=[],  # Will be populated by import analysis
        )
