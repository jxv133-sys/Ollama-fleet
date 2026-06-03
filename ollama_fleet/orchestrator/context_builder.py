"""Intelligent Context Builder.

Instead of passing the entire project to models, this builds focused context
by querying project memory for only the information needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ollama_fleet.memory.project_memory import ProjectMemoryManager, ProjectMemoryEntry

logger = logging.getLogger(__name__)


@dataclass
class FocusedContext:
    """Minimal, focused context for a specific task."""

    target_file: str
    target_purpose: str
    relevant_dependencies: list[ProjectMemoryEntry]
    relevant_exports: dict[str, list[str]]  # file_path -> exports
    project_structure: str  # Text description of project
    explicit_requirements: str


class ContextBuilder:
    """Build focused context using project memory.

    Principle: Only provide the model with information it needs.
    Do not dump the entire project into the prompt.
    """

    def __init__(self, memory: ProjectMemoryManager) -> None:
        self._memory = memory

    async def build_context_for_file_generation(
        self,
        job_id: str,
        target_file: str,
        target_purpose: str,
        requirements: str = "",
    ) -> FocusedContext:
        """Build context for code generation task.

        For file: src/logic.py that depends on models.py

        Provide:
        1. Target file path and purpose
        2. Metadata for all dependencies (imports, exports, key functions)
        3. Project structure overview
        4. Explicit requirements

        Do NOT provide:
        - The entire project source code
        - Files that are not dependencies
        - Hypothetical architectures

        Args:
            job_id: Job identifier
            target_file: File to generate
            target_purpose: Purpose of the file
            requirements: Any explicit requirements

        Returns:
            FocusedContext object
        """
        # Get dependencies for this file
        all_files = await self._memory.get_project_files(job_id)
        project_structure = self._describe_project_structure(all_files, target_file)

        # Build exports dict: file -> exports
        relevant_exports: dict[str, list[str]] = {}
        for entry in all_files:
            if entry.file_path != target_file:
                relevant_exports[entry.file_path] = entry.exports

        return FocusedContext(
            target_file=target_file,
            target_purpose=target_purpose,
            relevant_dependencies=all_files,
            relevant_exports=relevant_exports,
            project_structure=project_structure,
            explicit_requirements=requirements,
        )

    async def build_context_for_code_review(
        self,
        job_id: str,
        file_path: str,
        source_code: str,
        requirements: str = "",
        test_failures: str = "",
    ) -> dict[str, Any]:
        """Build context for code review task.

        Principle: Critics should only review CONCRETE issues.

        Do NOT ask critics to find:
        - Hypothetical bugs
        - Style issues (use linters)
        - Potential future problems

        Only ask critics to evaluate:
        - Failed tests
        - Explicit requirement mismatches
        - Validation failures

        Args:
            job_id: Job identifier
            file_path: File being reviewed
            source_code: Source code to review
            requirements: Explicit requirements
            test_failures: Any test failures

        Returns:
            Context dict for critic
        """
        # If no requirements and no test failures, auto-approve
        if not requirements and not test_failures:
            return {
                "file_path": file_path,
                "source_code": source_code,
                "requirements": "No explicit requirements",
                "test_failures": "",
                "auto_approved": True,
                "reason": "No explicit requirements or test failures",
            }

        return {
            "file_path": file_path,
            "source_code": source_code,
            "requirements": requirements,
            "test_failures": test_failures,
            "auto_approved": False,
            "only_review": ["requirement_match", "test_pass"],
        }

    async def build_context_for_test_analysis(
        self,
        job_id: str,
        test_output: str,
        source_files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build context for test analysis.

        Args:
            job_id: Job identifier
            test_output: Raw test output
            source_files: Optional list of source files

        Returns:
            Context dict for test analysis
        """
        if source_files is None:
            files = await self._memory.get_project_files(job_id)
            source_files = [f.file_path for f in files]

        return {
            "test_output": test_output,
            "source_files": source_files,
            "focus": ["failed_tests", "error_messages", "stack_traces"],
        }

    async def build_dependencies_context(
        self,
        job_id: str,
        target_file: str,
    ) -> str:
        """Build text description of dependencies for a file.

        Args:
            job_id: Job identifier
            target_file: File to get dependencies for

        Returns:
            Text description of dependencies
        """
        target = await self._memory.get_file_metadata(job_id, target_file)
        if not target:
            return ""

        deps = await self._memory.get_dependencies_for_file(job_id, target_file)

        lines = [f"# Dependencies for {target_file}"]
        for dep in deps:
            lines.append(f"\n## {dep.file_path}")
            lines.append(f"Purpose: (see plan)")
            if dep.exports:
                lines.append(f"Exports: {', '.join(dep.exports)}")
            if dep.classes:
                lines.append(f"Classes: {', '.join(dep.classes)}")
            if dep.functions:
                lines.append(f"Functions: {', '.join(dep.functions)}")

        return "\n".join(lines)

    def _describe_project_structure(
        self,
        all_files: list[ProjectMemoryEntry],
        target_file: str,
    ) -> str:
        """Describe project structure in text form.

        Lists all files with their exports/responsibilities.

        Args:
            all_files: All files in project
            target_file: File being generated (highlight this)

        Returns:
            Text description
        """
        lines = ["# Project Structure"]
        for entry in all_files:
            marker = " (generating)" if entry.file_path == target_file else ""
            lines.append(f"\n## {entry.file_path}{marker}")
            if entry.exports:
                lines.append(f"Exports: {', '.join(entry.exports)}")
            if entry.classes:
                lines.append(f"Classes: {', '.join(entry.classes)}")
            if entry.functions:
                lines.append(f"Functions: {', '.join(entry.functions)}")

        return "\n".join(lines)


class ValidationLayer:
    """Pre-critic validation: reject invalid output before critics see it.

    If generated code:
    - is empty
    - only contains a file path
    - is not valid Python syntax
    - is not a valid source code file

    Then:
    - reject immediately
    - request regeneration
    - do NOT invoke critics
    """

    @staticmethod
    def validate_generated_code(source_code: str, file_path: str) -> tuple[bool, str]:
        """Validate generated source code.

        Args:
            source_code: Generated source code
            file_path: Target file path

        Returns:
            (is_valid, error_message_if_invalid)
        """
        # Check: not empty
        if not source_code or not source_code.strip():
            return False, "Generated code is empty"

        # Check: not just a file path
        if source_code.strip() == file_path:
            return False, "Generated code is only a file path, not actual code"

        # Check: basic syntax validation
        if file_path.endswith(".py"):
            try:
                compile(source_code, file_path, "exec")
            except SyntaxError as e:
                return False, f"Python syntax error: {e}"

        return True, ""

    @staticmethod
    def should_skip_critic_review(
        source_code: str,
        requirements: str = "",
        test_failures: str = "",
    ) -> tuple[bool, str]:
        """Determine if critic review is even needed.

        If there are no explicit requirements and no test failures,
        and code is syntactically valid, auto-approve.

        Args:
            source_code: Generated source code
            requirements: Explicit requirements
            test_failures: Test failure output

        Returns:
            (should_skip_critic, reason)
        """
        if not requirements and not test_failures:
            # No explicit criteria, auto-approve
            return True, "No explicit requirements or test failures to evaluate"

        return False, ""
