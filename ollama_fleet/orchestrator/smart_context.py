"""Smart Context System - Intelligent context building for orchestrator.

This builds minimal, focused context for each capability based on:
1. What the capability needs to know
2. What's available in project memory
3. What will help the model succeed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ollama_fleet.memory.project_memory import ProjectMemoryManager
from ollama_fleet.orchestrator.context_builder import ContextBuilder, ValidationLayer
from ollama_fleet.orchestrator.model_router import ModelRouter
from ollama_fleet.agents.capabilities import CapabilityType

logger = logging.getLogger(__name__)


@dataclass
class EnrichedContext:
    """Context enriched with project-aware information."""

    capability_type: CapabilityType
    base_context: dict[str, Any]
    project_structure: str
    relevant_files: list[dict[str, Any]]
    dependencies: list[dict[str, Any]]
    project_insights: str


class SmartContextBuilder:
    """Build intelligent context for capabilities.

    Uses ProjectMemory to understand:
    - Project structure and relationships
    - File dependencies and exports
    - What context would help the model
    - What can be omitted (reduce noise)
    """

    def __init__(
        self,
        memory: ProjectMemoryManager,
        context_builder: ContextBuilder,
        model_router: ModelRouter,
    ) -> None:
        self._memory = memory
        self._context_builder = context_builder
        self._model_router = model_router

    async def build_for_planning(
        self,
        job_id: str,
        goal: str,
        existing_context: str = "",
    ) -> dict[str, Any]:
        """Build context for planning capability.

        Planner needs:
        - The goal clearly stated
        - Any existing files (to avoid duplication)
        - Project purpose/type hints

        Args:
            job_id: Job identifier
            goal: Project goal
            existing_context: Optional existing context

        Returns:
            Context dict for planner capability
        """
        existing_files = await self._memory.get_project_files(job_id)

        context = {
            "goal": goal,
            "existing_context": existing_context,
            "focus": ["break goal into files", "identify dependencies", "plan structure"],
        }

        if existing_files:
            context["existing_files"] = [
                {
                    "path": f.file_path,
                    "exports": f.exports,
                    "purpose": f"(see plan)",
                }
                for f in existing_files
            ]

        return context

    async def build_for_code_generation(
        self,
        job_id: str,
        target_file: str,
        target_purpose: str,
        requirements: str = "",
    ) -> dict[str, Any]:
        """Build context for code generation.

        Coder needs:
        - Target file path and purpose
        - Signatures/exports of dependencies
        - Explicit requirements
        - NOT the full source of all files (noise)

        Args:
            job_id: Job identifier
            target_file: File to generate
            target_purpose: Purpose of the file
            requirements: Explicit requirements

        Returns:
            Context dict for code generation
        """
        # Build focused context
        focused = await self._context_builder.build_context_for_file_generation(
            job_id,
            target_file,
            target_purpose,
            requirements,
        )

        # Get dependencies
        all_files = await self._memory.get_project_files(job_id)
        target_metadata = await self._memory.get_file_metadata(job_id, target_file)

        dependencies = []
        if target_metadata:
            for dep_path in target_metadata.dependencies:
                dep = await self._memory.get_file_metadata(job_id, dep_path)
                if dep:
                    dependencies.append({
                        "path": dep.file_path,
                        "exports": dep.exports,
                        "classes": dep.classes,
                        "functions": dep.functions,
                        "type": dep.file_type,
                    })

        # Build project structure description
        project_structure = self._describe_project_structure(all_files, target_file)

        return {
            "target_file": target_file,
            "target_purpose": target_purpose,
            "requirements": requirements,
            "project_structure": project_structure,
            "dependencies": dependencies,
            "focus": [
                "match file purpose",
                "import from dependencies",
                "follow requirements",
            ],
        }

    async def build_for_code_review(
        self,
        job_id: str,
        file_path: str,
        source_code: str,
        requirements: str = "",
        test_failures: str = "",
    ) -> dict[str, Any]:
        """Build context for code review.

        Critic needs:
        - Source code to review
        - Explicit requirements
        - Test failures
        - Auto-approve if nothing to check

        Does NOT need:
        - Hypothetical concerns
        - Style advice (use linters)
        - Speculative issues

        Args:
            job_id: Job identifier
            file_path: File being reviewed
            source_code: Source code to review
            requirements: Explicit requirements
            test_failures: Test failure output

        Returns:
            Context dict for code review
        """
        review_context = {
            "file_path": file_path,
            "source_code": source_code,
            "requirements": requirements or "No explicit requirements",
            "test_failures": test_failures or "No test failures reported",
        }

        # Auto-approval check
        should_skip, reason = ValidationLayer.should_skip_critic_review(
            source_code,
            requirements,
            test_failures,
        )

        if should_skip:
            review_context["auto_approved"] = True
            review_context["reason"] = reason
            logger.info(f"Auto-approved {file_path}: {reason}")
        else:
            review_context["auto_approved"] = False
            review_context["only_review"] = [
                "requirement_match",
                "test_failures",
                "validation_errors",
            ]

        return review_context

    async def build_for_test_analysis(
        self,
        job_id: str,
        test_output: str,
        source_files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build context for test analysis.

        Tester needs:
        - Test output to analyze
        - Source file list
        - What to focus on (failures, errors)

        Args:
            job_id: Job identifier
            test_output: Test execution output
            source_files: Optional source file list

        Returns:
            Context dict for test analysis
        """
        if source_files is None:
            files = await self._memory.get_project_files(job_id)
            source_files = [f.file_path for f in files]

        return {
            "test_output": test_output,
            "source_files": source_files,
            "focus": [
                "identify failing tests",
                "extract error messages",
                "suggest fixes",
            ],
        }

    async def build_for_file_fix(
        self,
        job_id: str,
        file_path: str,
        error_message: str,
        original_purpose: str,
    ) -> dict[str, Any]:
        """Build context for regenerating a failed file.

        When regenerating a file that failed validation,
        provide error context so model can fix it.

        Args:
            job_id: Job identifier
            file_path: File to fix
            error_message: Why it failed
            original_purpose: Original file purpose

        Returns:
            Context dict for code generation (fix mode)
        """
        all_files = await self._memory.get_project_files(job_id)
        target_metadata = await self._memory.get_file_metadata(job_id, file_path)

        dependencies = []
        if target_metadata:
            for dep_path in target_metadata.dependencies:
                dep = await self._memory.get_file_metadata(job_id, dep_path)
                if dep:
                    dependencies.append({
                        "path": dep.file_path,
                        "exports": dep.exports,
                    })

        return {
            "target_file": file_path,
            "target_purpose": original_purpose,
            "error_context": f"Previous generation failed: {error_message}",
            "regenerate_instructions": [
                "Fix the previous error",
                "Ensure valid syntax",
                "Maintain original purpose",
                "Follow file conventions",
            ],
            "dependencies": dependencies,
        }

    def _describe_project_structure(
        self,
        all_files: list[Any],
        target_file: str,
    ) -> str:
        """Describe project structure concisely."""
        lines = ["# Project Structure"]
        
        for entry in all_files:
            if entry.file_path == target_file:
                lines.append(f"\n## {entry.file_path} (GENERATING)")
            else:
                lines.append(f"\n## {entry.file_path}")
            
            if entry.exports:
                lines.append(f"  Exports: {', '.join(entry.exports[:3])}")  # Top 3 only
            if entry.classes:
                lines.append(f"  Classes: {', '.join(entry.classes[:2])}")  # Top 2 only

        return "\n".join(lines)

    async def validate_code_before_review(
        self,
        file_path: str,
        source_code: str,
    ) -> tuple[bool, str]:
        """Validate code before sending to critic.

        Returns:
            (is_valid, error_message)
        """
        is_valid, error_msg = ValidationLayer.validate_generated_code(
            source_code,
            file_path,
        )
        
        if not is_valid:
            logger.warning(f"Validation failed for {file_path}: {error_msg}")
        
        return is_valid, error_msg
