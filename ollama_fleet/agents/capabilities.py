"""Capabilities Registry and Base Classes.

In the new architecture, agents are tools/capabilities.
The orchestrator selects the appropriate capability for each action.

Capabilities are pure LLM wrappers with no decision-making logic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ollama_fleet.agents.executor import AgentExecutor

logger = logging.getLogger(__name__)


class CapabilityType(str, Enum):
    """Available capability types."""

    PLANNING = "planning"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    WORKSPACE_SEARCH = "workspace_search"
    INTERFACE_EXTRACTION = "interface_extraction"


@dataclass
class CapabilityResult:
    """Result from executing a capability."""

    capability_type: CapabilityType
    success: bool
    output: Any
    error: str | None = None
    metadata: dict[str, Any] | None = None


class Capability(ABC):
    """Base class for all capabilities (former agents).

    Capabilities are pure tools: they take input and produce output.
    They do NOT make decisions about what to do next.
    """

    def __init__(
        self,
        executor: AgentExecutor,
    ) -> None:
        self._executor = executor

    @property
    @abstractmethod
    def capability_type(self) -> CapabilityType:
        """Return the type of this capability."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> CapabilityResult:
        """Execute the capability.

        Args:
            **kwargs: Capability-specific arguments

        Returns:
            CapabilityResult with output or error
        """
        pass


class PlanningCapability(Capability):
    """Planning capability: break goal into tasks and file structure.

    Replaces: Planner agent (but simplified - only structure, no details)

    Input:
    - goal: Project goal
    - context: Optional project context

    Output:
    - files: List of (filename, purpose, dependencies)
    - architecture_notes: Any architectural decisions
    """

    @property
    def capability_type(self) -> CapabilityType:
        return CapabilityType.PLANNING

    async def execute(self, **kwargs: Any) -> CapabilityResult:
        goal = kwargs.get("goal")
        context = kwargs.get("context", "")

        if not goal:
            return CapabilityResult(
                capability_type=self.capability_type,
                success=False,
                output=None,
                error="Missing 'goal' parameter",
            )

        # Use executor to call the planner model
        # (Keep the existing planner prompt for now)
        try:
            output = await self._executor.execute_planner(goal=goal, context=context)
            return CapabilityResult(
                capability_type=self.capability_type,
                success=True,
                output=output,
            )
        except Exception as e:
            return CapabilityResult(
                capability_type=self.capability_type,
                success=False,
                output=None,
                error=str(e),
            )


class CodeGenerationCapability(Capability):
    """Code generation capability: write or modify a file.

    Replaces: Coder agent + File Specification Agent

    Input:
    - file_path: Target file path
    - file_purpose: What the file should do
    - dependencies: Context from dependent files
    - project_memory: Project structure/interfaces
    - requirements: Explicit requirements

    Output:
    - source_code: Generated file content
    - imports: Extracted imports
    - exports: Extracted exports
    """

    @property
    def capability_type(self) -> CapabilityType:
        return CapabilityType.CODE_GENERATION

    async def execute(self, **kwargs: Any) -> CapabilityResult:
        file_path = kwargs.get("file_path")
        file_purpose = kwargs.get("file_purpose")
        dependencies = kwargs.get("dependencies", [])
        project_memory = kwargs.get("project_memory", [])
        requirements = kwargs.get("requirements", "")

        if not file_path or not file_purpose:
            return CapabilityResult(
                capability_type=self.capability_type,
                success=False,
                output=None,
                error="Missing 'file_path' or 'file_purpose' parameter",
            )

        try:
            output = await self._executor.execute_coder(
                file_path=file_path,
                file_purpose=file_purpose,
                dependencies=dependencies,
                project_memory=project_memory,
                requirements=requirements,
            )
            return CapabilityResult(
                capability_type=self.capability_type,
                success=True,
                output=output,
            )
        except Exception as e:
            return CapabilityResult(
                capability_type=self.capability_type,
                success=False,
                output=None,
                error=str(e),
            )


class CodeReviewCapability(Capability):
    """Code review capability: evaluate code for issues.

    Replaces: Critic agent

    Only reviews:
    - Failed tests
    - Explicit requirements
    - Validation failures

    Does NOT invent hypothetical bugs.

    Input:
    - source_code: Code to review
    - requirements: Explicit requirements for the code
    - test_failures: Any failed test output
    - file_path: Context (what file is this)

    Output:
    - approved: boolean
    - issues: List of concrete issues (or empty)
    - suggestions: Optional improvements
    """

    @property
    def capability_type(self) -> CapabilityType:
        return CapabilityType.CODE_REVIEW

    async def execute(self, **kwargs: Any) -> CapabilityResult:
        source_code = kwargs.get("source_code")
        requirements = kwargs.get("requirements", "")
        test_failures = kwargs.get("test_failures", "")
        file_path = kwargs.get("file_path", "")

        if not source_code:
            return CapabilityResult(
                capability_type=self.capability_type,
                success=False,
                output=None,
                error="Missing 'source_code' parameter",
            )

        try:
            output = await self._executor.execute_critic(
                source_code=source_code,
                requirements=requirements,
                test_failures=test_failures,
                file_path=file_path,
            )
            return CapabilityResult(
                capability_type=self.capability_type,
                success=True,
                output=output,
            )
        except Exception as e:
            return CapabilityResult(
                capability_type=self.capability_type,
                success=False,
                output=None,
                error=str(e),
            )


class TestingCapability(Capability):
    """Testing capability: analyze test results.

    Replaces: Tester agent

    Input:
    - test_output: Raw test execution output
    - test_command: Command that was run
    - source_files: Source files being tested

    Output:
    - tests_passed: Number of passed tests
    - tests_failed: Number of failed tests
    - failures_detail: Failure details
    - ready_for_review: boolean
    """

    @property
    def capability_type(self) -> CapabilityType:
        return CapabilityType.TESTING

    async def execute(self, **kwargs: Any) -> CapabilityResult:
        test_output = kwargs.get("test_output")

        if not test_output:
            return CapabilityResult(
                capability_type=self.capability_type,
                success=False,
                output=None,
                error="Missing 'test_output' parameter",
            )

        try:
            output = await self._executor.execute_tester(test_output=test_output)
            return CapabilityResult(
                capability_type=self.capability_type,
                success=True,
                output=output,
            )
        except Exception as e:
            return CapabilityResult(
                capability_type=self.capability_type,
                success=False,
                output=None,
                error=str(e),
            )


class WorkspaceSearchCapability(Capability):
    """Workspace search capability: find files and understand structure.

    Input:
    - query: Search query or pattern
    - workspace_path: Root path to search in
    - file_types: Optional file type filters

    Output:
    - files_found: List of matching files
    - content_preview: Optional preview of file contents
    """

    @property
    def capability_type(self) -> CapabilityType:
        return CapabilityType.WORKSPACE_SEARCH

    async def execute(self, **kwargs: Any) -> CapabilityResult:
        # Stub implementation - actual implementation uses filesystem
        # This is more of a tool than an agent capability
        query = kwargs.get("query")
        if not query:
            return CapabilityResult(
                capability_type=self.capability_type,
                success=False,
                output=None,
                error="Missing 'query' parameter",
            )

        return CapabilityResult(
            capability_type=self.capability_type,
            success=True,
            output={"files_found": []},
        )


class InterfaceExtractionCapability(Capability):
    """Interface extraction capability: extract APIs from code.

    Input:
    - source_code: Source code to analyze
    - file_path: File path for context

    Output:
    - exports: List of exported names
    - classes: List of class definitions with signatures
    - functions: List of function definitions with signatures
    """

    @property
    def capability_type(self) -> CapabilityType:
        return CapabilityType.INTERFACE_EXTRACTION

    async def execute(self, **kwargs: Any) -> CapabilityResult:
        source_code = kwargs.get("source_code")

        if not source_code:
            return CapabilityResult(
                capability_type=self.capability_type,
                success=False,
                output=None,
                error="Missing 'source_code' parameter",
            )

        # For now, use ProjectMemoryManager's extraction
        # In future, could use LLM for more sophisticated analysis
        return CapabilityResult(
            capability_type=self.capability_type,
            success=True,
            output={"exports": [], "classes": [], "functions": []},
        )


class CapabilityRegistry:
    """Registry and factory for capabilities.

    The orchestrator uses this to select and execute capabilities.
    """

    def __init__(self, executor: AgentExecutor) -> None:
        self._executor = executor
        self._capabilities: dict[CapabilityType, Capability] = {
            CapabilityType.PLANNING: PlanningCapability(executor),
            CapabilityType.CODE_GENERATION: CodeGenerationCapability(executor),
            CapabilityType.CODE_REVIEW: CodeReviewCapability(executor),
            CapabilityType.TESTING: TestingCapability(executor),
            CapabilityType.WORKSPACE_SEARCH: WorkspaceSearchCapability(executor),
            CapabilityType.INTERFACE_EXTRACTION: InterfaceExtractionCapability(executor),
        }

    def get_capability(self, capability_type: CapabilityType) -> Capability:
        """Get a capability by type.

        Args:
            capability_type: Type of capability to retrieve

        Returns:
            Capability instance

        Raises:
            ValueError: If capability type not found
        """
        if capability_type not in self._capabilities:
            raise ValueError(f"Unknown capability type: {capability_type}")
        return self._capabilities[capability_type]

    async def execute(
        self,
        capability_type: CapabilityType,
        **kwargs: Any,
    ) -> CapabilityResult:
        """Execute a capability by type.

        Args:
            capability_type: Type of capability to execute
            **kwargs: Arguments for the capability

        Returns:
            CapabilityResult with output or error
        """
        capability = self.get_capability(capability_type)
        return await capability.execute(**kwargs)
