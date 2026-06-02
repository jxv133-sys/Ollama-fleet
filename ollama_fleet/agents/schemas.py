"""Pydantic output schemas for all five agent types.

Each agent returns a structured JSON payload conforming to one of these schemas.
Downstream components rely on these schemas for reliable, type-safe processing.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------


class PlannerTask(BaseModel):
    """A single task produced by the Planner_Agent."""

    task_id: str
    step_number: int = Field(default=0, ge=0)  # 0 = unset; normalizer fills it in
    title: str
    description: str
    # Accept both enum values and string variants
    agent_type: Union[Literal["coder", "tester", "synthesizer"], str]
    dependencies: list[str] = []
    # Accept int or string for priority, with default
    priority: Union[int, str] = Field(default=5)
    # Filename (no paths) - preferred over file_path for flat file structure
    filename: str | None = None
    # Optional file-level hints provided by the planner (legacy support)
    file_path: str | None = None
    required_contents: list[str] = []
    estimated_size: str | None = None
    # Structured file specification with exact requirements
    # Contains: purpose, public_exports, imports, functions[], exact_content, estimated_lines
    file_spec: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: Any) -> Any:
        """Normalise raw dict data before field assignment.

        This runs during ALL instantiation paths (direct construction,
        nested model validation via a parent, and explicit model_validate
        calls), unlike a @classmethod model_validate override which Pydantic v2
        does NOT call for nested models.
        """
        if not isinstance(data, dict):
            return data

        # Coerce priority to int
        if isinstance(data.get("priority"), str):
            try:
                data["priority"] = int(data["priority"])
            except (ValueError, TypeError):
                data["priority"] = 5

        # Normalise agent_type
        at = data.get("agent_type")
        if isinstance(at, list):
            at = str(at[0]) if at else "coder"
        elif isinstance(at, dict):
            at = str(next(iter(at.values()), "coder"))
        if isinstance(at, str):
            at = at.lower()
            if at.endswith("_agent"):
                at = at.replace("_agent", "")
            data["agent_type"] = at

        # Ensure required_contents is a list of strings
        rc = data.get("required_contents")
        if rc is not None:
            if isinstance(rc, str):
                data["required_contents"] = [rc]
            else:
                data["required_contents"] = [str(x) for x in rc]

        # filename → file_path (strip directory components for flat layout)
        if data.get("filename"):
            data["file_path"] = Path(str(data["filename"])).name
            data["filename"] = None
        elif data.get("file_path"):
            data["file_path"] = Path(str(data["file_path"])).name

        # Inject file_path into file_spec so downstream consumers don't need
        # to look in two places.
        if data.get("file_path") and isinstance(data.get("file_spec"), dict):
            if not data["file_spec"].get("file_path"):
                data["file_spec"]["file_path"] = data["file_path"]

        return data

    @classmethod
    def model_validate(cls, data, **kwargs):  # type: ignore[override]
        """Kept for call-site compatibility; delegates to super()."""
        return super().model_validate(data, **kwargs)


class PlannerOutput(BaseModel):
    """Structured output returned by the Planner_Agent.

    Validates: Requirements 5.1
    """

    clarifying_questions: list[str] = []
    technical_requirements: list[str] = []
    tasks: list[PlannerTask]
    milestones: list[str]
    architecture_notes: str


# ---------------------------------------------------------------------------
# Coder Agent
# ---------------------------------------------------------------------------


class FileModification(BaseModel):
    """A single file operation produced by the Coding_Agent."""

    file_path: str
    operation: Literal["create", "modify", "delete"]
    # content MUST be an empty string when operation == "delete"
    content: str


class CoderOutput(BaseModel):
    """Output returned by the Coding_Agent.

    The Coding Agent is now responsible only for generating complete file
    contents. Metadata such as file paths, summaries, and confidence scores
    are managed by the orchestrator.
    """

    content: str


class SpecificationOutput(BaseModel):
    """Output returned by the File Specification Agent.

    This agent produces structured file requirements that the coder uses
    to implement one complete source file.
    """

    file_path: str | None = None
    purpose: str
    imports: list[str] = []
    required_functions: list[str] = []
    required_behavior: list[str] = []
    forbidden_behavior: list[str] = []
    required_contents: list[str] = []


# ---------------------------------------------------------------------------
# Critic Agent
# ---------------------------------------------------------------------------


class CriticIssue(BaseModel):
    """A single issue identified by the Critic_Agent."""

    file_path: str
    # 0 indicates a file-level issue not tied to a specific line
    line_number: int
    severity: Literal["critical", "major", "minor"]
    description: str
    suggested_fix: str


class CriticOutput(BaseModel):
    """Structured output returned by the Critic_Agent.

    Validates: Requirements 5.3
    """

    approved: bool
    issues: list[CriticIssue]
    overall_assessment: str


# ---------------------------------------------------------------------------
# Tester Agent
# ---------------------------------------------------------------------------


class TestFailure(BaseModel):
    """Details of a single test failure reported by the Tester_Agent."""

    test_name: str
    error_message: str
    suggested_fix: str


class TesterOutput(BaseModel):
    """Structured output returned by the Tester_Agent.

    Validates: Requirements 5.4
    """

    tests_passed: int
    tests_failed: int
    failures: list[TestFailure]
    ready_for_review: bool


# ---------------------------------------------------------------------------
# Synthesizer Agent
# ---------------------------------------------------------------------------


class SynthesizerOutput(BaseModel):
    """Structured output returned by the Synthesizer_Agent.

    Validates: Requirements 5.5
    """

    summary: str
    changelog: list[str]
    files_produced: list[str]
    next_steps: list[str]


# ---------------------------------------------------------------------------
# Union type and AgentType enum
# ---------------------------------------------------------------------------

# Union of all possible agent output types.
AgentOutput = PlannerOutput | CoderOutput | SpecificationOutput | CriticOutput | TesterOutput | SynthesizerOutput


class AgentType(str, enum.Enum):
    """Enumeration of all supported agent types."""

    PLANNER = "planner"
    CODER = "coder"
    SPECIFICATION = "specification"
    CRITIC = "critic"
    TESTER = "tester"
    SYNTHESIZER = "synthesizer"
