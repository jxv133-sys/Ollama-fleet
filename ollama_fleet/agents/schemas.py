"""Pydantic output schemas for all five agent types.

Each agent returns a structured JSON payload conforming to one of these schemas.
Downstream components rely on these schemas for reliable, type-safe processing.
"""

from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------


class PlannerTask(BaseModel):
    """A single task produced by the Planner_Agent."""

    task_id: str
    title: str
    description: str
    agent_type: Literal["coder", "tester", "synthesizer"]
    dependencies: list[str] = []
    priority: int = Field(ge=1, le=10)


class PlannerOutput(BaseModel):
    """Structured output returned by the Planner_Agent.

    Validates: Requirements 5.1
    """

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
    """Structured output returned by the Coding_Agent.

    Validates: Requirements 5.2
    """

    file_modifications: list[FileModification]
    summary: str
    confidence_score: float = Field(ge=0.0, le=1.0)


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
AgentOutput = PlannerOutput | CoderOutput | CriticOutput | TesterOutput | SynthesizerOutput


class AgentType(str, enum.Enum):
    """Enumeration of all supported agent types."""

    PLANNER = "planner"
    CODER = "coder"
    CRITIC = "critic"
    TESTER = "tester"
    SYNTHESIZER = "synthesizer"
