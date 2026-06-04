"""Agents package — executor, prompt builders, and output schemas."""

from ollama_fleet.agents.schemas import AgentOutput, AgentType
from ollama_fleet.agents.capabilities import (
    Capability,
    CapabilityType,
    CapabilityResult,
    CapabilityRegistry,
    PlanningCapability,
    CodeGenerationCapability,
    CodeReviewCapability,
    TestingCapability,
)
from ollama_fleet.agents.executor import AgentExecutor

__all__ = [
    "AgentOutput",
    "AgentType",
    "Capability",
    "CapabilityType",
    "CapabilityResult",
    "CapabilityRegistry",
    "PlanningCapability",
    "CodeGenerationCapability",
    "CodeReviewCapability",
    "TestingCapability",
    "AgentExecutor",
]
