"""Tests for agent executor output normalization."""

from __future__ import annotations

import json

from ollama_fleet.agents.executor import AgentExecutor
from ollama_fleet.agents.schemas import AgentType, SynthesizerOutput


def test_parse_synthesizer_output_recovers_numbered_next_steps() -> None:
    executor = object.__new__(AgentExecutor)
    raw = json.dumps(
        {
            "summary": "Completed calculator work.",
            "changelog": ["Added core logic"],
            "files_produced": ["src/calculator.py"],
            "Step 1": "Run unit tests",
            "Step 2": "Review UI flow",
        }
    )

    parsed = executor._parse_output(raw, AgentType.SYNTHESIZER)

    assert isinstance(parsed, SynthesizerOutput)
    assert parsed.next_steps == ["Run unit tests", "Review UI flow"]


def test_parse_synthesizer_output_coerces_missing_lists() -> None:
    executor = object.__new__(AgentExecutor)
    raw = json.dumps(
        {
            "summary": "Done",
            "changes": "Added demo",
            "files": {"main": "src/demo.py"},
            "nextSteps": None,
        }
    )

    parsed = executor._parse_output(raw, AgentType.SYNTHESIZER)

    assert isinstance(parsed, SynthesizerOutput)
    assert parsed.changelog == ["Added demo"]
    assert parsed.files_produced == ["src/demo.py"]
    assert parsed.next_steps == []

