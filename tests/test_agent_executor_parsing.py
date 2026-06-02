"""Tests for agent executor output normalization."""

from __future__ import annotations

import json

from ollama_fleet.agents.coder import build_coder_prompt
from ollama_fleet.agents.executor import AgentExecutor
from ollama_fleet.agents.schemas import AgentType, CoderOutput, SynthesizerOutput
from ollama_fleet.config import FleetSettings
from ollama_fleet.ollama.client import OllamaClient


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


def test_parse_coder_output_accepts_raw_code_content() -> None:
    executor = object.__new__(AgentExecutor)
    raw = "def add(a, b):\n    return a + b\n"

    parsed = executor._parse_output(raw, AgentType.CODER)

    assert isinstance(parsed, CoderOutput)
    assert parsed.content == "def add(a, b):\n    return a + b"


def test_parse_coder_output_strips_code_fences() -> None:
    executor = object.__new__(AgentExecutor)
    raw = "```python\ndef add(a, b):\n    return a + b\n```"

    parsed = executor._parse_output(raw, AgentType.CODER)

    assert isinstance(parsed, CoderOutput)
    assert parsed.content == "def add(a, b):\n    return a + b"


def test_parse_coder_output_extracts_content_from_legacy_json_wrapper() -> None:
    executor = object.__new__(AgentExecutor)
    raw = json.dumps(
        {
            "content": "def add(a, b):\n    return a + b\n"
        }
    )

    parsed = executor._parse_output(raw, AgentType.CODER)

    assert isinstance(parsed, CoderOutput)
    assert parsed.content == "def add(a, b):\n    return a + b"


def test_parse_coder_output_rejects_placeholder_example_content() -> None:
    executor = object.__new__(AgentExecutor)
    raw = json.dumps(
        {
            "file_modifications": [
                {
                    "file_path": "src/models.py",
                    "operation": "create",
                    "content": "# Example module content here; replace with actual file content",
                }
            ],
            "summary": "Placeholder output",
            "confidence_score": 0.5,
        }
    )

    try:
        executor._parse_output(raw, AgentType.CODER)
        assert False, "Expected ValueError for placeholder coder output"
    except ValueError as exc:
        assert "placeholder" in str(exc).lower()


def test_parse_coder_output_rejects_multiple_file_modifications() -> None:
    executor = object.__new__(AgentExecutor)
    raw = json.dumps(
        {
            "file_modifications": [
                {
                    "file_path": "src/models.py",
                    "operation": "create",
                    "content": "class Model:\n    pass\n",
                },
                {
                    "file_path": "src/logic.py",
                    "operation": "create",
                    "content": "def process(data):\n    return data\n",
                },
            ],
            "summary": "Created two files",
            "confidence_score": 0.8,
        }
    )

    try:
        executor._parse_output(raw, AgentType.CODER)
        assert False, "Expected ValueError for multiple file modifications"
    except ValueError as exc:
        assert "exactly one file_modifications" in str(exc)


def test_build_coder_prompt_includes_critic_issues() -> None:
    prompt = build_coder_prompt(
        task_description="Fix bug in src/app.py",
        active_files=["src/app.py"],
        episodic_summaries=["Initial implementation created app.py"],
        file_path="src/app.py",
        required_contents=["main()"],
        estimated_size="small",
        goal="Build a simple CLI app.",
        critic_issues=[
            {
                "file_path": "src/app.py",
                "line_number": 42,
                "severity": "major",
                "description": "Function main does not handle invalid input.",
                "suggested_fix": "Add input validation before processing."
            }
        ],
    )

    assert "CRITIC ISSUES:" in prompt
    assert "regenerate the complete file content" in prompt
    assert "file_path=src/app.py" in prompt

