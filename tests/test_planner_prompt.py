from __future__ import annotations

from ollama_fleet.agents import planner as planner_module
from ollama_fleet.agents.executor import AgentExecutor


def test_build_planner_prompt_includes_required_format() -> None:
    prompt = planner_module.build_planner_prompt("Build an API server", "")

    assert prompt.startswith("You are a software project planning agent.")
    assert "Do NOT write code." in prompt
    assert "Do NOT explain your reasoning." in prompt
    assert "Return ONLY valid JSON." in prompt
    assert "Response format:" in prompt
    assert "Goal: Build an API server" in prompt
    assert "Create one task per file." in prompt or "Create one task per file" in prompt


def test_normalize_planner_output_adds_missing_optional_lists() -> None:
    executor = AgentExecutor(client=None, settings=None)
    raw = {
        "tasks": [
            {
                "task_id": "task_001",
                "title": "A",
                "description": "B",
                "agent_type": "coder",
                "dependencies": [],
                "priority": 1,
            }
        ],
        "milestones": ["done"],
        "architecture_notes": "notes",
    }

    normalized = executor._normalize_planner_output(raw)

    assert normalized["clarifying_questions"] == []
    assert normalized["technical_requirements"] == []
    assert normalized["milestones"] == ["done"]
    assert normalized["architecture_notes"] == "notes"


def test_normalize_planner_output_coerces_string_fields() -> None:
    executor = AgentExecutor(client=None, settings=None)
    raw = {
        "tasks": [],
        "clarifying_questions": "Need more info",
        "technical_requirements": "Use FastAPI",
        "milestones": ["done"],
        "architecture_notes": "notes",
    }

    normalized = executor._normalize_planner_output(raw)

    assert normalized["clarifying_questions"] == ["Need more info"]
    assert normalized["technical_requirements"] == ["Use FastAPI"]
