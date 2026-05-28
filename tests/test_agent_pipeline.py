"""Unit tests for the agent pipeline helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ollama_fleet.agents.pipeline import AgentPipeline, AgentPromptBuilder
from ollama_fleet.agents.schemas import AgentType
from ollama_fleet.config import FleetSettings
from ollama_fleet.ollama.client import OllamaClient


def test_build_prompt_contains_json_instructions() -> None:
    prompt = AgentPromptBuilder.build_prompt(
        AgentType.PLANNER,
        task_description="Define tasks for feature X.",
        context="Existing repository has module A.",
    )

    assert "TASK:" in prompt
    assert "CONTEXT:" in prompt
    assert "parsable JSON only" in prompt
    assert "tasks" in prompt


def test_select_model_falls_back_to_coder_model() -> None:
    settings = FleetSettings(
        ollama={
            "planner_model": "planner-model",
            "coder_model": "coder-model",
            "critic_model": None,
            "tester_model": None,
            "summarizer_model": "summarizer-model",
            "timeout": 1200.0,
        }
    )
    pipeline = AgentPipeline(client=OllamaClient(), settings=settings)

    assert pipeline.select_model(AgentType.PLANNER) == "planner-model"
    assert pipeline.select_model(AgentType.CODER) == "coder-model"
    assert pipeline.select_model(AgentType.CRITIC) == "coder-model"
    assert pipeline.select_model(AgentType.TESTER) == "coder-model"
    assert pipeline.select_model(AgentType.SYNTHESIZER) == "summarizer-model"


@pytest.mark.asyncio
async def test_run_agent_calls_client_and_parses_output() -> None:
    settings = FleetSettings()
    client = OllamaClient()
    pipeline = AgentPipeline(client=client, settings=settings)
    fake_response = '{"approved": true, "issues": [], "overall_assessment": "Good."}'

    mock_generate = AsyncMock(return_value=fake_response)
    with patch.object(client, "generate", mock_generate):
        result = await pipeline.run_agent(
            AgentType.CRITIC,
            task_description="Review code quality.",
            context="No extra context.",
            timeout=10.0,
        )

    assert result.approved is True
    assert result.issues == []
    mock_generate.assert_awaited_once()
