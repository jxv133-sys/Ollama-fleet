"""Unit tests for the agent pipeline helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ollama_fleet.agents.pipeline import AgentPipeline, AgentPromptBuilder
from ollama_fleet.agents.schemas import AgentType, CoderOutput, PlannerOutput
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
    assert "clarifying_questions" in prompt
    assert "technical_requirements" in prompt


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


@pytest.mark.asyncio
async def test_run_planner_can_return_clarifying_questions() -> None:
    settings = FleetSettings()
    client = OllamaClient()
    pipeline = AgentPipeline(client=client, settings=settings)
    fake_response = '{"clarifying_questions": ["What is the target platform?"], "technical_requirements": ["REST API"], "tasks": [], "milestones": [], "architecture_notes": "Need layered architecture."}'

    mock_generate = AsyncMock(return_value=fake_response)
    with patch.object(client, "generate", mock_generate):
        result = await pipeline.run_planner(
            "Build a TODO app.",
            context="Repository is empty.",
        )

    assert result.clarifying_questions == ["What is the target platform?"]
    assert result.technical_requirements == ["REST API"]


@pytest.mark.asyncio
async def test_run_coder_sequence_runs_only_coder_tasks_in_order() -> None:
    settings = FleetSettings()
    client = OllamaClient()
    pipeline = AgentPipeline(client=client, settings=settings)

    output_a = CoderOutput(file_modifications=[], summary="A", confidence_score=0.9)
    output_b = CoderOutput(file_modifications=[], summary="B", confidence_score=0.8)

    async def fake_run_agent(agent_type, task_description, context="", timeout=None):
        return output_a if "first" in task_description else output_b

    pipeline.run_agent = AsyncMock(side_effect=fake_run_agent)

    planner_output = PlannerOutput(
        clarifying_questions=[],
        technical_requirements=["Requirement A"],
        tasks=[
            {
                "task_id": "task-2",
                "step_number": 2,
                "title": "Second task",
                "description": "Implement second feature.",
                "agent_type": "coder",
                "dependencies": ["task-1"],
                "priority": 5,
            },
            {
                "task_id": "task-1",
                "step_number": 1,
                "title": "First task",
                "description": "Implement first feature.",
                "agent_type": "coder",
                "dependencies": [],
                "priority": 5,
            },
            {
                "task_id": "task-3",
                "step_number": 3,
                "title": "Review task",
                "description": "Review output.",
                "agent_type": "tester",
                "dependencies": ["task-2"],
                "priority": 5,
            },
        ],
        milestones=[],
        architecture_notes="",
    )

    results = await pipeline.run_coder_sequence(planner_output, context="Repo context")

    assert results == [output_a, output_b]
    assert pipeline.run_agent.await_count == 2


@pytest.mark.asyncio
async def test_run_scoring_runs_critic_and_tester() -> None:
    settings = FleetSettings()
    client = OllamaClient()
    pipeline = AgentPipeline(client=client, settings=settings)

    critic_output = AsyncMock()
    tester_output = AsyncMock()

    async def fake_run_agent(agent_type, task_description, context="", timeout=None):
        return critic_output if agent_type == AgentType.CRITIC else tester_output

    pipeline.run_agent = AsyncMock(side_effect=fake_run_agent)

    critic_result, tester_result = await pipeline.run_scoring("Generated code", context="Repo context")

    assert critic_result is critic_output
    assert tester_result is tester_output
    assert pipeline.run_agent.await_count == 2
