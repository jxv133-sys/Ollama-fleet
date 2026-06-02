"""Agent pipeline helpers for Ollama Fleet.

This module centralizes prompt construction, model selection, and output
schema validation for planner, coder, critic, tester, and synthesizer agents.
"""

from __future__ import annotations

from pydantic import ValidationError

from ollama_fleet.agents import coder as coder_module
from ollama_fleet.agents.schemas import (
    AgentOutput,
    AgentType,
    CoderOutput,
    CriticOutput,
    PlannerOutput,
    SynthesizerOutput,
    TesterOutput,
)
from ollama_fleet.config import FleetSettings
from ollama_fleet.ollama.client import OllamaClient


_PROMPT_TEMPLATES: dict[AgentType, str] = {
    AgentType.PLANNER: (
        "You are a planning agent. Analyze the goal and existing context. "
        "If details are missing, ask clarifying questions before producing final plan. "
        "Build a numbered task list that covers the full project and includes technical requirements. "
        "Output valid JSON only."
        "\n\nRequired keys: clarifying_questions, technical_requirements, tasks, milestones, architecture_notes."
        "\nTask list item: task_id, step_number, title, description, agent_type, dependencies, priority."
    ),
    AgentType.CODER: (
        "You are a coding agent. Generate the complete contents of a single file. "
        "Return only raw source code. Do not output JSON, markdown, code fences, or explanation."
    ),
    AgentType.CRITIC: (
        "You are a critic agent. Review the code and report any issues. Output valid JSON only."
        "\n\nRequired keys: approved, issues, overall_assessment."
        "\nIssue items: file_path, line_number, severity, description, suggested_fix."
    ),
    AgentType.TESTER: (
        "You are a tester agent. Evaluate the code using the provided tests and context. "
        "Output valid JSON only."
        "\n\nRequired keys: tests_passed, tests_failed, failures, ready_for_review."
        "\nFailure items: test_name, error_message, suggested_fix."
    ),
    AgentType.SYNTHESIZER: (
        "You are a synthesizer agent. Summarize the completed work and propose next steps. "
        "Output valid JSON only."
        "\n\nRequired keys: summary, changelog, files_produced, next_steps."
    ),
}

_SCHEMA_BY_AGENT: dict[AgentType, type[AgentOutput]] = {
    AgentType.PLANNER: PlannerOutput,
    AgentType.CODER: CoderOutput,
    AgentType.CRITIC: CriticOutput,
    AgentType.TESTER: TesterOutput,
    AgentType.SYNTHESIZER: SynthesizerOutput,
}


class AgentPromptBuilder:
    """Build consistent prompts for each agent type."""

    @staticmethod
    def build_prompt(
        agent_type: AgentType,
        task_description: str,
        context: str = "",
    ) -> str:
        """Construct a prompt that is explicit about format and responsibilities."""
        template = _PROMPT_TEMPLATES[agent_type]
        prompt_parts = [template]

        if task_description:
            prompt_parts.append(f"\n\nTASK:\n{task_description}")

        if context:
            prompt_parts.append(f"\n\nCONTEXT:\n{context}")

        if agent_type == AgentType.CODER:
            prompt_parts.append(
                "\n\nImportant: return only raw source code for the requested file. "
                "Do not include markdown, explanation, code fences, or JSON."
            )
        else:
            prompt_parts.append(
                "\n\nImportant: return parsable JSON only. Do not include markdown, explanation, or code fences."
            )
        return "".join(prompt_parts)


class AgentPipeline:
    """Lightweight agent orchestration and validation layer."""

    def __init__(self, client: OllamaClient, settings: FleetSettings) -> None:
        self.client = client
        self.settings = settings
        self.default_timeout = settings.ollama.timeout

    def select_model(self, agent_type: AgentType) -> str:
        """Choose the appropriate model for the agent type, with explicit fallback."""
        if agent_type is AgentType.PLANNER:
            return self.settings.ollama.planner_model
        if agent_type is AgentType.CODER:
            return self.settings.ollama.coder_model
        if agent_type is AgentType.CRITIC:
            return self.settings.ollama.critic_model or self.settings.ollama.coder_model
        if agent_type is AgentType.TESTER:
            return self.settings.ollama.tester_model or self.settings.ollama.coder_model
        if agent_type is AgentType.SYNTHESIZER:
            return self.settings.ollama.summarizer_model
        raise ValueError(f"Unsupported agent type: {agent_type}")

    def parse_agent_output(self, agent_type: AgentType, output_text: str) -> AgentOutput:
        """Validate model response against the expected agent schema."""
        if agent_type == AgentType.CODER:
            content = coder_module.normalize_coder_response(output_text)
            return CoderOutput.model_validate({"content": content})

        schema_cls = _SCHEMA_BY_AGENT[agent_type]
        try:
            return schema_cls.model_validate_json(output_text)
        except ValidationError as exc:
            raise ValueError(
                "Agent output did not match expected schema."
                f" AgentType={agent_type.value}, error={exc}"
            ) from exc

    async def run_agent(
        self,
        agent_type: AgentType,
        task_description: str,
        context: str = "",
        timeout: float | None = None,
    ) -> AgentOutput:
        """Execute an agent end-to-end: prompt build, model call, and validation."""
        prompt = AgentPromptBuilder.build_prompt(agent_type, task_description, context)
        model = self.select_model(agent_type)
        response = await self.client.generate(model, prompt, timeout or self.default_timeout)
        return self.parse_agent_output(agent_type, response)

    async def run_planner(self, user_prompt: str, context: str = "") -> PlannerOutput:
        """Run the planning agent against a user prompt and context."""
        return await self.run_agent(AgentType.PLANNER, user_prompt, context)  # type: ignore[return-value]

    async def run_coder_sequence(
        self,
        planner_output: PlannerOutput,
        context: str = "",
    ) -> list[CoderOutput]:
        """Execute coder tasks sequentially in step order."""
        coder_outputs: list[CoderOutput] = []
        ordered_tasks = sorted(planner_output.tasks, key=lambda task: task.step_number)

        for task in ordered_tasks:
            if task.agent_type != "coder":
                continue
            task_context = (
                f"{context}\n\nPrevious tasks:\n"
                + "\n".join(
                    f"{t.step_number}. {t.title} ({t.agent_type})" for t in ordered_tasks if t.step_number < task.step_number
                )
            )
            result = await self.run_agent(
                AgentType.CODER,
                task_description=task.description,
                context=task_context,
            )
            coder_outputs.append(result)

        return coder_outputs

    async def run_scoring(
        self,
        code_context: str,
        context: str = "",
    ) -> tuple[CriticOutput, TesterOutput]:
        """Run critic and tester agents after coding is complete."""
        critic_output = await self.run_agent(
            AgentType.CRITIC,
            task_description="Review the produced code and identify issues.",
            context=code_context,
        )  # type: ignore[return-value]
        tester_output = await self.run_agent(
            AgentType.TESTER,
            task_description="Run tests against the produced code and report failures.",
            context=code_context,
        )  # type: ignore[return-value]

        return critic_output, tester_output

    def build_prompt_with_model(self, agent_type: AgentType, task_description: str, context: str = "") -> tuple[str, str]:
        """Return both selected model and final prompt text."""
        return self.select_model(agent_type), AgentPromptBuilder.build_prompt(agent_type, task_description, context)
