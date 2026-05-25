from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from ollama_fleet.agents import coder as coder_module
from ollama_fleet.agents import critic as critic_module
from ollama_fleet.agents import planner as planner_module
from ollama_fleet.agents import synthesizer as synthesizer_module
from ollama_fleet.agents import tester as tester_module
from ollama_fleet.agents.schemas import AgentOutput, AgentType, CoderOutput, CriticOutput, PlannerOutput, SynthesizerOutput, TesterOutput
from ollama_fleet.config import FleetSettings
from ollama_fleet.ollama.client import OllamaClient, OllamaTimeoutError

logger = logging.getLogger(__name__)


class AgentExecutor:
    def __init__(self, client: OllamaClient, settings: FleetSettings) -> None:
        self.client = client
        self.settings = settings

    async def execute(
        self,
        task: dict[str, Any],
        agent_type: AgentType,
        extra_context: dict[str, Any] | None = None,
    ) -> AgentOutput:
        prompt = self._build_prompt(task, agent_type, extra_context or {})
        retries = 0
        while True:
            start = time.monotonic()
            try:
                raw = await self.client.generate(
                    model=self._select_model(agent_type),
                    prompt=prompt,
                    timeout=self.settings.ollama.timeout,
                )
                parsed = self._parse_output(raw, agent_type)
                duration = time.monotonic() - start
                logger.info(
                    "AgentExecutor.success task_id=%s agent_type=%s duration=%.3f raw_len=%d",
                    task.get("task_id"),
                    agent_type.value,
                    duration,
                    len(raw),
                )
                return parsed
            except ValidationError as exc:
                duration = time.monotonic() - start
                logger.warning(
                    "AgentExecutor.validation_error task_id=%s agent_type=%s duration=%.3f error=%s",
                    task.get("task_id"),
                    agent_type.value,
                    duration,
                    exc,
                )
                if retries >= self.settings.scheduler.retry_limit:
                    raise
                retries += 1
                prompt = self._build_retry_prompt(prompt, exc)
            except ValueError as exc:
                # Treat parsing/value errors similarly to validation errors so
                # the executor can retry with a corrected prompt rather than
                # crashing the orchestrator immediately.
                duration = time.monotonic() - start
                logger.warning(
                    "AgentExecutor.parse_error task_id=%s agent_type=%s duration=%.3f error=%s",
                    task.get("task_id"),
                    agent_type.value,
                    duration,
                    exc,
                )
                if retries >= self.settings.scheduler.retry_limit:
                    raise
                retries += 1
                # Re-use the same retry prompt path as for validation failures.
                # Pass the actual exception so we can include its message
                # when asking the model to correct its output.
                prompt = self._build_retry_prompt(prompt, exc)
            except OllamaTimeoutError:
                logger.error(
                    "AgentExecutor.timeout task_id=%s agent_type=%s",
                    task.get("task_id"),
                    agent_type.value,
                )
                raise

    def _build_prompt(self, task: dict[str, Any], agent_type: AgentType, extra_context: dict[str, Any]) -> str:
        if agent_type == AgentType.PLANNER:
            return planner_module.build_planner_prompt(
                goal=task.get("goal", ""), architecture_notes=extra_context.get("architecture_notes", "")
            )
        if agent_type == AgentType.CODER:
            return coder_module.build_coder_prompt(
                task_description=task.get("description", ""),
                active_files=extra_context.get("active_files", []),
                episodic_summaries=extra_context.get("episodic_summaries", []),
            )
        if agent_type == AgentType.TESTER:
            return tester_module.build_tester_prompt(
                workspace_state=extra_context.get("workspace_state", ""),
                test_results=extra_context.get("test_results", ""),
            )
        if agent_type == AgentType.CRITIC:
            return critic_module.build_critic_prompt(
                modified_files=extra_context.get("modified_files", []),
                file_contents=extra_context.get("file_contents", {}),
                lint_results=extra_context.get("lint_results", []),
            )
        if agent_type == AgentType.SYNTHESIZER:
            return synthesizer_module.build_synthesizer_prompt(
                goal=extra_context.get("goal", ""),
                completed_summaries=extra_context.get("completed_summaries", []),
                files_produced=extra_context.get("files_produced", []),
            )
        return json.dumps(task)

    def _parse_output(self, raw: str, agent_type: AgentType) -> AgentOutput:
        raw = raw.strip()
        if not raw:
            raise ValueError(
                "Ollama returned an empty response. "
                "This can happen when the Ollama server streams NDJSON 'thinking' fragments without a final 'response' field. "
                "Verify the OllamaClient streaming behavior and inspect the raw NDJSON stream (for example via curl) to diagnose the issue. "
                f"agent_type={agent_type.value}"
            )
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ollama returned invalid JSON: {exc}") from exc

        if agent_type == AgentType.PLANNER:
            return PlannerOutput.model_validate(body)
        if agent_type == AgentType.CODER:
            return CoderOutput.model_validate(body)
        if agent_type == AgentType.TESTER:
            return TesterOutput.model_validate(body)
        if agent_type == AgentType.CRITIC:
            return CriticOutput.model_validate(body)
        if agent_type == AgentType.SYNTHESIZER:
            return SynthesizerOutput.model_validate(body)
        raise ValueError(f"Unsupported agent type: {agent_type}")

    def _build_retry_prompt(self, prompt: str, exc: Exception) -> str:
        # Accept either a Pydantic ValidationError (has .errors()) or any
        # other exception. Fall back to the exception message when errors()
        # is not available.
        try:
            details = exc.errors()  # type: ignore[attr-defined]
        except Exception:
            details = [str(exc)]
        return (
            prompt
            + "\n\nThe previous response failed validation. Please return valid JSON conforming to the schema."
            + " Validation errors: "
            + json.dumps(details)
        )

    def _select_model(self, agent_type: AgentType) -> str:
        if agent_type == AgentType.PLANNER:
            return self.settings.ollama.planner_model
        if agent_type == AgentType.CODER:
            return self.settings.ollama.coder_model
        if agent_type == AgentType.CRITIC:
            return self.settings.ollama.critic_model or self.settings.ollama.coder_model
        if agent_type == AgentType.TESTER:
            return self.settings.ollama.tester_model or self.settings.ollama.coder_model
        if agent_type == AgentType.SYNTHESIZER:
            return self.settings.ollama.summarizer_model or self.settings.ollama.coder_model
        return self.settings.ollama.coder_model
