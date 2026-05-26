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

    def _normalize_planner_output(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize Planner output: fix field names, types, flatten structures."""
        # Normalize tasks: map 'id' -> 'task_id' and ensure all required fields
        for i, task in enumerate(data.get("tasks", [])):
            if "id" in task and "task_id" not in task:
                task["task_id"] = task.pop("id")
            # Ensure title exists
            if "title" not in task and "description" in task:
                task["title"] = task["description"][:50]
            # Ensure agent_type exists
            if "agent_type" not in task:
                task["agent_type"] = "coder"
        
        # Flatten milestones if they're objects
        if data.get("milestones") and isinstance(data["milestones"][0], dict):
            data["milestones"] = [
                m.get("title", str(m)) for m in data["milestones"]
            ]
        
        # Flatten architecture_notes if it's a list
        if data.get("architecture_notes") and isinstance(data["architecture_notes"], list):
            data["architecture_notes"] = " ".join([
                n.get("note", str(n)) for n in data["architecture_notes"]
            ])
        
        return data

    def _normalize_coder_output(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize Coder output: ensure required fields."""
        # Normalize file_modifications
        for mod in data.get("file_modifications", []):
            if "path" in mod and "file_path" not in mod:
                mod["file_path"] = mod.pop("path")
            if "content" not in mod:
                mod["content"] = ""
        
        # Ensure confidence_score exists and is in range
        if "confidence_score" not in data:
            data["confidence_score"] = 0.5
        else:
            try:
                score = float(data["confidence_score"])
                data["confidence_score"] = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                data["confidence_score"] = 0.5
        
        return data

    def _normalize_critic_output(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize Critic output: ensure issue structure."""
        # Normalize issues
        for issue in data.get("issues", []):
            if "line_number" not in issue:
                issue["line_number"] = 0
        
        return data

    def _normalize_synthesizer_output(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize Synthesizer output: coerce common model shape drift."""
        key_aliases = {
            "changes": "changelog",
            "change_log": "changelog",
            "files": "files_produced",
            "produced_files": "files_produced",
            "nextSteps": "next_steps",
            "nextStep": "next_steps",
            "next_step": "next_steps",
            "recommendations": "next_steps",
        }
        for old_key, new_key in key_aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data.pop(old_key)

        if "next_steps" not in data:
            step_items = [
                value
                for key, value in sorted(data.items())
                if isinstance(key, str) and key.lower().startswith("step ")
            ]
            if step_items:
                data["next_steps"] = step_items

        for key in ("changelog", "files_produced", "next_steps"):
            value = data.get(key, [])
            if isinstance(value, list):
                data[key] = [str(item) for item in value if item is not None]
            elif value is None:
                data[key] = []
            elif isinstance(value, dict):
                data[key] = [str(item) for item in value.values() if item is not None]
            else:
                data[key] = [str(value)]

        if "summary" not in data or data["summary"] is None:
            data["summary"] = ""
        else:
            data["summary"] = str(data["summary"])

        return data

    def _parse_output(self, raw: str, agent_type: AgentType) -> AgentOutput:
        raw = raw.strip()
        if not raw:
            raise ValueError(
                "Ollama returned an empty response. "
                "This can happen when the Ollama server streams NDJSON 'thinking' fragments without a final 'response' field. "
                "Verify the OllamaClient streaming behavior and inspect the raw NDJSON stream (for example via curl) to diagnose the issue. "
                f"agent_type={agent_type.value}"
            )
        # Extraction wrapper: try to extract JSON from messy output
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract the first {...} or [...] block
            import re
            match = re.search(r'({.*}|\[.*\])', raw, re.DOTALL)
            if match:
                try:
                    body = json.loads(match.group(1))
                except Exception as exc2:
                    raise ValueError(f"Ollama returned invalid JSON (after extraction): {exc2}")
            else:
                raise ValueError(f"Ollama returned invalid JSON: {raw[:200]}")

        # Apply agent-specific normalization before validation
        if agent_type == AgentType.PLANNER:
            body = self._normalize_planner_output(body)
            return PlannerOutput.model_validate(body)
        if agent_type == AgentType.CODER:
            body = self._normalize_coder_output(body)
            return CoderOutput.model_validate(body)
        if agent_type == AgentType.TESTER:
            return TesterOutput.model_validate(body)
        if agent_type == AgentType.CRITIC:
            body = self._normalize_critic_output(body)
            return CriticOutput.model_validate(body)
        if agent_type == AgentType.SYNTHESIZER:
            body = self._normalize_synthesizer_output(body)
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
