from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from ollama_fleet.ui.event_bus import UIEventBus
from pydantic import ValidationError

from ollama_fleet.agents import coder as coder_module
from ollama_fleet.agents import critic as critic_module
from ollama_fleet.agents import planner as planner_module
from ollama_fleet.agents import specification as specification_module
from ollama_fleet.agents import synthesizer as synthesizer_module
from ollama_fleet.agents import tester as tester_module
from ollama_fleet.agents.schemas import AgentOutput, AgentType, CoderOutput, CriticOutput, PlannerOutput, SpecificationOutput, SynthesizerOutput, TesterOutput
from ollama_fleet.config import FleetSettings
from ollama_fleet.ollama.client import OllamaClient, OllamaTimeoutError

logger = logging.getLogger(__name__)


class AgentExecutor:
    def __init__(self, client: OllamaClient, settings: FleetSettings, ui_bus: UIEventBus | None = None) -> None:
        self.client = client
        self.settings = settings
        self.ui_bus = ui_bus

    def _publish_agent_progress(
        self,
        agent_type: AgentType,
        prompt: str,
        partial_response: str,
        done: bool,
    ) -> None:
        if not self.ui_bus:
            return
        self.ui_bus.publish(
            {
                "type": "agent_progress",
                "agent_type": agent_type.value,
                "prompt": prompt,
                "partial": partial_response,
                "done": done,
            }
        )

    def _publish_prompt_sent(self, agent_type: AgentType, prompt: str) -> None:
        """Publish an event so UIs can show the prompt being sent to the agent."""
        if not self.ui_bus:
            return
        self.ui_bus.publish(
            {
                "type": "prompt_sent",
                "agent_type": agent_type.value,
                "prompt": prompt,
            }
        )

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
                def _stream_callback(full_response: str, done: bool) -> None:
                    self._publish_agent_progress(agent_type, prompt, full_response, done)

                self._publish_prompt_sent(agent_type, prompt)
                raw = await self.client.generate(
                    model=self._select_model(agent_type),
                    prompt=prompt,
                    timeout=self.settings.ollama.timeout,
                    on_stream_update=_stream_callback,
                    response_format=self._select_response_format(agent_type),
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
                # Return both the parsed agent output and the prompt used
                return parsed, prompt
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
                prompt = self._build_retry_prompt(prompt, exc, agent_type)
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
                prompt = self._build_retry_prompt(prompt, exc, agent_type)
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
        if agent_type == AgentType.SPECIFICATION:
            return specification_module.build_specification_prompt(
                task_description=task.get("description", ""),
                file_path=extra_context.get("file_path"),
                required_contents=extra_context.get("required_contents", []),
                estimated_size=extra_context.get("estimated_size"),
                goal=extra_context.get("goal", task.get("goal", "")),
                active_files=extra_context.get("active_files", []),
            )
        if agent_type == AgentType.CODER:
            return coder_module.build_coder_prompt(
                task_description=task.get("description", ""),
                active_files=extra_context.get("active_files", []),
                episodic_summaries=extra_context.get("episodic_summaries", []),
                file_path=extra_context.get("file_path"),
                required_contents=extra_context.get("required_contents"),
                estimated_size=extra_context.get("estimated_size"),
                goal=extra_context.get("goal", task.get("goal", "")),
                imports=extra_context.get("imports", []),
                required_functions=extra_context.get("required_functions", []),
                required_behavior=extra_context.get("required_behavior", []),
                forbidden_behavior=extra_context.get("forbidden_behavior", []),
                purpose=extra_context.get("purpose"),
                critic_issues=extra_context.get("critic_issues", []),
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
                critic_issues=extra_context.get("critic_issues", []),
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
        # tasks may be a dict keyed by task_id — convert to list first
        raw_tasks = data.get("tasks", [])
        logger.debug(f"🔧 Normalizing planner output with {len(raw_tasks)} raw tasks")
        if isinstance(raw_tasks, dict):
            raw_tasks = [{"task_id": k, **v} if isinstance(v, dict) else {"task_id": k, "description": str(v)} for k, v in raw_tasks.items()]
            data["tasks"] = raw_tasks

        normalized: list[dict[str, Any]] = []
        for i, task in enumerate(raw_tasks):
            # Coerce non-dict entries (strings, ints, etc.) into a minimal task dict
            if not isinstance(task, dict):
                task = {"description": str(task)}

            # Map 'id' -> 'task_id'
            if "id" in task and "task_id" not in task:
                task["task_id"] = task.pop("id")

            # Ensure task_id exists
            if "task_id" not in task:
                task["task_id"] = f"task_{i + 1:03d}"

            # Ensure title exists
            if "title" not in task:
                task["title"] = task.get("description", f"Task {i + 1}")[:60]

            # Ensure description exists
            if "description" not in task:
                task["description"] = task.get("title", f"Task {i + 1}")

            # Ensure agent_type exists and is valid
            if "agent_type" not in task:
                task["agent_type"] = "coder"
            elif isinstance(task["agent_type"], list):
                task["agent_type"] = str(task["agent_type"][0]) if task["agent_type"] else "coder"
            elif isinstance(task["agent_type"], dict):
                task["agent_type"] = str(next(iter(task["agent_type"].values()), "coder"))
            elif not isinstance(task["agent_type"], str):
                task["agent_type"] = str(task["agent_type"])
            task["agent_type"] = task["agent_type"].lower().replace("_agent", "")

            # Ensure dependencies is a list
            if "dependencies" not in task or not isinstance(task["dependencies"], list):
                task["dependencies"] = []

            # Ensure priority is an int
            if "priority" not in task:
                task["priority"] = 5
            else:
                try:
                    task["priority"] = int(task["priority"])
                except (ValueError, TypeError):
                    task["priority"] = 5

            # Ensure step_number exists and is an int (required by schema)
            if "step_number" not in task:
                task["step_number"] = i + 1
                logger.debug(f"✏️ Added step_number={i + 1} to task {task.get('task_id')}")
            else:
                try:
                    task["step_number"] = int(task["step_number"])
                except (ValueError, TypeError):
                    task["step_number"] = i + 1
                    logger.debug(f"✏️ Corrected step_number to {i + 1} for task {task.get('task_id')}")

            normalized.append(task)

        data["tasks"] = normalized

        # Ensure clarifying_questions is a list of strings
        clarifying = data.get("clarifying_questions", [])
        if isinstance(clarifying, str):
            clarifying = [clarifying]
        elif isinstance(clarifying, dict):
            clarifying = [str(v) for v in clarifying.values()]
        elif clarifying is None:
            clarifying = []
        data["clarifying_questions"] = [str(item) for item in clarifying if item is not None]

        # Ensure technical_requirements is a list of strings
        requirements = data.get("technical_requirements", [])
        if isinstance(requirements, str):
            requirements = [requirements]
        elif isinstance(requirements, dict):
            requirements = [str(v) for v in requirements.values()]
        elif requirements is None:
            requirements = []
        data["technical_requirements"] = [str(item) for item in requirements if item is not None]

        # Ensure milestones is a list of strings (not objects)
        milestones = data.get("milestones", [])
        if not isinstance(milestones, list):
            milestones = [str(milestones)] if milestones else []
        data["milestones"] = [
            m.get("title", str(m)) if isinstance(m, dict) else str(m)
            for m in milestones
        ]

        # Ensure architecture_notes is a string (not list or dict)
        arch = data.get("architecture_notes", "")
        if isinstance(arch, list):
            arch = " ".join(item.get("note", str(item)) if isinstance(item, dict) else str(item) for item in arch)
        elif isinstance(arch, dict):
            arch = str(arch)
        data["architecture_notes"] = arch or ""

        logger.debug(f"✅ Normalized {len(data.get('tasks', []))} tasks for planner output")
        return data

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown fenced code blocks and return the inner code."""
        text = text.strip()
        if text.startswith("```"):
            fence_match = re.search(r'^```(?:\w+)?\n(.*)```$', text, re.DOTALL)
            if fence_match:
                return fence_match.group(1).strip()
        block_match = re.search(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        if block_match:
            return block_match.group(1).strip()
        return text

    def _normalize_coder_output(self, raw: str) -> str:
        """Normalize Coder output: extract the file contents from the model response."""
        return coder_module.normalize_coder_response(raw)

    def _normalize_critic_output(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize Critic output: ensure issue structure and detect placeholder responses."""
        # Placeholder assessments the model emits when it hasn't actually reviewed the code.
        # If we detect one, force approved=true so the revision loop doesn't spin forever.
        _PLACEHOLDER_ASSESSMENTS = {
            "overall assessment of the code changes.",
            "code has some issues that need attention.",
            "the code changes look good.",
            "no issues found.",
        }

        # Phrases in the assessment that indicate the file is empty/stub — force rejection.
        _EMPTY_CODE_PHRASES = (
            "no source code",
            "no code generated",
            "no implementation",
            "empty file",
            "placeholder",
            "stub",
            "not implemented",
        )

        assessment = str(data.get("overall_assessment", "")).strip().lower().rstrip(".")
        assessment_lower = assessment.lower()

        # If the assessment indicates missing/empty code, force rejection with an issue
        if any(phrase in assessment_lower for phrase in _EMPTY_CODE_PHRASES):
            data["approved"] = False
            if not data.get("issues"):
                data["issues"] = [{
                    "file_path": "unknown",
                    "line_number": 0,
                    "severity": "critical",
                    "description": f"Critic flagged: {data.get('overall_assessment', 'no implementation')}",
                    "suggested_fix": "Write a complete working implementation with all required functions and logic.",
                }]
            return data

        if assessment + "." in _PLACEHOLDER_ASSESSMENTS or assessment in _PLACEHOLDER_ASSESSMENTS:
            # The model returned a template string — it didn't actually review the code.
            # Force approval so we don't loop forever on a non-review.
            data["approved"] = True
            data["issues"] = []
            data["overall_assessment"] = "Auto-approved: critic returned a placeholder assessment."
            return data

        # Normalize issues list
        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        cleaned: list[dict[str, Any]] = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            if "line_number" not in issue:
                issue["line_number"] = 0
            # Drop issues that are themselves placeholder text
            desc = str(issue.get("description", "")).strip().lower()
            if desc in ("issue description", "description", "", "none"):
                continue
            cleaned.append(issue)
        data["issues"] = cleaned

        # If all issues were placeholders, approve
        if not cleaned:
            data["approved"] = True

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

    def _parse_planner_list_output(self, raw: str) -> dict[str, Any]:
        """Parse a numbered-list planner response into a PlannerOutput-compatible dict.

        Expected format per task:
            N. filename.py — Short title
               PURPOSE: ...
               EXPORTS: ...
               IMPORTS: ...
               FUNCTIONS:
                 - name(params) -> type: description
               BEHAVIOR: ...
               LINES: 100
               DEPENDS ON: 1, 2  (or "(none)")
               PRIORITY: 1
        """
        tasks: list[dict[str, Any]] = []

        # Split on task boundaries: lines that start with a number followed by a dot
        task_blocks = re.split(r'(?m)^(?=\d+\.)', raw.strip())

        for block in task_blocks:
            block = block.strip()
            if not block:
                continue

            # First line: "N. filename.py — Title"
            first_line_match = re.match(r'^(\d+)\.\s+(\S+\.py)\s*[—\-–]+\s*(.+)', block)
            if not first_line_match:
                # Try without separator: "N. Title"
                first_line_match = re.match(r'^(\d+)\.\s+(.+)', block)
                if not first_line_match:
                    continue
                step = int(first_line_match.group(1))
                filename = None
                title = first_line_match.group(2).strip()
            else:
                step = int(first_line_match.group(1))
                filename = first_line_match.group(2).strip()
                title = first_line_match.group(3).strip()

            def _extract(field: str) -> str:
                """Pull single-line field value."""
                m = re.search(rf'^\s*{field}\s*:\s*(.+)', block, re.MULTILINE | re.IGNORECASE)
                return m.group(1).strip() if m else ""

            def _extract_multiline(field: str) -> str:
                """Pull everything after 'FIELD:' until the next ALL-CAPS field."""
                m = re.search(
                    rf'^\s*{field}\s*:\s*\n?(.*?)(?=\n\s*[A-Z][A-Z ]+\s*:|$)',
                    block,
                    re.MULTILINE | re.IGNORECASE | re.DOTALL,
                )
                return m.group(1).strip() if m else ""

            purpose = _extract("PURPOSE")
            exports_raw = _extract("EXPORTS")
            imports_raw = _extract("IMPORTS")
            behavior = _extract_multiline("BEHAVIOR")
            lines_raw = _extract("LINES")
            depends_raw = _extract("DEPENDS ON")
            priority_raw = _extract("PRIORITY")

            # Parse functions block
            functions_text = _extract_multiline("FUNCTIONS")
            functions: list[dict[str, Any]] = []
            for fn_line in functions_text.splitlines():
                fn_line = fn_line.strip().lstrip("- ").strip()
                if not fn_line:
                    continue
                fn_match = re.match(r'(\w+)\s*\(([^)]*)\)\s*->\s*(\S+):\s*(.*)', fn_line)
                if fn_match:
                    functions.append({
                        "name": fn_match.group(1),
                        "params": [p.strip() for p in fn_match.group(2).split(",") if p.strip()],
                        "returns": fn_match.group(3),
                        "docstring": fn_match.group(4).strip(),
                    })
                else:
                    # Class or free-form entry
                    class_match = re.match(r'(\w+)\s*\(class\):\s*(.*)', fn_line, re.IGNORECASE)
                    if class_match:
                        functions.append({"name": class_match.group(1), "type": "class", "docstring": class_match.group(2).strip()})
                    elif fn_line:
                        functions.append({"name": fn_line.split("(")[0].strip()})

            # Parse estimated lines
            try:
                estimated_lines = int(re.search(r'\d+', lines_raw).group()) if lines_raw else 100
            except (AttributeError, ValueError):
                estimated_lines = 100

            # Parse dependencies: reference by task step number → convert to task_id strings
            dependencies: list[str] = []
            if depends_raw and depends_raw.lower() not in ("(none)", "none", ""):
                for part in re.split(r'[,\s]+', depends_raw):
                    part = part.strip()
                    if re.match(r'^\d+$', part):
                        dependencies.append(f"task_{int(part):03d}")
                    elif part and part.lower() not in ("none", "(none)"):
                        dependencies.append(part)

            # Parse priority
            try:
                priority = int(re.search(r'\d+', priority_raw).group()) if priority_raw else step
            except (AttributeError, ValueError):
                priority = step

            # Parse imports
            imports: list[str] = []
            if imports_raw and imports_raw.lower() not in ("(none)", "none", ""):
                for imp in imports_raw.split(","):
                    imp = imp.strip()
                    if imp:
                        imports.append(imp)

            task_id = f"task_{step:03d}"
            task: dict[str, Any] = {
                "task_id": task_id,
                "step_number": step,
                "title": title,
                "description": purpose or title,
                "agent_type": "coder",
                "dependencies": dependencies,
                "priority": priority,
                "filename": filename,
                "file_spec": {
                    "purpose": purpose,
                    "public_exports": [e.strip() for e in exports_raw.split(",") if e.strip()] if exports_raw else [],
                    "imports": imports,
                    "functions": functions,
                    "exact_content": behavior,
                    "estimated_lines": estimated_lines,
                },
            }
            tasks.append(task)

        if not tasks:
            raise ValueError(
                f"Planner returned no parseable tasks from list output. "
                f"Raw response (first 300 chars): {raw[:300]}"
            )

        logger.info("Parsed %d tasks from planner list output", len(tasks))
        return {
            "tasks": tasks,
            "clarifying_questions": [],
            "technical_requirements": [],
            "milestones": [],
            "architecture_notes": "",
        }

    def _parse_output(self, raw: str, agent_type: AgentType) -> AgentOutput:
        raw = raw.strip()
        if not raw:
            raise ValueError(
                "Ollama returned an empty response. "
                "This can happen when the Ollama server streams NDJSON 'thinking' fragments without a final 'response' field. "
                "Verify the OllamaClient streaming behavior and inspect the raw NDJSON stream (for example via curl) to diagnose the issue. "
                f"agent_type={agent_type.value}"
            )

        # Detect obviously truncated responses (e.g. model returned just "{" or a few chars)
        if agent_type != AgentType.CODER and len(raw) < 20:
            raise ValueError(
                f"Response too short to be valid ({len(raw)} chars): {raw!r}. "
                "The model likely truncated its output. Please try again."
            )

        if agent_type == AgentType.CODER:
            content = self._normalize_coder_output(raw)
            return CoderOutput.model_validate({"content": content})

        # Planner: try numbered list first; fall back to JSON if the model returned JSON anyway
        if agent_type == AgentType.PLANNER:
            if raw.lstrip().startswith(("{", "[")):
                # Model returned JSON despite the prompt asking for a list — parse it
                logger.warning("Planner returned JSON instead of a numbered list; falling back to JSON parsing")
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    # Try to salvage a complete {...} block from a partial response
                    match = re.search(r'\{.*\}', raw, re.DOTALL)
                    if match:
                        try:
                            body = json.loads(match.group(0))
                        except json.JSONDecodeError:
                            raise ValueError(
                                f"Planner returned truncated or malformed JSON. "
                                f"Response was {len(raw)} chars starting with: {raw[:100]!r}"
                            )
                    else:
                        raise ValueError(
                            f"Planner returned truncated JSON (no complete object found). "
                            f"Response was {len(raw)} chars: {raw[:100]!r}"
                        )
            else:
                body = self._parse_planner_list_output(raw)
            body = self._normalize_planner_output(body)
            return PlannerOutput.model_validate(body)

        # All other agents: parse JSON
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract the first {...} or [...] block
            match = re.search(r'({.*}|\[.*\])', raw, re.DOTALL)
            if match:
                try:
                    body = json.loads(match.group(1))
                except Exception as exc2:
                    raise ValueError(f"Ollama returned invalid JSON (after extraction): {exc2}")
            else:
                raise ValueError(f"Ollama returned invalid JSON: {raw[:200]}")

        # Apply agent-specific normalization before validation
        if agent_type == AgentType.SPECIFICATION:
            return SpecificationOutput.model_validate(body)
        if agent_type == AgentType.TESTER:
            return TesterOutput.model_validate(body)
        if agent_type == AgentType.CRITIC:
            body = self._normalize_critic_output(body)
            return CriticOutput.model_validate(body)
        if agent_type == AgentType.SYNTHESIZER:
            body = self._normalize_synthesizer_output(body)
            return SynthesizerOutput.model_validate(body)
        raise ValueError(f"Unsupported agent type: {agent_type}")


    def _build_retry_prompt(self, prompt: str, exc: Exception, agent_type: AgentType | None = None) -> str:
        # Accept either a Pydantic ValidationError (has .errors()) or any
        # other exception. Fall back to the exception message when errors()
        # is not available.
        try:
            details = exc.errors()  # type: ignore[attr-defined]
        except Exception:
            details = [str(exc)]

        if agent_type == AgentType.PLANNER:
            return (
                prompt
                + "\n\nThe previous response could not be parsed."
                + " You MUST respond with ONLY a numbered list using the exact format shown in the example."
                + " Do NOT use JSON, markdown, or any other format."
                + " Each task must start with 'N. filename.py — Title' on its own line."
                + " Parsing error: "
                + json.dumps(details)
            )
        return (
            prompt
            + "\n\nThe previous response failed validation or parsing."
            + " For coder tasks, please return raw file contents only."
            + " For all other tasks, please return valid JSON conforming to the schema."
            + " Validation/parsing errors: "
            + json.dumps(details)
        )

    def _select_response_format(self, agent_type: AgentType) -> str | None:
        """Return the Ollama response format for this agent.

        Planner and Coder produce free-text output (list and code respectively),
        so we must NOT constrain them to JSON mode. All other agents return JSON.
        """
        if agent_type in (AgentType.PLANNER, AgentType.CODER):
            return None
        return "json"

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
