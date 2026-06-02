from __future__ import annotations

import json
import re
from typing import Any, Iterable


def build_coder_prompt(
  task_description: str,
  active_files: list[str],
  episodic_summaries: list[str],
  file_path: str | None = None,
  required_contents: list[str] | None = None,
  estimated_size: str | None = None,
  goal: str | None = None,
  imports: list[str] | None = None,
  required_functions: list[str] | None = None,
  required_behavior: list[str] | None = None,
  forbidden_behavior: list[str] | None = None,
  purpose: str | None = None,
  critic_issues: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        "You are a File Generation Agent.",
        "",
        "Your only responsibility is generating the complete contents of a single file.",
        "Return only the file contents.",
        "Do not explain your work.",
        "Do not generate markdown.",
        "Do not generate code fences.",
        "Do not generate JSON.",
        "",
        "Generate a complete implementation that satisfies the provided specification.",
        "",
        "TARGET FILE:",
        file_path or "(unspecified)",
        "",
    ]
    if purpose:
        lines.extend(["PURPOSE:", purpose, ""])
    if imports:
        lines.append("REQUIRED IMPORTS:")
        lines.extend([f"- {imp}" for imp in imports])
        lines.append("")
    if required_functions:
        lines.append("REQUIRED FUNCTIONS:")
        lines.extend([f"- {fn}" for fn in required_functions])
        lines.append("")
    if required_behavior:
        lines.append("REQUIRED BEHAVIOR:")
        lines.extend([f"- {behavior}" for behavior in required_behavior])
        lines.append("")
    if forbidden_behavior:
        lines.append("FORBIDDEN BEHAVIOR:")
        lines.extend([f"- {behavior}" for behavior in forbidden_behavior])
        lines.append("")
    if required_contents:
        lines.append("REQUIRED SYMBOLS:")
        lines.extend([f"- {symbol}" for symbol in required_contents])
        lines.append("")
    if goal:
        lines.extend(["GOAL:", goal, ""])
    if task_description:
        lines.extend(["TASK DESCRIPTION:", task_description, ""])
    if active_files:
        lines.append("ACTIVE FILES:")
        lines.extend([f"- {path}" for path in active_files])
        lines.append("")
    if episodic_summaries:
        lines.append("CONTEXT:")
        lines.extend([f"- {summary}" for summary in episodic_summaries])
        lines.append("")
    if critic_issues:
        lines.append("CRITIC ISSUES:")
        for issue in critic_issues:
            lines.append(
                f"- file_path={issue.get('file_path', '')}, line_number={issue.get('line_number', '')}, "
                f"severity={issue.get('severity', '')}, description={issue.get('description', '')}"
            )
        lines.append("")
        lines.append(
            "Regenerate the entire file content to resolve the critic issues. "
            "Do not output patch diffs or partial edits."
        )
    lines.extend([
        "",
        "REMEMBER:",
        "- Return only the contents of the requested file.",
        "- Do not include markdown, code fences, JSON, or any explanation.",
        "- Do not invent new public interfaces. Implement the provided specification.",
        "- If a file is being revised, regenerate the complete file content.",
    ])
    return "\n".join(lines)


def normalize_coder_response(raw: str) -> str:
    """Extract raw source code from coder responses, including code-fence wrapped output."""
    content = raw.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        file_mods = data.get("file_modifications")
        if isinstance(file_mods, list):
            if len(file_mods) != 1:
                raise ValueError(
                    "Coder should generate one file at a time. "
                    "If using legacy structured output, return exactly one file_modifications entry."
                )
            first = file_mods[0]
            if isinstance(first, dict) and "content" in first:
                content = str(first["content"])
            elif "content" in data:
                content = str(data["content"])
        elif "content" in data:
            content = str(data["content"])

    content = content.strip()
    if content.startswith("```"):
        fence_match = re.search(r'^```(?:\w+)?\n(.*)```$', content, re.DOTALL)
        if fence_match:
            content = fence_match.group(1).strip()
        else:
            content = re.sub(r'^```(?:\w+)?\n', "", content)
            content = re.sub(r'```\s*$', "", content).strip()

    placeholder_markers = [
        "replace with actual file content",
        "example module content here",
        "example file content here",
    ]
    if any(marker in content.lower() for marker in placeholder_markers):
        raise ValueError(
            "Coder output contains placeholder or example content; request a real file implementation"
        )

    if not content:
        raise ValueError("Coder returned empty file contents; request full file content")

    return content
