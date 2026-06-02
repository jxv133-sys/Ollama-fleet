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
        "You are a Python File Generator.",
        "",
        "Your job: write the complete contents of ONE file.",
        "All files are in the same directory.",
        "",
        "Output ONLY file contents. No explanations, markdown, code fences, or JSON.",
        "",
    ]

    if file_path:
        lines.extend([f"FILE TO WRITE: {file_path}", ""])

    if task_description:
        lines.extend(["TASK:", task_description, ""])

    if purpose:
        lines.extend(["PURPOSE:", purpose, ""])

    if required_functions:
        lines.append("FUNCTIONS TO IMPLEMENT:")
        lines.extend([f"- {fn}" for fn in required_functions])
        lines.append("")

    if imports:
        lines.append("IMPORTS NEEDED:")
        lines.extend([f"- {imp}" for imp in imports])
        lines.append("")

    if required_contents:
        lines.append("PUBLIC EXPORTS:")
        lines.extend([f"- {symbol}" for symbol in required_contents])
        lines.append("")

    if required_behavior:
        lines.append("EXACT REQUIREMENTS:")
        lines.extend([f"- {behavior}" for behavior in required_behavior])
        lines.append("")

    if episodic_summaries:
        lines.append("CONTEXT FROM PREVIOUS FILES:")
        lines.extend([f"- {summary}" for summary in episodic_summaries])
        lines.append("")

    if active_files:
        lines.append("OTHER FILES IN PROJECT:")
        lines.extend([f"- {path}" for path in active_files])
        lines.append("")

    if critic_issues:
        lines.append("FIX THESE ISSUES:")
        for issue in critic_issues:
            lines.append(
                f"- Line {issue.get('line_number', '?')}: {issue.get('description', '')} "
                f"(severity: {issue.get('severity', 'unknown')})"
            )
        lines.append("")
        lines.append("Regenerate the entire file to fix all issues.")

    lines.extend([
        "",
        "RULES:",
        "- Complete file implementation only.",
        "- No markdown, fences, JSON, or explanation.",
        "- All files are in the same directory (use direct imports, no paths).",
        "- If rewriting a file, output the whole thing.",
        "- Implement the exact specification provided.",
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
