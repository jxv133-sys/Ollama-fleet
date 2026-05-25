from __future__ import annotations

from typing import Any


def build_critic_prompt(
    modified_files: list[str],
    file_contents: dict[str, str],
    lint_results: list[dict[str, str]],
    critic_issues: list[dict[str, Any]] | None = None,
) -> str:
    lines: list[str] = [
        "You are Critic_Agent. Review the provided code modifications and return a JSON object matching the CriticOutput schema.",
        "Evaluate code quality, correctness, and adherence to best practices.",
        "Include all issues found and provide an overall assessment.",
        "Respond with only valid JSON.",
        "\nModified files:\n",
    ]
    for path in modified_files:
        lines.append(f"- {path}")
    lines.append("\nFile contents:\n")
    for path in modified_files:
        lines.append(f"=== {path} ===")
        lines.append(file_contents.get(path, ""))
    lines.append("\nLint results:\n")
    if not lint_results:
        lines.append("No lint issues reported.")
    else:
        for issue in lint_results:
            lines.append(
                f"- {issue.get('file_path', '')}:{issue.get('line_number', '')} "
                f"[{issue.get('code', '')}] {issue.get('message', '')}"
            )

    if critic_issues:
        lines.append("\nPrevious critique issues:\n")
        for issue in critic_issues:
            lines.append(
                f"- {issue.get('file_path', '')}:{issue.get('line_number', '')} "
                f"{issue.get('severity', '')} {issue.get('description', '')} "
                f"Suggested fix: {issue.get('suggested_fix', '')}"
            )
    return "\n".join(lines)
