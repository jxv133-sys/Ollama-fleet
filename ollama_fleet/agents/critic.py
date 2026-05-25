from __future__ import annotations

from typing import Any


def build_critic_prompt(
    modified_files: list[str],
    file_contents: dict[str, str],
    lint_results: list[dict[str, str]],
    critic_issues: list[dict[str, Any]] | None = None,
) -> str:
    schema_example = """{
  "approved": false,
  "issues": [
    {
      "file_path": "src/module.py",
      "line_number": 42,
      "severity": "major",
      "description": "Issue description",
      "suggested_fix": "How to fix it"
    }
  ],
  "overall_assessment": "Overall assessment of the code changes."
}"""
    lines: list[str] = [
        "You are Critic_Agent. Your ONLY task is to return valid JSON.",
        "\nRESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT.",
        "\nJSON SCHEMA REQUIREMENTS:",
        "- approved: boolean (true if code is acceptable, false if issues found)",
        "- issues: array of objects, EACH with: file_path (string), line_number (integer, 0 for file-level), severity (string: 'critical'|'major'|'minor'), description (string), suggested_fix (string)",
        "- overall_assessment: single STRING summarizing the review",
        "\nEXAMPLE OUTPUT:",
        schema_example,
        "\nReview the following code modifications:",
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
