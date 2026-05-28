from __future__ import annotations

from typing import Iterable


def build_synthesizer_prompt(
    goal: str,
    completed_summaries: list[str],
    files_produced: list[str],
) -> str:
    schema_example = """{
  "summary": "Comprehensive summary of all completed work",
  "changelog": ["Change 1", "Change 2"],
  "files_produced": ["src/file1.py", "src/file2.py"],
  "next_steps": ["Step 1", "Step 2"]
}"""
    lines: list[str] = [
        "You are Synthesizer_Agent. Your ONLY task is to return valid JSON.",
        "\nRESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT.",
        "\nJSON SCHEMA REQUIREMENTS:",
        "- summary: STRING summarizing all completed work",
        "- changelog: array of STRINGS describing each change",
        "- files_produced: array of STRINGS listing all produced files",
        "- next_steps: array of STRINGS proposing next steps",
        "\nEXAMPLE OUTPUT:",
        schema_example,
        f"\nGoal: {goal}",
        "\nCompleted task summaries:\n",
    ]
    if completed_summaries:
        lines.extend(completed_summaries)
    else:
        lines.append("No completed summaries available.")
    lines.append("\nFiles produced:\n")
    lines.extend(files_produced or ["(none)"])
    return "\n".join(lines)
