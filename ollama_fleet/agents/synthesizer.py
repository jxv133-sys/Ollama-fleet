from __future__ import annotations

from typing import Iterable


def build_synthesizer_prompt(
    goal: str,
    completed_summaries: list[str],
    files_produced: list[str],
) -> str:
    lines: list[str] = [
        "You are Synthesizer_Agent. Produce a JSON object matching the SynthesizerOutput schema.",
        "Summarize the completed work, list produced files, and propose next steps.",
        "Respond with only valid JSON.",
        f"Goal: {goal}",
        "\nCompleted task summaries:\n",
    ]
    if completed_summaries:
        lines.extend(completed_summaries)
    else:
        lines.append("No completed summaries available.")
    lines.append("\nFiles produced:\n")
    lines.extend(files_produced or ["(none)"])
    return "\n".join(lines)
