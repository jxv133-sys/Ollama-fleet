from __future__ import annotations

from typing import Iterable


def build_coder_prompt(task_description: str, active_files: list[str], episodic_summaries: list[str]) -> str:
    return (
        "You are Coding_Agent. Produce a JSON object matching the CoderOutput schema.\n"
        "Write file modifications to implement the requested task.\n"
        "Task description: "
        + task_description
        + "\nActive files: "
        + ", ".join(active_files)
        + "\nEpisodic summaries: "
        + " | ".join(episodic_summaries)
        + "\nRespond with only valid JSON."
    )
