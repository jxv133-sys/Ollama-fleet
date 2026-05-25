from __future__ import annotations

from typing import Iterable


def build_coder_prompt(task_description: str, active_files: list[str], episodic_summaries: list[str]) -> str:
    schema_example = """{
  "file_modifications": [
    {
      "file_path": "src/module.py",
      "operation": "create",
      "content": "# File content here..."
    }
  ],
  "summary": "Summary of changes made",
  "confidence_score": 0.95
}"""
    return (
        "You are Coding_Agent. Your ONLY task is to return valid JSON.\n"
        "\nRESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT.\n"
        "\nJSON SCHEMA REQUIREMENTS:\n"
        "- file_modifications: array of objects, EACH with: file_path (string), operation (string: 'create'|'modify'|'delete'), content (string - empty for delete)\n"
        "- summary: single STRING describing changes\n"
        "- confidence_score: number between 0.0 and 1.0\n"
        "\nEXAMPLE OUTPUT:\n"
        + schema_example
        + "\n\nTask to implement:\n"
        "Description: "
        + task_description
        + "\nActive files: "
        + (', '.join(active_files) if active_files else "(none)")
        + "\nContext: "
        + (' | '.join(episodic_summaries) if episodic_summaries else "(no prior context)")
        + "\n\nRETURN ONLY VALID JSON MATCHING THE SCHEMA ABOVE."
    )
