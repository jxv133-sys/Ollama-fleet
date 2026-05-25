from __future__ import annotations


def build_planner_prompt(goal: str, architecture_notes: str) -> str:
    schema_example = """{
  "tasks": [
    {
      "task_id": "task_001",
      "title": "Brief task title",
      "description": "Detailed task description",
      "agent_type": "coder",
      "dependencies": [],
      "priority": 1
    }
  ],
  "milestones": ["Milestone 1 description", "Milestone 2 description"],
  "architecture_notes": "Overall architecture description and design decisions."
}"""
    return (
        "You are Planner_Agent. Your ONLY task is to return valid JSON.\n"
        "\nRESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT.\n"
        "\nJSON SCHEMA REQUIREMENTS:\n"
        "- tasks: array of objects, EACH with: task_id (string), title (string), description (string), agent_type (string: 'coder'|'tester'|'synthesizer'), dependencies (array of strings), priority (integer 1-10)\n"
        "- milestones: array of STRINGS only (not objects)\n"
        "- architecture_notes: single STRING (not array, not object)\n"
        "\nEXAMPLE OUTPUT:\n"
        + schema_example
        + "\n\nNOW plan the following goal:\n"
        "Goal: "
        + goal
        + "\nArchitecture notes: "
        + architecture_notes
        + "\n\nRETURN ONLY VALID JSON MATCHING THE SCHEMA ABOVE."
    )
