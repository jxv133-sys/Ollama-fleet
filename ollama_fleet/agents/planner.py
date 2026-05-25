from __future__ import annotations


def build_planner_prompt(goal: str, architecture_notes: str) -> str:
    return (
        "You are Planner_Agent. Produce a JSON object matching the PlannerOutput schema.\n"
        "Include a list of tasks, milestones, and architecture notes.\n"
        "Goal: "
        + goal
        + "\nArchitecture notes: "
        + architecture_notes
        + "\nRespond with only valid JSON."
    )
