from __future__ import annotations


def build_planner_prompt(goal: str, architecture_notes: str) -> str:
    schema_example = """{
  "tasks": [
    {
      "task_id": "task_001",
      "title": "Implement core data models",
      "description": "Create src/models.py with all data classes and type definitions needed by the application. Include docstrings and type hints.",
      "agent_type": "coder",
      "dependencies": [],
      "priority": 1
    },
    {
      "task_id": "task_002",
      "title": "Implement business logic",
      "description": "Create src/logic.py with the main application logic. Import from models.py. Handle edge cases and errors.",
      "agent_type": "coder",
      "dependencies": ["task_001"],
      "priority": 2
    },
    {
      "task_id": "task_003",
      "title": "Implement CLI entry point",
      "description": "Create main.py as the runnable entry point. Import from src/logic.py. Add argument parsing and user-facing output.",
      "agent_type": "coder",
      "dependencies": ["task_002"],
      "priority": 3
    },
    {
      "task_id": "task_004",
      "title": "Write unit tests",
      "description": "Create tests/test_logic.py with pytest tests covering the main functions in src/logic.py. Test normal cases and edge cases.",
      "agent_type": "tester",
      "dependencies": ["task_002"],
      "priority": 4
    },
    {
      "task_id": "task_005",
      "title": "Summarize completed project",
      "description": "Review all produced files and write a summary of what was built, how to run it, and what was accomplished.",
      "agent_type": "synthesizer",
      "dependencies": ["task_003", "task_004"],
      "priority": 5
    }
  ],
  "clarifying_questions": [],
  "technical_requirements": [
    "Use pytest for testing",
    "Keep file paths relative",
    "Write type hints and docstrings"
  ],
  "milestones": [
    "Core data models and types defined",
    "Business logic implemented and tested",
    "Runnable entry point complete",
    "Test suite passing",
    "Project summary delivered"
  ],
  "architecture_notes": "Single-package Python project. src/ holds modules, tests/ holds pytest tests, main.py is the entry point. All modules use type hints and docstrings."
}"""

    return (
        "You are Planner_Agent. Your ONLY task is to return valid JSON.\n"
        "\nRESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT.\n"
        "\n=== PLANNING RULES ===\n"
        "1. DECOMPOSE the goal into 3-7 focused tasks. Each task should produce one or more specific files.\n"
        "2. EVERY build/create/implement goal MUST have multiple coder tasks — one per logical module or file group.\n"
        "   Do NOT put everything in a single coder task.\n"
        "3. Task descriptions must be SPECIFIC: name the exact files to create, what functions/classes to implement,\n"
        "   and what the file should import from other tasks.\n"
        "4. Use DEPENDENCIES to order tasks correctly. Later tasks that import from earlier ones must list them.\n"
        "5. Include a tester task (agent_type='tester') that writes pytest tests for the core logic.\n"
        "6. End with a synthesizer task (agent_type='synthesizer') that depends on all coder/tester tasks.\n"
        "7. Synthesizer tasks MUST depend on coder tasks — never make synthesizer the first or only task.\n"
        "8. clarifying_questions: array of strings. If details are missing, ask concise follow-up questions. "
        "If no clarification is needed, return an empty list.\n"
        "9. technical_requirements: array of strings describing exact libraries, frameworks, file structure, and quality expectations.\n"
        "10. milestones: array of plain strings describing each phase of completion.\n"
        "11. architecture_notes: one paragraph describing the file structure, module layout, and design decisions.\n"
        "\n=== AGENT TYPES ===\n"
        "- 'coder': writes or modifies source files\n"
        "- 'tester': writes pytest test files\n"
        "- 'synthesizer': summarizes completed work (always last, always depends on coders)\n"
        "\n=== EXAMPLE OUTPUT (for a calculator goal) ===\n"
        + schema_example
        + "\n\n=== NOW PLAN THIS GOAL ===\n"
        "Goal: " + goal + "\n"
        + ("Architecture notes: " + architecture_notes + "\n" if architecture_notes else "")
        + "\nThink through the full file structure needed, then return ONLY the JSON object."
    )
