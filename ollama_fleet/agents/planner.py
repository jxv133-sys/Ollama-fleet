from __future__ import annotations


def build_planner_prompt(goal: str, architecture_notes: str) -> str:
    schema_example = """{
  "tasks": [
    {
      "task_id": "task_001",
      "step_number": 1,
      "title": "Implement core data models",
      "description": "Create src/models.py with all data classes and type definitions needed by the application. Include docstrings and type hints.",
      "agent_type": "coder",
      "file_path": "src/module.py",
      "file_spec": {
        "purpose": "Data model definitions and type hints",
        "imports": ["from typing import TypedDict"],
        "required_functions": [],
        "required_behavior": ["Define data classes with type hints", "Include __repr__ for debugging"],
        "forbidden_behavior": ["Create side-effectful global state", "Invent top-level CLI behavior"]
      },
      "required_contents": ["ClassName", "function_name()", "type hints for all functions"],
      "estimated_size": "small",
      "dependencies": [],
      "priority": 1
    },
    {
      "task_id": "task_002",
      "step_number": 2,
      "title": "Implement business logic",
      "description": "Create src/logic.py with the main application logic. Import from models.py. Handle edge cases and errors.",
      "agent_type": "coder",
      "file_path": "src/logic.py",
      "required_contents": ["process_request()", "handle_errors()"],
      "estimated_size": "medium",
      "dependencies": ["task_001"],
      "priority": 2
    },
    {
      "task_id": "task_003",
      "step_number": 3,
      "title": "Implement CLI entry point",
      "description": "Create main.py as the runnable entry point. Import from src/logic.py. Add argument parsing and user-facing output.",
      "agent_type": "coder",
      "file_path": "main.py",
      "required_contents": ["if __name__ == '__main__'", "argparse usage"],
      "estimated_size": "small",
      "dependencies": ["task_002"],
      "priority": 3
    },
    {
      "task_id": "task_004",
      "step_number": 4,
      "title": "Add CLI helper module",
      "description": "Create src/cli.py with command parsing helpers and output formatting utilities. main.py should import these helpers.",
      "agent_type": "coder",
      "file_path": "src/cli.py",
      "required_contents": ["parse_args()", "format_output()"],
      "estimated_size": "small",
      "dependencies": ["task_002"],
      "priority": 4
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
        "You are a software project planning agent.\n"
        "\nYour job is to break a software project into implementation tasks.\n"
        "\nFocus on:\n"
        "- What files need to be created\n"
        "- What each file is responsible for\n"
        "- Dependencies between files\n"
        "- Required functions and classes\n"
        "\nDo NOT write code.\n"
        "\nDo NOT explain your reasoning.\n"
        "\nReturn ONLY valid JSON.\n"
        "\nResponse format:\n"
        "{\n"
        "  \"project_summary\": \"short summary\",\n"
        "  \"tasks\": [\n"
        "    {\n"
        "      \"title\": \"task name\",\n"
        "      \"file_path\": \"path/to/file.py\",\n"
        "      \"purpose\": \"what this file does\",\n"
        "      \"depends_on\": [],\n"
        "      \"requirements\": [\n"
        "        \"requirement 1\",\n"
        "        \"requirement 2\"\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\nRules:\n"
        "- Create one task per file.\n"
        "- Use exact file paths when possible.\n"
        "- Keep requirements specific.\n"
        "- Do not invent unnecessary files.\n"
        "- Return only JSON.\n"
        "\nGoal: " + goal + "\n"
        + ("Architecture notes: " + architecture_notes + "\n" if architecture_notes else "")
    )
