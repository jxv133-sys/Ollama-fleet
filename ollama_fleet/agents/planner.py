from __future__ import annotations


def build_planner_prompt(goal: str, architecture_notes: str) -> str:
    example = """\
1. models.py — Data Models
   PURPOSE: Define application data structures
   DEPENDS ON: (none)

2. logic.py — Business Logic
   PURPOSE: Core application functionality
   DEPENDS ON: models.py

3. main.py — CLI Entry Point
   PURPOSE: User interface and argument parsing
   DEPENDS ON: logic.py"""

    prompt = (
        "You are a software project planning agent.\n"
        "\n"
        "Your job is to break a software project into a numbered list of files.\n"
        "Each file has ONE responsibility.\n"
        "\n"
        "RULES:\n"
        "1. One responsibility per file. Do NOT combine multiple features.\n"
        "2. For multi-feature goals: create AT LEAST 3-5 separate files.\n"
        "3. Base modules (models, config, utils) come first with no dependencies.\n"
        "4. If file A depends on file B, B must be listed first.\n"
        "5. The CLI/main entry point comes LAST.\n"
        "6. All files are in the SAME flat directory — no subdirectories.\n"
        "\n"
        "For EACH file write EXACTLY this format (nothing else):\n"
        "\n"
        "N. filename.py — Short title\n"
        "   PURPOSE: What this file does\n"
        "   DEPENDS ON: other files, or (none)\n"
        "\n"
        "Do NOT design APIs, functions, or algorithms.\n"
        "Do NOT write code.\n"
        "Do NOT add any text before the first file or after the last file.\n"
        "Do NOT use JSON, markdown, or any other format.\n"
        "ONLY output the numbered list.\n"
        "\n"
        "EXAMPLE OUTPUT:\n"
        "\n"
        + example
        + "\n\n"
        "Now plan this goal:\n"
        "\n"
        "Goal: " + goal + "\n"
        + ("Architecture notes: " + architecture_notes + "\n" if architecture_notes else "")
    )
    return prompt
