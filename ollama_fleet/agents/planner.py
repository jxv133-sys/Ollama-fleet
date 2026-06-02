from __future__ import annotations


def build_planner_prompt(goal: str, architecture_notes: str) -> str:
    example = """\
1. config.py — Configuration loader
   PURPOSE: Load and validate settings from a config file. No project dependencies.
   EXPORTS: load_config(path: str) -> Config, Config (class)
   IMPORTS: (none)
   FUNCTIONS:
     - load_config(path: str) -> Config: Parse YAML/TOML/JSON, validate required fields
     - Config (class): Typed config object with all settings as attributes
   BEHAVIOR: Parse file format by extension, raise ConfigError on missing required fields
   LINES: 80
   DEPENDS ON: (none)
   PRIORITY: 1

2. validators.py — Data validation module
   PURPOSE: Validate CSV schema and column types using Config rules.
   EXPORTS: validate_dataframe(df: DataFrame, schema: Dict) -> ValidationResult, ValidationResult (class)
   IMPORTS: from config import Config
   FUNCTIONS:
     - validate_dataframe(df: DataFrame, schema: Dict) -> ValidationResult: Check all rows match schema
     - ValidationResult (class): Contains valid: bool, errors: List[str]
   BEHAVIOR: Type-check each column, validate ranges/formats, collect all errors before returning
   LINES: 120
   DEPENDS ON: 1
   PRIORITY: 2

3. main.py — CLI entry point
   PURPOSE: Parse CLI args, coordinate validation and output, handle logging.
   EXPORTS: main() -> None
   IMPORTS: from config import load_config, from validators import validate_dataframe
   FUNCTIONS:
     - main() -> None: Load config, read CSV, validate, export
   BEHAVIOR: argparse for CLI, setup logging, call modules in sequence, try/except/finally for errors
   LINES: 150
   DEPENDS ON: 1, 2
   PRIORITY: 3"""

    prompt = (
        "You are a software project planning agent.\n"
        "\n"
        "Your job is to break a software project into a numbered list of implementation tasks.\n"
        "Each task is ONE file with ONE responsibility.\n"
        "\n"
        "RULES:\n"
        "1. One task per file. Do NOT combine multiple features into one file.\n"
        "2. For multi-feature goals: create AT LEAST 3-5 separate tasks.\n"
        "3. Base modules (config, utils, schemas) come first with no dependencies.\n"
        "4. If file A imports from file B, B must be its own task and listed first.\n"
        "5. The CLI/main entry point comes LAST and depends on everything else.\n"
        "6. All files are in the SAME flat directory — no subdirectories.\n"
        "\n"
        "For EACH task write EXACTLY this format (nothing else):\n"
        "\n"
        "N. filename.py — Short title\n"
        "   PURPOSE: What this file does and its role in the system\n"
        "   EXPORTS: function_name(param: type) -> return_type, ClassName\n"
        "   IMPORTS: from x import Y (or \"(none)\" if no project imports)\n"
        "   FUNCTIONS:\n"
        "     - func_name(params) -> return_type: what it does\n"
        "     - ClassName (class): purpose and key methods\n"
        "   BEHAVIOR: Precise algorithms, error handling, edge cases\n"
        f"   LINES: estimated line count (50-150 for modules, 100-200 for CLI)\n"
        "   DEPENDS ON: comma-separated task numbers, or \"(none)\"\n"
        "   PRIORITY: N\n"
        "\n"
        "Do NOT write code.\n"
        "Do NOT add any text before the first task or after the last task.\n"
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
