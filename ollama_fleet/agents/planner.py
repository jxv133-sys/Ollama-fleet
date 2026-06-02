from __future__ import annotations


def build_planner_prompt(goal: str, architecture_notes: str) -> str:
    schema_example = """{
  "tasks": [
    {
      "task_id": "task_001",
      "step_number": 1,
      "title": "Configuration loader module",
      "filename": "config.py",
      "file_spec": {
        "purpose": "Load and validate configuration from files and environment variables. This is a standalone utility with no project dependencies.",
        "public_exports": ["load_config(path: str) -> Config", "Config (class)"],
        "imports": [],
        "functions": [
          {"name": "load_config", "params": ["path: str"], "returns": "Config", "docstring": "Load config from file, validate schema"},
          {"name": "Config", "type": "class", "docstring": "Configuration object with fields for all settings"}
        ],
        "exact_content": "Parse YAML/TOML/JSON, return typed Config object. Include validation of required fields.",
        "estimated_lines": 100
      },
      "dependencies": [],
      "priority": 1
    },
    {
      "task_id": "task_002",
      "step_number": 2,
      "title": "Data validation module",
      "filename": "validators.py",
      "file_spec": {
        "purpose": "Validate data schemas and types. Import config to get validation rules.",
        "public_exports": ["validate_dataframe(df: DataFrame, schema: Dict) -> ValidationResult", "ValidationResult (class)"],
        "imports": ["from config import Config"],
        "functions": [
          {"name": "validate_dataframe", "params": ["df: DataFrame", "schema: Dict"], "returns": "ValidationResult", "docstring": "Check all rows match schema, return errors"},
          {"name": "ValidationResult", "type": "class", "docstring": "Contains valid: bool, errors: List[str]"}
        ],
        "exact_content": "Type check each column, validate ranges/formats, collect all errors",
        "estimated_lines": 120
      },
      "dependencies": ["task_001"],
      "priority": 2
    },
    {
      "task_id": "task_003",
      "step_number": 3,
      "title": "Data transformation module",
      "filename": "transformers.py",
      "file_spec": {
        "purpose": "Filter columns and aggregate rows based on rules. Use config for rules.",
        "public_exports": ["filter_columns(df: DataFrame, cols: List[str]) -> DataFrame", "aggregate_rows(df: DataFrame, agg_rules: Dict) -> DataFrame"],
        "imports": ["from config import Config"],
        "functions": [
          {"name": "filter_columns", "params": ["df: DataFrame", "cols: List[str]"], "returns": "DataFrame", "docstring": "Keep only specified columns"},
          {"name": "aggregate_rows", "params": ["df: DataFrame", "agg_rules: Dict"], "returns": "DataFrame", "docstring": "Group and aggregate by rules (sum, mean, count, etc)"}
        ],
        "exact_content": "Implement pandas operations for column selection and groupby aggregations",
        "estimated_lines": 90
      },
      "dependencies": ["task_001"],
      "priority": 3
    },
    {
      "task_id": "task_004",
      "step_number": 4,
      "title": "Output formatters module",
      "filename": "formatters.py",
      "file_spec": {
        "purpose": "Export dataframes to CSV, JSON, and Parquet formats.",
        "public_exports": ["export_csv(df: DataFrame, path: str) -> None", "export_json(df: DataFrame, path: str) -> None", "export_parquet(df: DataFrame, path: str) -> None"],
        "imports": [],
        "functions": [
          {"name": "export_csv", "params": ["df: DataFrame", "path: str"], "returns": "None", "docstring": "Write dataframe to CSV file"},
          {"name": "export_json", "params": ["df: DataFrame", "path: str"], "returns": "None", "docstring": "Write dataframe to JSON file"},
          {"name": "export_parquet", "params": ["df: DataFrame", "path: str"], "returns": "None", "docstring": "Write dataframe to Parquet file"}
        ],
        "exact_content": "Use pandas to_csv/to_json and pyarrow for Parquet. Handle errors with try/except.",
        "estimated_lines": 70
      },
      "dependencies": [],
      "priority": 4
    },
    {
      "task_id": "task_005",
      "step_number": 5,
      "title": "CLI entry point",
      "filename": "main.py",
      "file_spec": {
        "purpose": "Main CLI interface. Parse arguments, coordinate modules, handle logging.",
        "public_exports": ["main() -> None"],
        "imports": ["from config import load_config", "from validators import validate_dataframe", "from transformers import filter_columns, aggregate_rows", "from formatters import export_csv, export_json, export_parquet"],
        "functions": [
          {"name": "main", "params": [], "returns": "None", "docstring": "Entry point: load config, read CSV, validate, transform, export"}
        ],
        "exact_content": "Use argparse for CLI args, setup logging, call validators/transformers/formatters in sequence, handle errors with try/except/finally",
        "estimated_lines": 150
      },
      "dependencies": ["task_001", "task_002", "task_003", "task_004"],
      "priority": 5
    }
  ]
}"""

    return (
        "You are a software project planning agent.\n"
        "\nYour job is to break a software project into implementation tasks.\n"
        "\nCRITICAL RULES FOR DECOMPOSITION:\n"
        "1. Create one task per file. Each file is ONE module with ONE responsibility.\n"
        "2. For multi-feature goals: Create AT LEAST 3-5 separate tasks. Do NOT combine unrelated features into one file.\n"
        "3. If task A imports from file B, then task B MUST exist as a separate task.\n"
        "4. Base modules (config, utils, schemas) MUST come first with no dependencies.\n"
        "5. Dependent modules (business logic) depend on base modules.\n"
        "6. CLI/main entry point comes LAST and depends on all others.\n"
        "\nAll files go in the SAME directory. Do NOT use subdirectories or file paths.\n"
        "\nFor each file, specify EXACTLY:\n"
        "- What functions/classes it exports (with full signatures)\n"
        "- What it imports from other project files (MUST be real files being created)\n"
        "- Exact docstrings and behavior needed\n"
        "- Realistic estimated line count (10-20 for tiny, 50-100 for small, 100-200 for medium)\n"
        "\nDo NOT write code.\n"
        "\nDo NOT explain your reasoning.\n"
        "\nReturn ONLY valid JSON.\n"
        "\nResponse format:\n"
        "{\n"
        "  \"tasks\": [\n"
        "    {\n"
        "      \"task_id\": \"task_NNN\",\n"
        "      \"step_number\": 1,\n"
        "      \"title\": \"Descriptive title\",\n"
        "      \"filename\": \"just_the_filename.py\",\n"
        "      \"file_spec\": {\n"
        "        \"purpose\": \"What this file does and its role in the system\",\n"
        "        \"public_exports\": [\"function_name(param: type) -> return_type\", \"ClassName\"],\n"
        "        \"imports\": [\"from config import Config\"],\n"
        "        \"functions\": [\n"
        "          {\"name\": \"func\", \"type\": \"function\", \"params\": [\"x: type\"], \"returns\": \"type\", \"docstring\": \"what it does\"},\n"
        "          {\"name\": \"MyClass\", \"type\": \"class\", \"docstring\": \"class purpose and key methods\"}\n"
        "        ],\n"
        "        \"exact_content\": \"Precise implementation requirements. Be specific about algorithms and error handling.\",\n"
        "        \"estimated_lines\": 100\n"
        "      },\n"
        "      \"dependencies\": [\"task_001\"],\n"
        "      \"priority\": 1\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\nRULES:\n"
        "- Create one task per file.\n"
        "- Estimate lines realistically: base modules 50-150, business logic 80-150, CLI 100-200.\n"
        "- Do NOT create single file handling 4+ distinct features.\n"
        "- List COMPLETE function signatures with all parameters and return types.\n"
        "- Use exact_content to specify algorithms, error handling, edge cases.\n"
        "- Do NOT invent unnecessary files.\n"
        "- Return only JSON.\n"
        "\nGoal: " + goal + "\n"
        + ("Architecture notes: " + architecture_notes + "\n" if architecture_notes else "")
    )
