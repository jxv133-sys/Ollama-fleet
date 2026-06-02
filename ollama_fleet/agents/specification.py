"""File Specification Agent

The Specification Agent creates detailed file requirements that the Coder Agent uses
to implement a complete, correct source file. This separates the architectural and
specification concerns from the implementation concerns, allowing the Coder to focus
solely on generating high-quality code that satisfies the specification.
"""

from __future__ import annotations

from typing import Any


def build_specification_prompt(
    task_description: str,
    file_path: str | None = None,
    required_contents: list[str] | None = None,
    estimated_size: str | None = None,
    goal: str | None = None,
    active_files: list[str] | None = None,
) -> str:
    """Build a detailed file specification.
    
    This prompt asks the agent to create structured requirements that include:
    - The file's purpose and responsibilities
    - Required imports and dependencies
    - Required functions/classes and their signatures
    - Required behavior and contracts
    - Forbidden patterns or anti-patterns
    """
    lines = [
        "You are a File Specification Agent.",
        "",
        "Your job is to create a detailed specification for a single source file.",
        "The specification will be used by a Coder to implement the complete file.",
        "",
        "IMPORTANT:",
        "- Do NOT write code.",
        "- Do NOT explain your reasoning.",
        "- Return ONLY valid JSON.",
        "",
        "Response format:",
        "{",
        '  "file_path": "path/to/file.py",',
        '  "purpose": "Brief description of what this file does",',
        '  "imports": ["import sys", "from pathlib import Path"],',
        '  "required_functions": ["def function_name(args): \'\'\'signature and docstring\'\'\'"],',
        '  "required_behavior": ["The function must handle X case", "Must return Y type"],',
        '  "forbidden_behavior": ["Must not use global variables", "Must not modify input"],',
        '  "required_contents": ["ClassName", "function_name", "CONSTANT_VALUE"]',
        "}",
        "",
        "Rules:",
        "- file_path: exact path where this file will be created",
        "- purpose: one sentence describing the file's single responsibility",
        "- imports: list of import statements this file needs",
        "- required_functions: function signatures with docstrings (not implementations)",
        "- required_behavior: specific behaviors or contracts this file must implement",
        "- forbidden_behavior: patterns or practices to avoid",
        "- required_contents: symbol names that must be exported",
        "",
        "Task Context:",
    ]
    
    if goal:
        lines.append(f"Goal: {goal}")
    
    if file_path:
        lines.append(f"Target file: {file_path}")
    
    if task_description:
        lines.append(f"Task description: {task_description}")
    
    if active_files:
        lines.append("Existing files in the project:")
        for f in active_files[:10]:  # Limit to 10 to avoid overwhelming the prompt
            lines.append(f"  - {f}")
        if len(active_files) > 10:
            lines.append(f"  ... and {len(active_files) - 10} more files")
    
    if required_contents:
        lines.append("Required symbols/contents:")
        for content in required_contents:
            lines.append(f"  - {content}")
    
    if estimated_size:
        lines.append(f"Estimated file size: {estimated_size}")
    
    lines.extend([
        "",
        "Generate a complete specification for the file that satisfies the task.",
        "Return ONLY valid JSON with no explanation or markdown.",
    ])
    
    return "\n".join(lines)
