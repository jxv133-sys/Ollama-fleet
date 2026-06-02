"""Output Validator Agent

The Output Validator checks if the Coder Agent's output is valid source code
and contains the required elements. If the output is invalid, it signals for
regeneration before sending to the Critic.
"""

from __future__ import annotations

import ast
import re
from typing import Any


class ValidationError(Exception):
    """Raised when code validation fails."""
    pass


def validate_coder_output(
    content: str,
    file_path: str,
    required_functions: list[str] | None = None,
    required_contents: list[str] | None = None,
) -> dict[str, Any]:
    """Validate that coder output is valid Python code with required elements.
    
    Args:
        content: The generated code content
        file_path: Path to the file being generated
        required_functions: Function/class names that must appear in the code
        required_contents: Symbol names that must be exported
        
    Returns:
        A dict with validation results
        
    Raises:
        ValidationError: If the code is empty, not valid Python, or missing required elements
    """
    # Check if content is empty or just whitespace
    if not content or not content.strip():
        raise ValidationError("Generated code is empty or contains only whitespace")
    
    # Check if content is just a file path
    content_stripped = content.strip()
    if content_stripped == file_path or content_stripped.startswith(file_path) and len(content_stripped) < len(file_path) + 50:
        raise ValidationError(f"Generated code appears to be just a file path, not actual source code: {content_stripped[:100]}")
    
    # Try to parse as Python to validate syntax
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        raise ValidationError(f"Generated code has syntax errors: {e}")
    
    # Extract all defined names (functions, classes, variables)
    defined_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)
                elif isinstance(target, ast.Attribute):
                    defined_names.add(target.attr)
    
    # Check required functions
    if required_functions:
        missing_functions = []
        for req_func in required_functions:
            # Extract function name from signature (e.g., "def foo(x, y):" -> "foo")
            func_match = re.search(r'(?:def|async\s+def)\s+(\w+)', req_func)
            if func_match:
                func_name = func_match.group(1)
                if func_name not in defined_names:
                    missing_functions.append(func_name)
        
        if missing_functions:
            raise ValidationError(f"Generated code is missing required functions: {missing_functions}")
    
    # Check required contents (exported symbols)
    if required_contents:
        missing_contents = [name for name in required_contents if name not in defined_names]
        if missing_contents:
            raise ValidationError(f"Generated code is missing required symbols: {missing_contents}")
    
    # If all checks pass, return validation result
    return {
        "valid": True,
        "line_count": len(content.splitlines()),
        "defined_names": sorted(defined_names),
        "file_path": file_path,
    }


def validate_specification_output(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate that specification output is well-formed.
    
    Args:
        spec: The specification dict
        
    Returns:
        A dict with validation results
        
    Raises:
        ValidationError: If the specification is malformed
    """
    required_fields = ["file_path", "purpose", "imports", "required_functions", "required_behavior"]
    
    missing_fields = [field for field in required_fields if field not in spec or not spec[field]]
    if missing_fields:
        raise ValidationError(f"Specification is missing required fields: {missing_fields}")
    
    # Validate that imports are strings
    if not isinstance(spec.get("imports", []), list):
        raise ValidationError("'imports' must be a list of strings")
    
    if not all(isinstance(imp, str) for imp in spec.get("imports", [])):
        raise ValidationError("All items in 'imports' must be strings")
    
    # Validate that required_functions are strings
    if not isinstance(spec.get("required_functions", []), list):
        raise ValidationError("'required_functions' must be a list of strings")
    
    if not all(isinstance(fn, str) for fn in spec.get("required_functions", [])):
        raise ValidationError("All items in 'required_functions' must be strings")
    
    return {
        "valid": True,
        "file_path": spec.get("file_path"),
        "purpose": spec.get("purpose"),
        "required_count": len(spec.get("required_functions", [])),
    }
