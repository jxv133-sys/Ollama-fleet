from __future__ import annotations

from typing import Any


def build_critic_prompt(
    modified_files: list[str],
    file_contents: dict[str, str],
    lint_results: list[dict[str, str]],
    critic_issues: list[dict[str, Any]] | None = None,
) -> str:
    schema_example = """{
  "approved": true,
  "issues": [],
  "overall_assessment": "Code is correct and complete. No issues found."
}"""
    schema_example_reject = """{
  "approved": false,
  "issues": [
    {
      "file_path": "src/calculator.py",
      "line_number": 12,
      "severity": "major",
      "description": "Division by zero not handled: divide() does not check if divisor is 0.",
      "suggested_fix": "Add: if b == 0: raise ValueError('Cannot divide by zero')"
    }
  ],
  "overall_assessment": "The divide function will crash on zero input. Fix the guard clause."
}"""
    schema_example_empty = """{
  "approved": false,
  "issues": [
    {
      "file_path": "src/calculator.py",
      "line_number": 0,
      "severity": "critical",
      "description": "No source code generated",
      "suggested_fix": "Generate actual Python code with implementations"
    }
  ],
  "overall_assessment": "No valid source code was generated."
}"""

    lines: list[str] = [
        "You are Critic_Agent. Your ONLY task is to return valid JSON.",
        "",
        "RESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT.",
        "",
        "CRITICAL RULES:",
        "0. CHECK FIRST: If the code is empty, whitespace-only, or not actual source code,",
        "   immediately reject with approved=false and an issue describing 'No source code generated'.",
        "   Do NOT try to analyze or guess what the code should do.",
        "1. If the code is syntactically valid Python and implements the described task reasonably,",
        "   you MUST set approved=true and return an empty issues array.",
        "2. Only set approved=false if you find a SPECIFIC, CONCRETE bug — not style preferences.",
        "3. Every issue MUST include the real file_path, the actual line_number (not 0),",
        "   a precise description of the bug, and a concrete suggested_fix with example code.",
        "4. Do NOT use placeholder text like 'Overall assessment of the code changes.'",
        "   Write a real one-sentence summary of what you actually found.",
        "5. Do NOT reject code just because it is simple or short.",
        "6. If the previous revision already addressed the prior issues, set approved=true.",
        "",
        "JSON SCHEMA:",
        "- approved: boolean",
        "- issues: array (empty [] when approved=true)",
        "  Each issue: file_path (string), line_number (integer >= 0; 0 for file-level issues),",
        "               severity ('critical'|'major'|'minor'),",
        "               description (specific issue), suggested_fix (concrete fix)",
        "- overall_assessment: one real sentence describing your finding",
        "",
        "EXAMPLE — approved:",
        schema_example,
        "",
        "EXAMPLE — rejected with a specific bug:",
        schema_example_reject,
        "",
        "EXAMPLE — rejected for no code generated:",
        schema_example_empty,
        "",
        "Now review the following code:",
        "",
        "Modified files:",
    ]

    for path in modified_files:
        lines.append(f"  {path}")

    lines.append("")
    lines.append("File contents:")
    for path in modified_files:
        content = file_contents.get(path, "(empty)")
        lines.append(f"\n=== {path} ===")
        lines.append(content)

    lines.append("")
    lines.append("Lint results:")
    if not lint_results:
        lines.append("  None — no lint issues detected.")
    else:
        for issue in lint_results:
            lines.append(
                f"  {issue.get('file_path', '')}:{issue.get('line_number', '')} "
                f"[{issue.get('code', '')}] {issue.get('message', '')}"
            )

    if critic_issues:
        lines.append("")
        lines.append("Issues raised in the previous revision (check if they are now fixed):")
        for issue in critic_issues:
            lines.append(
                f"  {issue.get('file_path', '')}:{issue.get('line_number', '')} "
                f"[{issue.get('severity', '')}] {issue.get('description', '')} "
                f"→ Fix was: {issue.get('suggested_fix', '')}"
            )
        lines.append("If these issues are resolved, set approved=true.")

    lines.append("")
    lines.append("RETURN ONLY VALID JSON. Check first: is there actual code? If not, reject immediately.")

    return "\n".join(lines)
