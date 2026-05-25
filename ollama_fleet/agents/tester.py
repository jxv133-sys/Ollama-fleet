from __future__ import annotations


def build_tester_prompt(workspace_state: str, test_results: str) -> str:
    schema_example = """{
  "tests_passed": 8,
  "tests_failed": 2,
  "failures": [
    {
      "test_name": "test_function_xyz",
      "error_message": "AssertionError: expected X but got Y",
      "suggested_fix": "Fix the logic to handle edge case"
    }
  ],
  "ready_for_review": false
}"""
    return (
        "You are Tester_Agent. Your ONLY task is to return valid JSON.\n"
        "\nRESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT.\n"
        "\nJSON SCHEMA REQUIREMENTS:\n"
        "- tests_passed: integer (number of passing tests)\n"
        "- tests_failed: integer (number of failing tests)\n"
        "- failures: array of objects, EACH with: test_name (string), error_message (string), suggested_fix (string)\n"
        "- ready_for_review: boolean (true if all tests pass)\n"
        "\nEXAMPLE OUTPUT:\n"
        + schema_example
        + "\n\nWorkspace state:\n"
        + workspace_state
        + "\nExisting test results:\n"
        + test_results
        + "\n\nRETURN ONLY VALID JSON MATCHING THE SCHEMA ABOVE."
    )
