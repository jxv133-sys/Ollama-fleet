from __future__ import annotations


def build_tester_prompt(workspace_state: str, test_results: str) -> str:
    return (
        "You are Tester_Agent. Produce a JSON object matching the TesterOutput schema.\n"
        "Review the workspace state and any existing test results.\n"
        "Workspace state: "
        + workspace_state
        + "\nTest results: "
        + test_results
        + "\nRespond with only valid JSON."
    )
