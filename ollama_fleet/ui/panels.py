from __future__ import annotations

from typing import Any


class ValidationPanel:
    def render(self, validation_result: dict[str, Any]) -> str:
        lines = ["Validation Results:"]
        if not validation_result:
            return "Validation Results: no data"

        lines.append(f"Syntax OK: {validation_result.get('syntax_ok')}")
        lines.append(f"Linter available: {validation_result.get('linter_available')}")
        for issue in validation_result.get('lint_results', []):
            lines.append(
                f"- {issue.get('file_path')}:{issue.get('line_number')} {issue.get('code')} {issue.get('message')}"
            )
        return "\n".join(lines)


class EscalationPanel:
    def render(self, escalations: list[dict[str, Any]]) -> str:
        if not escalations:
            return "Escalations: none"

        lines = ["Escalation Records:"]
        for escalation in escalations[-5:]:
            lines.append(
                f"{escalation.get('timestamp')} {escalation.get('task_id')} {escalation.get('reason')}"
            )
        return "\n".join(lines)
