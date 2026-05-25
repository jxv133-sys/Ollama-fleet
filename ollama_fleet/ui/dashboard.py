from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from ollama_fleet.ui.event_bus import UIEventBus
from ollama_fleet.ui.panels import EscalationPanel, ValidationPanel


class FleetDashboard(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #header, #footer {
        dock: top;
    }
    #layout {
        height: 1fr;
        width: 1fr;
    }
    Static {
        border: round white;
        padding: 1;
        min-width: 40;
        min-height: 10;
    }
    """

    def __init__(self, event_bus: UIEventBus, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.event_bus = event_bus
        self.task_queue: list[str] = []
        self.logs: list[str] = []
        self.job_status = "idle"
        self.validation_panel = ValidationPanel()
        self.escalation_panel = EscalationPanel()
        self.latest_validation: dict[str, Any] = {}
        self.escalations: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Header(id="header")
        with Horizontal(id="layout"):
            with Vertical():
                yield Static("Job Status: idle", id="status")
                yield Static("Task Queue: no active tasks", id="tasks")
            with Vertical():
                yield Static("Logs will appear here", id="logs")
                yield Static("Validation results will appear here", id="validation")
                yield Static("Escalations will appear here", id="escalations")
        yield Footer(id="footer")

    def on_mount(self) -> None:
        self.event_bus.subscribe(self.handle_event)
        self.refresh_layout()

    def handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "task_state_changed":
            self.task_queue.append(
                f"{event.get('task_id')} -> {event.get('new_state')} ({event.get('agent_type')})"
            )
        elif event_type == "job_state_changed":
            self.job_status = f"Job {event.get('job_id')} is {event.get('new_state')}"
        elif event_type == "validation_result":
            self.latest_validation = event.get("validation_result", {})
        elif event_type == "escalation_added":
            self.escalations.append(event.get("escalation", {}))
        elif event_type == "agent_log":
            self.logs.append(event.get("message", ""))
        self.refresh_layout()

    def refresh_layout(self) -> None:
        status_widget = self.query_one("#status", Static)
        tasks_widget = self.query_one("#tasks", Static)
        logs_widget = self.query_one("#logs", Static)
        validation_widget = self.query_one("#validation", Static)
        escalations_widget = self.query_one("#escalations", Static)

        status_widget.update(self.job_status)
        tasks_widget.update("\n".join(self.task_queue[-10:]) or "Task Queue: no active tasks")
        logs_widget.update("\n".join(self.logs[-10:]) or "Logs will appear here")
        validation_widget.update(self.validation_panel.render(self.latest_validation))
        escalations_widget.update(self.escalation_panel.render(self.escalations))

    def action_quit(self) -> None:
        self.exit()

    def on_key(self, event: Any) -> None:
        if hasattr(event, "key") and event.key == "q":
            self.exit()
