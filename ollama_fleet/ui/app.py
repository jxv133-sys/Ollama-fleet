from __future__ import annotations

import asyncio
import logging
from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Static, Header, Footer
    from textual.containers import Vertical
except Exception:  # pragma: no cover - optional dependency
    App = object
    ComposeResult = object
    Static = object
    Header = object
    Footer = object
    Vertical = object

from .event_bus import UIEventBus

logger = logging.getLogger(__name__)


class FleetUI(App):
    CSS = """
    Screen {
        align: center middle;
    }
    """

    def __init__(self, ui_bus: UIEventBus):
        super().__init__()
        self.ui_bus = ui_bus
        self.jobs: dict[str, dict[str, Any]] = {}
        self.escalations: list[dict[str, Any]] = []
        self.ui_bus.subscribe(self._handle_event)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(Static("Jobs:\n", id="jobs"), Static("Escalations:\n", id="escalations"))
        yield Footer()

    async def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh_view)

    def _handle_event(self, event: dict[str, Any]) -> None:
        t = event.get("type")
        if t == "job_state_changed":
            jid = event.get("job_id")
            self.jobs[jid] = {**self.jobs.get(jid, {}), "state": event.get("new_state")}
        elif t == "task_state_changed":
            tid = event.get("task_id")
            job = event.get("job_id")
            if job:
                self.jobs.setdefault(job, {})["last_task"] = tid
        elif t == "escalation_added":
            self.escalations.append(event.get("escalation"))

    def refresh_view(self) -> None:
        try:
            jobs_w = self.query_one("#jobs", Static)
            esc_w = self.query_one("#escalations", Static)
            jobs_lines = [f"Jobs ({len(self.jobs)}):"]
            for jid, meta in sorted(self.jobs.items()):
                jobs_lines.append(f"{jid}: {meta.get('state')} last_task={meta.get('last_task')}")
            jobs_w.update("\n".join(jobs_lines))

            esc_lines = [f"Escalations ({len(self.escalations)}):"]
            for e in self.escalations[-10:]:
                esc_lines.append(f"{e.get('timestamp')} {e.get('task_id')} {e.get('reason')}")
            esc_w.update("\n".join(esc_lines))
        except Exception:
            logger.exception("Failed to refresh UI view")

    def run_blocking(self) -> None:
        try:
            super().run()
        except Exception:
            logger.exception("Textual UI failed to run")
