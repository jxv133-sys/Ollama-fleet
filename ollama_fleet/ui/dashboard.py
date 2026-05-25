from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Static, TextLog, Tree

from ollama_fleet.ui.event_bus import UIEventBus


class JobInfoPanel(Static):
    """Display current job info (goal, workspace, elapsed time)."""

    def render(self) -> str:
        job_id = getattr(self, "_job_id", "")
        goal = getattr(self, "_goal", "")
        workspace = getattr(self, "_workspace", "")
        start_time = getattr(self, "_start_time", 0.0)
        
        lines = ["[bold]JOB INFO[/bold]"]
        lines.append(f"ID:      {job_id[:16]}" if job_id else "ID:      -")
        lines.append(f"Goal:    {goal[:50]}" if goal else "Goal:    -")
        lines.append(f"Path:    {Path(workspace).name}" if workspace else "Path:    -")
        if start_time:
            elapsed = datetime.now(timezone.utc).timestamp() - start_time
            lines.append(f"Elapsed: {elapsed:.1f}s")
        return "\n".join(lines)


class AgentStatusPanel(Static):
    """Display agent execution status and timing."""

    def render(self) -> str:
        agents = getattr(self, "_agents", {})
        lines = ["[bold]AGENT STATUS[/bold]"]
        for agent_type in ["planner", "coder", "critic", "tester", "synthesizer"]:
            info = agents.get(agent_type, {})
            state = info.get("state", "idle")
            elapsed = info.get("elapsed", 0)
            icon = "▶" if state == "running" else "✓" if state == "completed" else "✕" if state == "failed" else "-"
            time_str = f"({elapsed:.1f}s)" if elapsed else ""
            color = "green" if state == "completed" else "red" if state == "failed" else "blue" if state == "running" else "white"
            lines.append(f"[{color}]{icon}[/{color}] {agent_type:12} {state:10} {time_str}")
        return "\n".join(lines)


class ProgressPanel(Static):
    """Display task progress and timeline."""

    def render(self) -> str:
        tasks = getattr(self, "_tasks", {})
        lines = ["[bold]PROGRESS[/bold]"]
        for task_id, info in sorted(tasks.items())[:15]:
            state = info.get("state", "pending")
            agent = info.get("agent_type", "?")
            icon = {"pending": "○", "running": "◐", "completed": "●", "failed": "✕"}.get(state, "?")
            elapsed = info.get("elapsed", 0)
            time_str = f"({elapsed:.1f}s)" if elapsed else ""
            color = "green" if state == "completed" else "red" if state == "failed" else "blue" if state == "running" else "white"
            lines.append(f"[{color}]{icon}[/{color}] {agent:12} {state:10} {time_str}")
        return "\n".join(lines)


class FileTreePanel(Static):
    """Display generated project files."""

    def render(self) -> str:
        files = getattr(self, "_files", [])
        lines = ["[bold]FILES[/bold]"]
        if not files:
            lines.append("(no files yet)")
        else:
            for f in files[:20]:
                lines.append(f"  • {f}")
        return "\n".join(lines)


class OllamaFleetDashboard(App):
    """Comprehensive multi-panel dashboard for Ollama Fleet orchestration."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 3;
        grid-columns: 1fr 1fr 2fr;
        grid-rows: auto 1fr 1fr;
    }
    Header { dock: top; height: 1; }
    Footer { dock: bottom; height: 1; }
    #job-info { row-span: 1; column-span: 1; border: solid $primary; }
    #agent-status { row-span: 2; column-span: 1; border: solid $accent; }
    #file-tree { row-span: 2; column-span: 1; border: solid $success; }
    #progress { row-span: 2; column-span: 1; border: solid $warning; }
    #raw-output { row-span: 3; column-span: 1; border: solid $info; }
    TextLog { height: 1fr; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, ui_bus: UIEventBus):
        super().__init__()
        self.ui_bus = ui_bus
        self.title = "Ollama Fleet Dashboard"
        self.job_info = JobInfoPanel(id="job-info")
        self.agent_status = AgentStatusPanel(id="agent-status")
        self.progress = ProgressPanel(id="progress")
        self.file_tree = FileTreePanel(id="file-tree")
        self.raw_output = TextLog(id="raw-output")

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.job_info
        yield self.agent_status
        yield self.file_tree
        yield self.progress
        yield self.raw_output
        yield Footer()

    def on_mount(self) -> None:
        """Set up event subscription and refresh."""
        self.ui_bus.subscribe(self._handle_event)
        self.set_interval(0.5, self._refresh_panels)

    def _handle_event(self, event: dict[str, Any]) -> None:
        """Process events from UIEventBus."""
        t = event.get("type")

        if t == "job_state_changed":
            jid = event.get("job_id")
            self.job_info._job_id = jid
            if not hasattr(self.job_info, "_start_time") or not self.job_info._start_time:
                self.job_info._start_time = datetime.now(timezone.utc).timestamp()

        elif t == "agent_log":
            msg = event.get("message", "")
            self.raw_output.write_line(msg)

        elif t == "agent_output":
            agent = event.get("agent_type", "?")
            output = event.get("output", "")
            if isinstance(output, dict):
                output = json.dumps(output, indent=2)[:500]
            self.raw_output.write_line(f"[yellow]{agent}[/yellow]: {output}")

        elif t == "task_state_changed":
            tid = event.get("task_id")
            agent = event.get("agent_type", "?")
            state = event.get("new_state", "pending")
            if not hasattr(self.progress, "_tasks"):
                self.progress._tasks = {}
            self.progress._tasks[tid] = {
                "state": state,
                "agent_type": agent,
                "start_time": datetime.now(timezone.utc).timestamp(),
                "elapsed": 0,
            }

        elif t == "workspace_created":
            ws = event.get("workspace_path")
            if ws:
                self.job_info._workspace = str(ws)
                self._update_file_tree(str(ws))

        elif t == "file_written":
            path = event.get("path")
            if path:
                if not hasattr(self.file_tree, "_files"):
                    self.file_tree._files = []
                if path not in self.file_tree._files:
                    self.file_tree._files.append(path)
                self.raw_output.write_line(f"[blue]File:[/blue] {path}")

        elif t == "validation_result":
            result = event.get("validation_result", {})
            status = "[green]✓[/green]" if result.get("syntax_ok") else "[red]✕[/red]"
            self.raw_output.write_line(f"Validation: {status}")

        elif t == "escalation_added":
            esc = event.get("escalation", {})
            self.raw_output.write_line(f"[red]⚠ ESCALATION:[/red] {esc.get('reason')}")

    def _update_file_tree(self, workspace_path: str) -> None:
        """Scan and update file tree from workspace."""
        if not hasattr(self.file_tree, "_files"):
            self.file_tree._files = []
        try:
            src = Path(workspace_path) / "src"
            if src.exists():
                for py_file in sorted(src.glob("*.py")):
                    fname = f"src/{py_file.name}"
                    if fname not in self.file_tree._files:
                        self.file_tree._files.append(fname)
        except Exception:
            pass

    def _refresh_panels(self) -> None:
        """Update all panel displays."""
        try:
            if hasattr(self.progress, "_tasks"):
                now = datetime.now(timezone.utc).timestamp()
                for tid, info in self.progress._tasks.items():
                    if info.get("state") in ("running", "completed"):
                        start = info.get("start_time", now)
                        info["elapsed"] = now - start
            
            if hasattr(self.agent_status, "_agents"):
                now = datetime.now(timezone.utc).timestamp()
                for agent, info in self.agent_status._agents.items():
                    if info.get("state") in ("running", "completed"):
                        start = info.get("start_time", now)
                        info["elapsed"] = now - start

            self.job_info.update()
            self.agent_status.update()
            self.progress.update()
            self.file_tree.update()
        except Exception:
            pass

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def run_blocking(self) -> None:
        """Run the app, catching exceptions gracefully."""
        try:
            super().run()
        except Exception:
            pass
