from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, Log, Static, Input, Button
from textual.reactive import reactive

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
        lines.append(f"State:   {getattr(self, '_state', 'unknown')}")
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
        layout: vertical;
    }
    Header { 
        dock: top; 
        height: 1; 
    }
    Footer { 
        dock: bottom; 
        height: 1; 
    }
    #input-container { 
        height: auto;
        border: solid white;
        padding: 1;
    }
    #main-grid {
        layout: grid;
        grid-size: 3 3;
        height: 1fr;
    }
    #job-info { 
        border: solid white;
    }
    #agent-status { 
        row-span: 2;
        border: solid cyan;
    }
    #file-tree { 
        row-span: 2;
        border: solid green;
    }
    #progress { 
        row-span: 2;
        border: solid yellow;
    }
    #raw-output { 
        row-span: 3;
        border: solid blue;
    }
    Log { height: 1fr; }
    Input { width: 1fr; }
    Button { margin: 0 1; }
    """

    BINDINGS = [("q", "quit", "Quit"), ("ctrl+c", "quit", "Quit")]

    job_running = reactive(False)

    def __init__(self, ui_bus: UIEventBus, orchestrator: Any | None = None, goal: str = "", config: dict[str, Any] | None = None):
        super().__init__()
        self.ui_bus = ui_bus
        self.orchestrator = orchestrator
        self.goal = goal
        self.config = config or {}
        self.title = "Ollama Fleet Dashboard"
        self.job_info = JobInfoPanel(id="job-info")
        self.agent_status = AgentStatusPanel(id="agent-status")
        self.progress = ProgressPanel(id="progress")
        self.file_tree = FileTreePanel(id="file-tree")
        self.raw_output = Log(id="raw-output")
        self.goal_input = Input(placeholder="Enter project goal...", id="goal-input")
        self.submit_btn = Button("Submit Goal", id="submit-btn")
        self._input_container = Container(id="input-container")

    def compose(self) -> ComposeResult:
        yield Header()
        with self._input_container:
            yield self.goal_input
            yield self.submit_btn
        with Container(id="main-grid"):
            yield self.job_info
            yield self.agent_status
            yield self.file_tree
            yield self.progress
            yield self.raw_output
        yield Footer()

    async def on_mount(self) -> None:
        """Set up event subscription, refresh, and optionally start the orchestrator."""
        self.ui_bus.subscribe(self._handle_event)
        self.set_interval(0.5, self._refresh_panels)
        
        # If a goal was provided via command line, pre-populate it
        if self.goal:
            self.goal_input.value = self.goal
            await asyncio.sleep(0.5)  # Give UI time to render
            # Auto-submit if goal was provided
            await self._submit_goal()
        else:
            self.raw_output.write_line("[cyan]Enter a project goal above and press 'Submit Goal'[/cyan]")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press for goal submission."""
        if event.button.id == "submit-btn":
            await self._submit_goal()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in goal input."""
        if event.input.id == "goal-input":
            await self._submit_goal()

    async def _submit_goal(self) -> None:
        """Submit the goal to the orchestrator."""
        goal_text = self.goal_input.value.strip()
        if not goal_text:
            self.raw_output.write_line("[red]Goal cannot be empty[/red]")
            return
        
        if self.job_running:
            self.raw_output.write_line("[red]Job already running[/red]")
            return

        if self.orchestrator is None:
            self.raw_output.write_line("[red]Orchestrator not available[/red]")
            return

        self.job_running = True
        self.goal_input.disabled = True
        self.submit_btn.disabled = True
        
        self.raw_output.write_line(f"[cyan]Submitting goal: {goal_text}[/cyan]")
        await self._start_job(goal_text)

    async def _start_job(self, goal: str) -> None:
        """Start a job with the given goal."""
        if self.orchestrator is None:
            return
        try:
            job_id = await self.orchestrator.submit_job(goal, self.config)
            self.raw_output.write_line(f"[green]✓ Job completed: {job_id}[/green]")
        except Exception as exc:
            self.raw_output.write_line(f"[red]✗ Orchestrator failed: {exc}[/red]")
        finally:
            self.job_running = False
            self.goal_input.disabled = False
            self.submit_btn.disabled = False
            self.raw_output.write_line("[cyan]Ready for next goal[/cyan]")

    def _handle_event(self, event: dict[str, Any]) -> None:
        """Process events from UIEventBus."""
        t = event.get("type")

        if t == "job_state_changed":
            jid = event.get("job_id")
            self.job_info._job_id = jid
            self.job_info._state = event.get("new_state", "unknown")
            self.job_info._goal = event.get("goal", getattr(self.job_info, "_goal", ""))
            if not hasattr(self.job_info, "_start_time") or not self.job_info._start_time:
                self.job_info._start_time = datetime.now(timezone.utc).timestamp()
            self.raw_output.write_line(f"[green]Job {jid} is {self.job_info._state}[/green]")

        elif t == "agent_log":
            msg = event.get("message", "")
            self.raw_output.write_line(msg)

        elif t == "prompt_sent":
            agent = event.get("agent_type", "?").upper()
            prompt = event.get("prompt", "")
            # Show a header line + the full prompt in the log
            self.raw_output.write_line(f"[bold cyan]── {agent} PROMPT ──[/bold cyan]")
            for line in prompt.splitlines():
                self.raw_output.write_line(f"[cyan]{line}[/cyan]")
            self.raw_output.write_line(f"[bold cyan]── END PROMPT ──[/bold cyan]")

        elif t == "agent_progress":
            # Only log the final completed response, not every streaming chunk
            if event.get("done"):
                agent = event.get("agent_type", "?").upper()
                partial = event.get("partial", "")
                self.raw_output.write_line(f"[bold yellow]── {agent} RESPONSE ──[/bold yellow]")
                for line in partial.splitlines():
                    self.raw_output.write_line(f"[yellow]{line}[/yellow]")
                self.raw_output.write_line(f"[bold yellow]── END RESPONSE ──[/bold yellow]")

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
            if not hasattr(self.agent_status, "_agents"):
                self.agent_status._agents = {}

            task_info = self.progress._tasks.get(tid, {})
            if state == "running" or not task_info:
                task_info["start_time"] = datetime.now(timezone.utc).timestamp()
            task_info["state"] = state
            task_info["agent_type"] = agent
            task_info["elapsed"] = task_info.get("elapsed", 0)
            self.progress._tasks[tid] = task_info

            agent_info = self.agent_status._agents.get(agent, {})
            if state == "running":
                agent_info["start_time"] = agent_info.get("start_time", datetime.now(timezone.utc).timestamp())
            agent_info["state"] = state
            agent_info["elapsed"] = agent_info.get("elapsed", 0)
            self.agent_status._agents[agent] = agent_info

        elif t == "workspace_created":
            ws = event.get("workspace_path")
            if ws:
                self.job_info._workspace = str(ws)
                self._update_file_tree(str(ws))
                self.raw_output.write_line(f"[blue]Workspace created:[/blue] {ws}")

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
