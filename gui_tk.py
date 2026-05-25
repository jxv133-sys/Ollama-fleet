#!/usr/bin/env python3
"""Tkinter GUI for Ollama Fleet - replaces terminal-based UI.

Run:
    python3 gui_tk.py --demo
    python3 gui_tk.py --goal "Create REST API"
    python3 gui_tk.py --goal "Build calculator" --demo

The GUI runs in the main thread; the orchestrator runs in a background
thread with its own asyncio loop. Events are delivered via a thread-safe
queue and polled into the Tk mainloop.
"""
from __future__ import annotations

import argparse
import asyncio
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from ollama_fleet.config import load_settings
from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.orchestrator import Orchestrator


class TkEventBus:
    """Adapter used by the orchestrator to publish events into a Queue."""

    def __init__(self, q: "queue.Queue[dict[str, Any]]") -> None:
        self.q = q

    def publish(self, event: dict[str, Any]) -> None:
        try:
            self.q.put_nowait(event)
        except Exception:
            pass


class FleetTkApp(tk.Tk):
    def __init__(self, event_queue: "queue.Queue[dict[str, Any]]", goal: str = "Demo GUI", use_demo: bool = False) -> None:
        super().__init__()
        self.title("Ollama Fleet - Multi-Agent Orchestration")
        self.geometry("1400x900")
        self.event_queue = event_queue
        self.goal = goal
        self.use_demo = use_demo
        self.start_time: float | None = None
        self.elapsed_var = tk.StringVar(value="0s")
        
        self._build_ui()
        self._poll()

    def _build_ui(self) -> None:
        """Build the complete Tkinter UI with modern layout."""
        # Configure grid weights for responsive layout
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # ========== Header Frame ==========
        header = ttk.Frame(self, relief=tk.SUNKEN, borderwidth=2)
        header.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        header.grid_columnconfigure(2, weight=1)
        
        ttk.Label(header, text="Goal:", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, padx=4, pady=4)
        self.goal_var = tk.StringVar(value=self.goal)
        goal_entry = ttk.Entry(header, textvariable=self.goal_var, font=("TkDefaultFont", 10), width=50)
        goal_entry.grid(row=0, column=1, padx=4, pady=4)
        
        button_frame = ttk.Frame(header)
        button_frame.grid(row=0, column=2, padx=4, pady=4, sticky="e")
        
        self.start_btn = ttk.Button(button_frame, text="▶ Start Job", command=self._on_start)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop", command=self._on_stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # ========== Main Content ==========
        main_container = ttk.Frame(self)
        main_container.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Left panel (Job info, Agents, Files)
        left_panel = ttk.Frame(main_container, relief=tk.RIDGE, borderwidth=2)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left_panel.grid_rowconfigure(3, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)
        
        # Job Info Section
        self._build_job_info_section(left_panel)
        
        # Agents Section
        self._build_agents_section(left_panel)
        
        # Right panel (Progress & Output)
        right_panel = ttk.Frame(main_container, relief=tk.RIDGE, borderwidth=2)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Tasks/Progress Section
        self._build_progress_section(right_panel)
        
        # Output Section
        self._build_output_section(right_panel)
        
        # ========== Status Bar ==========
        status_frame = ttk.Frame(self, relief=tk.SUNKEN, borderwidth=1)
        status_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        
        ttk.Label(status_frame, text="Elapsed:", font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT, padx=4)
        ttk.Label(status_frame, textvariable=self.elapsed_var, font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=4)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(20, 4))
        
        # State tracking
        self._orchestrator_thread: threading.Thread | None = None
        self._orchestrator_stop = threading.Event()
        self._orchestrator = None
        self._job_running = False
        
        # Auto-start if goal provided via CLI and demo mode
        if self.goal != "Demo GUI" or self.use_demo:
            self.after(100, self._on_start)

    def _build_job_info_section(self, parent: ttk.Frame) -> None:
        """Build the job information display section."""
        ttk.Label(parent, text="Job Information", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w", padx=4, pady=(4, 2))
        
        info_frame = ttk.Frame(parent, relief=tk.FLAT, borderwidth=1)
        info_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        info_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(info_frame, text="ID:", font=("TkDefaultFont", 9)).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.job_id_var = tk.StringVar(value="–")
        ttk.Label(info_frame, textvariable=self.job_id_var, font=("TkDefaultFont", 9, "bold"), foreground="blue").grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        
        ttk.Label(info_frame, text="State:", font=("TkDefaultFont", 9)).grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.job_state_var = tk.StringVar(value="idle")
        self.job_state_label = ttk.Label(info_frame, textvariable=self.job_state_var, font=("TkDefaultFont", 9, "bold"), foreground="orange")
        self.job_state_label.grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        
        ttk.Label(info_frame, text="Goal:", font=("TkDefaultFont", 9)).grid(row=2, column=0, sticky="nw", padx=4, pady=2)
        self.job_goal_var = tk.StringVar(value="–")
        goal_label = ttk.Label(info_frame, textvariable=self.job_goal_var, font=("TkDefaultFont", 9), wraplength=300, justify=tk.LEFT)
        goal_label.grid(row=2, column=1, sticky="ew", padx=4, pady=2)

    def _build_agents_section(self, parent: ttk.Frame) -> None:
        """Build the agent status section."""
        ttk.Label(parent, text="Agent Status", font=("TkDefaultFont", 11, "bold")).grid(row=2, column=0, sticky="w", padx=4, pady=(8, 2))
        
        # Frame for agents with scrollbar
        agents_container = ttk.Frame(parent, relief=tk.FLAT, borderwidth=1)
        agents_container.grid(row=3, column=0, sticky="nsew", padx=4, pady=2)
        agents_container.grid_rowconfigure(0, weight=1)
        agents_container.grid_columnconfigure(0, weight=1)
        
        scrollbar = ttk.Scrollbar(agents_container)
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.agents_frame = ttk.Frame(agents_container)
        canvas = tk.Canvas(agents_container, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        
        canvas.create_window(0, 0, window=self.agents_frame, anchor="nw")
        canvas.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=canvas.yview)
        
        self.agents_update_canvas = canvas
        self.agents_vars: dict[str, tk.StringVar] = {}
        self.agents_labels: dict[str, ttk.Label] = {}
        
        for agent_name in ["planner", "coder", "critic", "tester", "synthesizer"]:
            agent_var = tk.StringVar(value="● Idle")
            self.agents_vars[agent_name] = agent_var
            label = ttk.Label(self.agents_frame, textvariable=agent_var, font=("TkDefaultFont", 9), foreground="gray")
            label.pack(anchor="w", padx=4, pady=2)
            self.agents_labels[agent_name] = label
        
        def on_agents_frame_configure():
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        self.agents_frame.bind("<Configure>", lambda e: on_agents_frame_configure())

    def _build_progress_section(self, parent: ttk.Frame) -> None:
        """Build the task progress section."""
        ttk.Label(parent, text="Task Progress", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=4, pady=(4, 2))
        
        # Treeview for tasks
        self.tasks_tree = ttk.Treeview(parent, columns=("agent", "state", "elapsed"), height=8, show="headings")
        self.tasks_tree.column("agent", width=80, anchor="w")
        self.tasks_tree.column("state", width=100, anchor="center")
        self.tasks_tree.column("elapsed", width=70, anchor="center")
        
        self.tasks_tree.heading("agent", text="Agent")
        self.tasks_tree.heading("state", text="State")
        self.tasks_tree.heading("elapsed", text="Elapsed")
        
        # Scrollbar
        tree_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tasks_tree.yview)
        self.tasks_tree.configure(yscroll=tree_scroll.set)
        
        self.tasks_tree.pack(fill=tk.BOTH, expand=False, padx=4, pady=2)
        tree_scroll.pack(fill=tk.Y, side=tk.RIGHT)

    def _build_output_section(self, parent: ttk.Frame) -> None:
        """Build the raw output section."""
        ttk.Label(parent, text="Output Log", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=4, pady=(8, 2))
        
        # Scrolled text
        self.raw_output = scrolledtext.ScrolledText(parent, height=20, font=("Courier", 9), wrap=tk.WORD)
        self.raw_output.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        
        # Configure text tags for colored output
        self.raw_output.tag_config("info", foreground="blue")
        self.raw_output.tag_config("success", foreground="green")
        self.raw_output.tag_config("warning", foreground="orange")
        self.raw_output.tag_config("error", foreground="red")
        self.raw_output.tag_config("debug", foreground="gray")
        self.raw_output.tag_config("planner", foreground="#0066CC")
        self.raw_output.tag_config("coder", foreground="#00AA00")
        self.raw_output.tag_config("critic", foreground="#CC6600")
        self.raw_output.tag_config("tester", foreground="#AA0000")
        self.raw_output.tag_config("synthesizer", foreground="#6600CC")

    def _on_start(self) -> None:
        if self._job_running:
            messagebox.showinfo("Info", "Job already running")
            return
        goal = self.goal_var.get().strip() or "Demo GUI"
        self._orchestrator_stop.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._job_running = True
        self.start_time = time.time()
        
        # Display goal immediately
        self.job_goal_var.set(goal)
        self.job_state_var.set("submitted")
        self.job_state_label.config(foreground="orange")
        
        # Clear agents status
        for agent_name in self.agents_vars:
            self.agents_vars[agent_name].set("● Idle")
            self.agents_labels[agent_name].config(foreground="gray")
        
        self._orchestrator_thread = threading.Thread(target=self._start_orchestrator_thread, args=(goal,), daemon=True)
        self._orchestrator_thread.start()
        self.status_var.set("Starting job...")
        self._update_elapsed_time()

    def _on_stop(self) -> None:
        """Stop the running job."""
        self._orchestrator_stop.set()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._job_running = False
        self.status_var.set("Job stopped")

    def _start_orchestrator_thread(self, goal: str) -> None:
        """Run orchestrator in background thread with its own asyncio loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        settings = load_settings()
        db = Database(Path("ollama_fleet.db"))

        async def run_orch() -> None:
            await db.connect()
            try:
                event_bus = TkEventBus(self.event_queue)
                orch = Orchestrator(db, settings, ui_bus=event_bus)
                
                # Debug event
                self.event_queue.put_nowait({
                    "type": "agent_log",
                    "message": "Orchestrator initialized, use_demo=" + str(self.use_demo),
                    "level": "debug"
                })
                
                # Use demo executor if enabled
                if self.use_demo:
                    try:
                        from scripts.demo_run import DummyExecutor
                        orch.executor = DummyExecutor(settings)
                        self.event_queue.put_nowait({
                            "type": "agent_log",
                            "message": "DummyExecutor loaded",
                            "level": "debug"
                        })
                    except Exception as e:
                        self.event_queue.put_nowait({
                            "type": "agent_log",
                            "message": f"Failed to load DummyExecutor: {e}",
                            "level": "error"
                        })
                
                self.event_queue.put_nowait({
                    "type": "agent_log",
                    "message": f"Submitting job with goal: {goal}",
                    "level": "debug"
                })
                
                job_id = await orch.submit_job(goal=goal, config={"source": "gui_tk"})
                
                self.event_queue.put_nowait({
                    "type": "agent_log",
                    "message": f"Job submitted successfully: {job_id}",
                    "level": "info"
                })
            except Exception as e:
                import traceback
                self.event_queue.put_nowait({
                    "type": "agent_log",
                    "message": f"Orchestrator error: {e}",
                    "level": "error"
                })
                self.event_queue.put_nowait({
                    "type": "agent_log",
                    "message": traceback.format_exc(),
                    "level": "error"
                })
            finally:
                await db.close()

        try:
            loop.run_until_complete(run_orch())
        except Exception as e:
            import traceback
            try:
                self.event_queue.put_nowait({
                    "type": "agent_log",
                    "message": f"Run error: {e}",
                    "level": "error"
                })
                self.event_queue.put_nowait({
                    "type": "agent_log",
                    "message": traceback.format_exc(),
                    "level": "error"
                })
            except Exception:
                pass
        finally:
            try:
                loop.close()
            except Exception:
                pass
            # Signal job completion
            try:
                self.event_queue.put_nowait({
                    "type": "job_state_changed",
                    "job_id": "",
                    "new_state": "completed"
                })
            except Exception:
                pass

    def _update_elapsed_time(self) -> None:
        """Update elapsed time display."""
        if self._job_running and self.start_time:
            elapsed = int(time.time() - self.start_time)
            self.elapsed_var.set(f"{elapsed}s")
            # Only schedule next update if window is still valid
            try:
                self.after(1000, self._update_elapsed_time)
            except tk.TclError:
                # Window has been destroyed
                pass

    def _poll(self) -> None:
        """Poll events from queue and update UI."""
        try:
            while True:
                event = self.event_queue.get_nowait()
                self._process_event(event)
        except queue.Empty:
            pass
        # Only schedule next poll if window is still valid
        try:
            self.after(100, self._poll)
        except tk.TclError:
            # Window has been destroyed
            pass

    def _process_event(self, event: dict[str, Any]) -> None:
        """Process a single event from the orchestrator."""
        event_type = event.get("type")
        
        if event_type == "job_state_changed":
            self._handle_job_state_changed(event)
        elif event_type == "agent_log":
            self._handle_agent_log(event)
        elif event_type == "agent_output":
            self._handle_agent_output(event)
        elif event_type == "task_state_changed":
            self._handle_task_state_changed(event)
        elif event_type == "file_written":
            self._handle_file_written(event)
        elif event_type == "validation_result":
            self._handle_validation_result(event)
        elif event_type == "escalation_added":
            self._handle_escalation_added(event)

    def _handle_job_state_changed(self, event: dict[str, Any]) -> None:
        """Handle job state change event."""
        job_id = event.get("job_id", "–")
        new_state = event.get("new_state", "–")
        goal = event.get("goal")
        
        self.job_id_var.set(job_id[:12] if job_id else "–")
        self.job_state_var.set(new_state)
        
        # Update goal if provided
        if goal:
            self.job_goal_var.set(goal)
        
        # Update label color based on state
        state_colors = {
            "idle": "gray",
            "running": "orange",
            "completed": "green",
            "failed": "red",
            "stopped": "gray"
        }
        self.job_state_label.config(foreground=state_colors.get(new_state, "black"))
        
        if new_state in ("completed", "failed", "stopped"):
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self._job_running = False
            status_msg = f"Job {new_state}"
            self.status_var.set(status_msg)
        else:
            self.status_var.set(f"Job {new_state}")
        
        # Log the state change
        self._append_output(f"Job {new_state.upper()}", "info")

    def _handle_agent_log(self, event: dict[str, Any]) -> None:
        """Handle agent log message."""
        msg = event.get("message", "")
        level = event.get("level", "debug")
        self._append_output(msg, level)
        
        # Parse agent logs to update agent status
        msg_lower = msg.lower()
        if "planner" in msg_lower and "created" in msg_lower:
            # Planner completed: "[PLANNER] Created X tasks ready for execution"
            self.agents_vars["planner"].set("✓ Completed")
            self.agents_labels["planner"].config(foreground="green")
            self._append_output("Planner agent completed", "info")

    def _handle_agent_output(self, event: dict[str, Any]) -> None:
        """Handle agent output."""
        agent_type = event.get("agent_type", "?").lower()
        output = event.get("output", "")
        
        # Format: [AGENT] output with agent-specific color
        msg = f"[{agent_type.upper()}] {output}"
        self._append_output(msg, agent_type)
        
        # Update agent status for agents with output
        if agent_type in self.agents_labels:
            # Agent is running/producing output
            if "error" not in str(output).lower():
                self.agents_vars[agent_type].set(f"✓ Completed")
                self.agents_labels[agent_type].config(foreground="green")

    def _handle_task_state_changed(self, event: dict[str, Any]) -> None:
        """Handle task state change."""
        task_id = event.get("task_id", "?")
        new_state = event.get("new_state", "–")
        agent_type = event.get("agent_type", "?").lower()
        
        # Update agent status based on task state
        if agent_type in self.agents_labels:
            if new_state == "running":
                self.agents_vars[agent_type].set(f"● Running")
                self.agents_labels[agent_type].config(foreground="orange")
            elif new_state == "completed":
                self.agents_vars[agent_type].set(f"✓ Completed")
                self.agents_labels[agent_type].config(foreground="green")
            elif new_state == "failed":
                self.agents_vars[agent_type].set(f"✗ Failed")
                self.agents_labels[agent_type].config(foreground="red")
        
        # Check if task already exists
        found = False
        for child in self.tasks_tree.get_children():
            item_text = self.tasks_tree.item(child, "text")
            if item_text == task_id:
                self.tasks_tree.item(child, values=(agent_type, new_state, "–"))
                found = True
                break
        
        if not found:
            self.tasks_tree.insert("", tk.END, text=task_id, values=(agent_type, new_state, "–"))
        
        self._append_output(f"Task {task_id}: {new_state} ({agent_type})", "debug")

    def _handle_file_written(self, event: dict[str, Any]) -> None:
        """Handle file written event."""
        path = event.get("path", "?")
        self._append_output(f"File created: {path}", "success")

    def _handle_validation_result(self, event: dict[str, Any]) -> None:
        """Handle validation result."""
        result = event.get("validation_result", {})
        status = result.get("status", "unknown")
        msg = f"Validation {status}: {result.get('message', '')}"
        level = "success" if status == "valid" else "warning"
        self._append_output(msg, level)

    def _handle_escalation_added(self, event: dict[str, Any]) -> None:
        """Handle escalation event."""
        esc = event.get("escalation", {})
        reason = esc.get("reason", "unknown")
        self._append_output(f"⚠ ESCALATION: {reason}", "warning")

    def _append_output(self, msg: str, tag: str = "debug") -> None:
        """Append message to output log with optional tag."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {msg}\n"
        
        self.raw_output.insert(tk.END, log_line, tag)
        self.raw_output.see(tk.END)
        self.raw_output.update_idletasks()


def main() -> int:
    parser = argparse.ArgumentParser(prog="gui_tk.py", description="Tkinter GUI for Ollama Fleet")
    parser.add_argument("--demo", action="store_true", help="Use demo executor (no Ollama server required)")
    parser.add_argument("--goal", default="Demo GUI", help="Initial goal for the job")
    parser.add_argument("--db-path", default="ollama_fleet.db", help="SQLite database path")
    args = parser.parse_args()

    q: "queue.Queue[dict[str, Any]]" = queue.Queue()
    app = FleetTkApp(q, goal=args.goal, use_demo=args.demo)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
