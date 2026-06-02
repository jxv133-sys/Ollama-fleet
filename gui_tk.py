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
import json
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
    def __init__(
        self,
        event_queue: "queue.Queue[dict[str, Any]]",
        goal: str = "Demo GUI",
        use_demo: bool = False,
        db_path: str | Path = "ollama_fleet.db",
    ) -> None:
        super().__init__()
        self.title("Ollama Fleet - Multi-Agent Orchestration")
        self.geometry("1400x900")
        self.event_queue = event_queue
        self.goal = goal
        self.use_demo = use_demo
        self.db_path = Path(db_path)
        self.start_time: float | None = None
        self.elapsed_var = tk.StringVar(value="0s")
        self.jobs_by_id: dict[str, dict[str, str]] = {}
        self.initial_settings = load_settings()
        self.model_vars: dict[str, tk.StringVar] = {}
        self.model_menus: dict[str, tk.OptionMenu] = {}
        self.default_model_ids = self._default_model_ids()
        self._selected_models: dict[str, str] = {}
        self._selected_base_url = self.initial_settings.ollama.base_url
        
        self._build_ui()
        self._refresh_jobs_from_db()
        self.after(250, self._fetch_models_async)
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

        self._build_model_selection_section(header)
        
        # ========== Main Content ==========
        main_container = ttk.Frame(self)
        main_container.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Left panel (Job info, Agents, Files)
        left_panel = ttk.Frame(main_container, relief=tk.RIDGE, borderwidth=2)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_rowconfigure(3, weight=1)
        left_panel.grid_rowconfigure(5, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)
        
        # AI Chat Section
        self._build_ai_chat_section(left_panel)
        
        # Agents Section
        self._build_agents_section(left_panel)

        # Jobs Section
        self._build_jobs_section(left_panel)
        
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
        
        self.file_changes: list[str] = []
        self.iteration_count = 0
        self.tests_status = "pending"

        models_frame = ttk.Frame(parent)
        models_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=4, pady=(0, 4))
        models_frame.grid_columnconfigure(1, weight=1)
        models_frame.grid_columnconfigure(3, weight=1)
        models_frame.grid_columnconfigure(5, weight=1)

        ttk.Label(models_frame, text="Models:", font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.models_url_var = tk.StringVar(value=self.initial_settings.ollama.base_url)
        ttk.Entry(models_frame, textvariable=self.models_url_var, width=32).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(models_frame, text="Load Models", command=self._fetch_models_async).grid(row=0, column=2, sticky="w", padx=2)
        self.models_status_var = tk.StringVar(value="Models not loaded")
        ttk.Label(models_frame, textvariable=self.models_status_var, font=("TkDefaultFont", 9)).grid(row=0, column=3, sticky="w", padx=6)

        defaults = {
            "planner": self.initial_settings.ollama.planner_model,
            "coder": self.initial_settings.ollama.coder_model,
            "critic": self.initial_settings.ollama.critic_model or self.initial_settings.ollama.coder_model,
            "tester": self.initial_settings.ollama.tester_model or self.initial_settings.ollama.coder_model,
            "synthesizer": self.initial_settings.ollama.summarizer_model,
        }
        labels = {
            "planner": "All-around",
            "coder": "Coding",
            "critic": "Critic",
            "tester": "Testing",
            "synthesizer": "Summary",
        }
        positions = {
            "planner": (1, 0),
            "coder": (1, 2),
            "critic": (1, 4),
            "tester": (2, 0),
            "synthesizer": (2, 2),
        }
        for key in ("planner", "coder", "critic", "tester", "synthesizer"):
            row, column = positions[key]
            ttk.Label(models_frame, text=f"{labels[key]}:").grid(row=row, column=column, sticky="w", padx=(0, 2), pady=(4, 0))
            var = tk.StringVar(value=defaults[key])
            menu = tk.OptionMenu(models_frame, var, *self.default_model_ids)
            menu.config(width=34, anchor="w")
            menu.grid(row=row, column=column + 1, sticky="ew", padx=(0, 8), pady=(4, 0))
            self.model_vars[key] = var
            self.model_menus[key] = menu

    def _build_ai_chat_section(self, parent: ttk.Frame) -> None:
        """Build the AI summary display section (chat removed)."""
        ttk.Label(parent, text="AI Status", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w", padx=4, pady=(4, 2))

        self.job_id_var = tk.StringVar(value="–")
        self.job_state_var = tk.StringVar(value="idle")
        self.job_goal_var = tk.StringVar(value="–")

        self.ai_summary_var = tk.StringVar(value="Files: none | Iterations: 0 | Tests: pending")
        ttk.Label(parent, textvariable=self.ai_summary_var, font=("TkDefaultFont", 9, "italic")).grid(row=1, column=0, sticky="w", padx=4, pady=(0, 2))

        # Container frame for layering loading indicator and summary
        container = ttk.Frame(parent)
        container.grid(row=2, column=0, sticky="nsew", padx=4, pady=2)

        # Loading indicator frame
        self.loading_frame = ttk.Frame(container)
        self.loading_frame.place(x=0, y=0, relwidth=1, relheight=1)
        self.loading_label = ttk.Label(self.loading_frame, text="⏳ Waiting on agent response...", font=("TkDefaultFont", 9, "italic"), foreground="gray")
        self.loading_label.pack(fill=tk.BOTH, expand=True)
        self.loading_visible = False

        # Summary text area
        self.ai_chat = scrolledtext.ScrolledText(
            container,
            height=8,
            font=("TkDefaultFont", 9),
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.ai_chat.place(x=0, y=0, relwidth=1, relheight=1)
        self.ai_chat.tag_config("planner", foreground="#0066CC")
        self.ai_chat.tag_config("coder", foreground="#00AA00")
        self.ai_chat.tag_config("critic", foreground="#CC6600")
        self.ai_chat.tag_config("tester", foreground="#AA0000")
        self.ai_chat.tag_config("synthesizer", foreground="#6600CC")
        self.ai_chat.tag_config("system", foreground="gray")
        self.ai_chat.tag_config("error", foreground="red")

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

    def _build_jobs_section(self, parent: ttk.Frame) -> None:
        """Build the persisted jobs list."""
        ttk.Label(parent, text="Jobs", font=("TkDefaultFont", 11, "bold")).grid(row=4, column=0, sticky="w", padx=4, pady=(8, 2))

        jobs_container = ttk.Frame(parent, relief=tk.FLAT, borderwidth=1)
        jobs_container.grid(row=5, column=0, sticky="nsew", padx=4, pady=2)
        jobs_container.grid_rowconfigure(0, weight=1)
        jobs_container.grid_columnconfigure(0, weight=1)

        self.jobs_tree = ttk.Treeview(
            jobs_container,
            columns=("id", "state", "goal", "updated"),
            height=8,
            show="headings",
        )
        self.jobs_tree.heading("id", text="ID")
        self.jobs_tree.heading("state", text="State")
        self.jobs_tree.heading("goal", text="Goal")
        self.jobs_tree.heading("updated", text="Updated")
        self.jobs_tree.column("id", width=95, anchor="w")
        self.jobs_tree.column("state", width=80, anchor="center")
        self.jobs_tree.column("goal", width=230, anchor="w")
        self.jobs_tree.column("updated", width=85, anchor="w")
        self.jobs_tree.grid(row=0, column=0, sticky="nsew")

        jobs_scroll = ttk.Scrollbar(jobs_container, orient=tk.VERTICAL, command=self.jobs_tree.yview)
        jobs_scroll.grid(row=0, column=1, sticky="ns")
        self.jobs_tree.configure(yscrollcommand=jobs_scroll.set)
        self.jobs_tree.bind("<<TreeviewSelect>>", self._on_job_selected)

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

    def _default_model_ids(self) -> list[str]:
        """Return unique configured model IDs for initial dropdown values."""
        models = [
            self.initial_settings.ollama.planner_model,
            self.initial_settings.ollama.coder_model,
            self.initial_settings.ollama.critic_model,
            self.initial_settings.ollama.tester_model,
            self.initial_settings.ollama.summarizer_model,
        ]
        return sorted({model for model in models if model})

    def _fetch_models_async(self) -> None:
        """Fetch model IDs without blocking the Tk mainloop."""
        self.models_status_var.set("Loading models...")
        base_url = self.models_url_var.get().strip()
        thread = threading.Thread(target=self._fetch_models_worker, args=(base_url,), daemon=True)
        thread.start()

    def _fetch_models_worker(self, base_url: str) -> None:
        try:
            model_ids = self._request_model_ids(base_url)
        except Exception as exc:
            self.event_queue.put_nowait(
                {
                    "type": "models_load_failed",
                    "message": str(exc),
                }
            )
            return

        self.event_queue.put_nowait(
            {
                "type": "models_loaded",
                "models": model_ids,
            }
        )

    @staticmethod
    def _request_model_ids(base_url: str) -> list[str]:
        """Request model IDs from an OpenAI-compatible /v1/models endpoint."""
        if not base_url:
            raise ValueError("Model server URL is empty")

        clean_url = base_url.rstrip("/")
        if clean_url.endswith("/v1/models"):
            endpoint = clean_url
        else:
            endpoint = f"{clean_url}/v1/models"

        request = Request(endpoint, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=10.0) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model request failed with HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not connect to model server: {exc.reason}") from exc

        payload = json.loads(raw)
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            model_ids = [
                str(item["id"])
                for item in payload["data"]
                if isinstance(item, dict) and item.get("id")
            ]
        elif isinstance(payload, dict) and isinstance(payload.get("models"), list):
            model_ids = [
                str(item.get("name") or item.get("id"))
                for item in payload["models"]
                if isinstance(item, dict) and (item.get("name") or item.get("id"))
            ]
        elif isinstance(payload, list):
            model_ids = [
                str(item.get("id") or item.get("name") if isinstance(item, dict) else item)
                for item in payload
            ]
        else:
            raise ValueError("Model server returned an unrecognized model list format")

        unique_ids = sorted({model_id for model_id in model_ids if model_id})
        if not unique_ids:
            raise ValueError("Model server returned no models")
        return unique_ids

    def _handle_models_loaded(self, event: dict[str, Any]) -> None:
        models = event.get("models", [])
        if not isinstance(models, list):
            models = []
        values = tuple(sorted({*self.default_model_ids, *(str(model) for model in models)}))
        self._update_model_menus(values)
        self.models_status_var.set(f"{len(models)} models loaded")
        self._append_output(f"Loaded {len(models)} models from {self.models_url_var.get().strip()}/v1/models", "info")

    def _handle_models_load_failed(self, event: dict[str, Any]) -> None:
        message = event.get("message", "unknown error")
        self._update_model_menus(tuple(self.default_model_ids))
        self.models_status_var.set("Model load failed")
        self._append_output(f"Could not load models: {message}", "warning")

    def _update_model_menus(self, values: tuple[str, ...]) -> None:
        """Replace OptionMenu choices while preserving current selections."""
        for key, menu_button in self.model_menus.items():
            variable = self.model_vars[key]
            current_value = variable.get()
            choices = values
            if current_value and current_value not in choices:
                choices = tuple(sorted({*choices, current_value}))
            menu = menu_button["menu"]
            menu.delete(0, "end")
            for choice in choices:
                menu.add_command(label=choice, command=tk._setit(variable, choice))

    def _apply_selected_models(self, settings: Any) -> None:
        """Apply GUI model choices to runtime settings."""
        selected = self._selected_models
        if selected.get("planner"):
            settings.ollama.planner_model = selected["planner"]
        if selected.get("coder"):
            settings.ollama.coder_model = selected["coder"]
        if selected.get("critic"):
            settings.ollama.critic_model = selected["critic"]
        if selected.get("tester"):
            settings.ollama.tester_model = selected["tester"]
        if selected.get("synthesizer"):
            settings.ollama.summarizer_model = selected["synthesizer"]

        base_url = self._selected_base_url
        if base_url:
            settings.ollama.base_url = base_url.removesuffix("/v1/models").rstrip("/")

    def _on_start(self) -> None:
        if self._job_running:
            messagebox.showinfo("Info", "Job already running")
            return
        goal = self.goal_var.get().strip() or "Demo GUI"
        self._selected_models = {
            key: var.get().strip()
            for key, var in self.model_vars.items()
            if var.get().strip()
        }
        self._selected_base_url = self.models_url_var.get().strip()
        self._orchestrator_stop.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._job_running = True
        self.start_time = time.time()
        
        # Display goal immediately
        self.job_goal_var.set(goal)
        self.job_state_var.set("submitted")
        self.ai_chat.configure(state=tk.NORMAL)
        self.ai_chat.delete("1.0", tk.END)
        self.ai_chat.configure(state=tk.DISABLED)
        self._append_chat("System", f"New job submitted: {goal}", "system")
        
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
        self._apply_selected_models(settings)
        db = Database(self.db_path)

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
                job = await orch.job_manager.get_job(job_id)
                final_state = job.state if job is not None else "finished"
                self.event_queue.put_nowait({
                    "type": "agent_log",
                    "message": f"Job {final_state}: {job_id}",
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
                self.event_queue.put_nowait({
                    "type": "job_state_changed",
                    "job_id": "",
                    "new_state": "failed"
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
        elif event_type == "models_loaded":
            self._handle_models_loaded(event)
        elif event_type == "models_load_failed":
            self._handle_models_load_failed(event)

    def _refresh_jobs_from_db(self) -> None:
        """Load persisted jobs into the jobs table."""
        try:
            with sqlite3.connect(self.db_path, timeout=2.0) as conn:
                rows = conn.execute(
                    """
                    SELECT job_id, goal, state, created_at, updated_at
                    FROM jobs
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT 100
                    """
                ).fetchall()
        except sqlite3.Error:
            return

        seen_job_ids: set[str] = set()
        for job_id, goal, state, created_at, updated_at in rows:
            seen_job_ids.add(job_id)
            updated_display = self._format_db_timestamp(updated_at or created_at)
            self.jobs_by_id[job_id] = {
                "job_id": job_id,
                "goal": goal or "",
                "state": state or "",
                "created_at": created_at or "",
                "updated_at": updated_at or "",
            }
            values = (job_id[:8], state, goal, updated_display)
            if self.jobs_tree.exists(job_id):
                self.jobs_tree.item(job_id, values=values)
            else:
                self.jobs_tree.insert("", tk.END, iid=job_id, values=values)

        for child in self.jobs_tree.get_children():
            if child not in seen_job_ids:
                self.jobs_tree.delete(child)

    def _load_tasks_for_job(self, job_id: str) -> None:
        """Load persisted tasks for the selected job."""
        try:
            with sqlite3.connect(self.db_path, timeout=2.0) as conn:
                rows = conn.execute(
                    """
                    SELECT task_id, agent_type, state
                    FROM tasks
                    WHERE job_id = ?
                    ORDER BY created_at ASC
                    """,
                    (job_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            self._append_output(f"Could not load tasks for job {job_id[:8]}: {exc}", "warning")
            return

        for child in self.tasks_tree.get_children():
            self.tasks_tree.delete(child)
        for task_id, agent_type, state in rows:
            self.tasks_tree.insert("", tk.END, iid=task_id, text=task_id, values=(agent_type, state, "–"))

    def _on_job_selected(self, _event: tk.Event) -> None:
        selected = self.jobs_tree.selection()
        if not selected:
            return
        job_id = selected[0]
        job = self.jobs_by_id.get(job_id)
        if job is None:
            return

        self.job_id_var.set(job_id[:12])
        self.job_state_var.set(job["state"])
        self.job_goal_var.set(job["goal"])
        self.status_var.set(f"Selected job {job_id[:8]} ({job['state']})")
        self._load_tasks_for_job(job_id)

    @staticmethod
    def _format_db_timestamp(value: str) -> str:
        if not value:
            return "–"
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return value[:16]
        return parsed.strftime("%H:%M:%S")

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
        
        if new_state in ("completed", "failed", "stopped"):
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self._job_running = False
            status_msg = f"Job {new_state}"
            self.status_var.set(status_msg)
        else:
            self.status_var.set(f"Job {new_state}")
        
        self._refresh_jobs_from_db()
        if job_id and self.jobs_tree.exists(job_id):
            self.jobs_tree.selection_set(job_id)
            self.jobs_tree.see(job_id)

        # Log the state change
        self._append_output(f"Job {new_state.upper()}", "info")
        if new_state in ("completed", "failed"):
            self._show_loading(False)

    def _handle_agent_log(self, event: dict[str, Any]) -> None:
        """Handle agent log message."""
        msg = event.get("message", "")
        level = event.get("level", "debug")
        self._append_output(msg, level)

    def _handle_agent_output(self, event: dict[str, Any]) -> None:
        """Handle agent output."""
        self._show_loading(False)
        agent_type = event.get("agent_type", "?").lower()
        output = event.get("output", "")
        prompt = event.get("prompt") or (output.get("prompt") if isinstance(output, dict) else None)
        
        # Show prompt and response in the summary panel.
        if prompt:
            short_prompt = self._shorten_text(str(prompt), max_lines=8)
            self._append_chat(f"{agent_type.title()} Prompt", short_prompt, agent_type)
        self._append_chat(f"{agent_type.title()} Response", self._format_agent_message(output), agent_type)
        self._append_output(f"[{agent_type.upper()}] {output}", agent_type)

        if agent_type == "coder":
            self.iteration_count += 1
            self._update_summary_label()

        # Update agent status for agents with output
        if agent_type in self.agents_labels:
            if "error" not in str(output).lower():
                self.agents_vars[agent_type].set(f"✓ Completed")
                self.agents_labels[agent_type].config(foreground="green")

    def _handle_task_state_changed(self, event: dict[str, Any]) -> None:
        """Handle task state change."""
        task_id = event.get("task_id", "?")
        new_state = event.get("new_state", "–")
        agent_type = event.get("agent_type", "?").lower()
        
        # Show loading when tasks are running, hide when complete or failed
        if new_state == "running":
            self._show_loading(True)
        elif new_state in ("completed", "failed"):
            self._show_loading(False)
        
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
                reason = event.get("reason")
                if reason:
                    self._append_chat(agent_type.title(), f"Failed: {reason}", "error")
        
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
        if path not in self.file_changes:
            self.file_changes.append(path)
        self._update_summary_label()
        self._append_output(f"File created: {path}", "success")

    def _handle_validation_result(self, event: dict[str, Any]) -> None:
        """Handle validation result."""
        result = event.get("validation_result", {})
        status = result.get("status", "unknown")
        self.tests_status = "OK" if status == "valid" else "Failed"
        self._update_summary_label()
        msg = f"Validation {status}: {result.get('message', '')}"
        level = "success" if status == "valid" else "warning"
        self._append_output(msg, level)

    def _handle_escalation_added(self, event: dict[str, Any]) -> None:
        """Handle escalation event."""
        esc = event.get("escalation", {})
        reason = esc.get("reason", "unknown")
        self._append_output(f"⚠ ESCALATION: {reason}", "warning")

    def _append_chat(self, speaker: str, msg: str, tag: str = "system") -> None:
        """Append one message to the AI summary panel."""
        if not msg:
            return
        self.ai_chat.configure(state=tk.NORMAL)
        self.ai_chat.insert(tk.END, f"{speaker}: {msg}\n\n", tag)
        self.ai_chat.configure(state=tk.DISABLED)
        self.ai_chat.see(tk.END)

    def _show_loading(self, visible: bool = True) -> None:
        """Show or hide the loading indicator."""
        if visible and not self.loading_visible:
            self.loading_frame.lift()
            self.loading_visible = True
        elif not visible and self.loading_visible:
            self.ai_chat.lift()
            self.loading_visible = False

    @staticmethod
    def _shorten_text(text: str, max_lines: int = 5) -> str:
        """Truncate text to max_lines for display in summary."""
        lines = str(text).split('\n')
        if len(lines) > max_lines:
            return '\n'.join(lines[:max_lines]) + f'\n... ({len(lines) - max_lines} more lines)'
        return text

    @staticmethod
    def _format_agent_message(output: Any) -> str:
        """Convert agent output dictionaries into compact chat text."""
        if isinstance(output, dict):
            if "content" in output:
                return str(output["content"])
            if "summary" in output:
                pieces = [str(output["summary"])]
                files = output.get("files_produced")
                if files:
                    pieces.append("Files: " + ", ".join(str(file) for file in files))
                return "\n".join(pieces)
            if "tasks_created" in output:
                milestones = output.get("milestones") or []
                return f"Created {output['tasks_created']} tasks. Milestones: {', '.join(map(str, milestones[:5]))}"
            return str(output)
        return str(output)

    def _append_output(self, msg: str, tag: str = "debug") -> None:
        """Append message to output log with optional tag."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {msg}\n"
        
        self.raw_output.insert(tk.END, log_line, tag)
        self.raw_output.see(tk.END)
        self.raw_output.update_idletasks()

    def _update_summary_label(self) -> None:
        files_label = ", ".join(self.file_changes) if self.file_changes else "none"
        self.ai_summary_var.set(f"Files: {files_label} | Iterations: {self.iteration_count} | Tests: {self.tests_status}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="gui_tk.py", description="Tkinter GUI for Ollama Fleet")
    parser.add_argument("--demo", action="store_true", help="Use demo executor (no Ollama server required)")
    parser.add_argument("--goal", default="Demo GUI", help="Initial goal for the job")
    parser.add_argument("--db-path", default="ollama_fleet.db", help="SQLite database path")
    args = parser.parse_args()

    q: "queue.Queue[dict[str, Any]]" = queue.Queue()
    app = FleetTkApp(q, goal=args.goal, use_demo=args.demo, db_path=args.db_path)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
