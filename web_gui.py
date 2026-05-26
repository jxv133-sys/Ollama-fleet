#!/usr/bin/env python3
"""Web GUI for Ollama Fleet — served on http://localhost:1020

Run:
    python3 web_gui.py
    python3 web_gui.py --demo
    python3 web_gui.py --goal "Create REST API"
    python3 web_gui.py --db-path /path/to/ollama_fleet.db

The FastAPI server and the orchestrator share the same asyncio event loop.
Events are broadcast to all connected browser clients via WebSocket.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ollama_fleet.config import load_settings
from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Broadcast orchestrator events to all connected browser clients."""

    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws) if hasattr(self._clients, "discard") else None
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, event: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        msg = json.dumps(event)
        for ws in list(self._clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ---------------------------------------------------------------------------
# Event bus adapter — bridges orchestrator events → WebSocket broadcast
# ---------------------------------------------------------------------------

class WebEventBus:
    """Adapter that publishes orchestrator events into the WebSocket manager."""

    def __init__(self, manager: ConnectionManager, loop: asyncio.AbstractEventLoop) -> None:
        self._manager = manager
        self._loop = loop

    def publish(self, event: dict[str, Any]) -> None:
        """Called synchronously by the orchestrator; schedules async broadcast."""
        try:
            asyncio.run_coroutine_threadsafe(
                self._manager.broadcast(event), self._loop
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, db_path: Path, use_demo: bool) -> None:
        self.db_path = db_path
        self.use_demo = use_demo
        self.db: Database | None = None
        self.orchestrator: Orchestrator | None = None
        self.manager = ConnectionManager()
        self.job_running = False
        self.start_time: float | None = None
        self._job_task: asyncio.Task | None = None

    async def startup(self) -> None:
        settings = load_settings()
        self.db = Database(self.db_path)
        await self.db.connect()
        loop = asyncio.get_event_loop()
        bus = WebEventBus(self.manager, loop)
        self.orchestrator = Orchestrator(self.db, settings, ui_bus=bus)
        if self.use_demo:
            try:
                from scripts.demo_run import DummyExecutor
                self.orchestrator.executor = DummyExecutor(settings)
            except Exception as exc:
                logger.warning("DummyExecutor not available: %s", exc)

    async def shutdown(self) -> None:
        if self._job_task and not self._job_task.done():
            self._job_task.cancel()
        if self.db:
            await self.db.close()

    async def submit_job(self, goal: str, model_overrides: dict[str, str], base_url: str) -> str:
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not initialised")
        settings = self.orchestrator.settings
        # Apply model overrides from the UI
        if model_overrides.get("planner"):
            settings.ollama.planner_model = model_overrides["planner"]
        if model_overrides.get("coder"):
            settings.ollama.coder_model = model_overrides["coder"]
        if model_overrides.get("critic"):
            settings.ollama.critic_model = model_overrides["critic"]
        if model_overrides.get("tester"):
            settings.ollama.tester_model = model_overrides["tester"]
        if model_overrides.get("synthesizer"):
            settings.ollama.summarizer_model = model_overrides["synthesizer"]
        if base_url:
            settings.ollama.base_url = base_url.removesuffix("/v1/models").rstrip("/")
            self.orchestrator.executor.client.base_url = settings.ollama.base_url
        self.job_running = True
        self.start_time = time.time()
        job_id = await self.orchestrator.submit_job(goal=goal, config={"source": "web_gui"})
        self.job_running = False
        return job_id


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="Ollama Fleet Web UI")

    @app.on_event("startup")
    async def _startup() -> None:
        await state.startup()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await state.shutdown()

    # -----------------------------------------------------------------------
    # WebSocket — real-time event stream
    # -----------------------------------------------------------------------
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await state.manager.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keep connection alive; client sends pings
        except WebSocketDisconnect:
            state.manager.disconnect(ws)

    # -----------------------------------------------------------------------
    # REST — job submission
    # -----------------------------------------------------------------------
    @app.post("/api/jobs")
    async def submit_job(body: dict[str, Any]) -> dict[str, Any]:
        if state.job_running:
            return {"error": "A job is already running"}
        goal = (body.get("goal") or "").strip()
        if not goal:
            return {"error": "goal is required"}
        model_overrides: dict[str, str] = body.get("models", {})
        base_url: str = body.get("base_url", "")

        async def _run() -> None:
            try:
                job_id = await state.submit_job(goal, model_overrides, base_url)
                await state.manager.broadcast({
                    "type": "agent_log",
                    "message": f"Job finished: {job_id}",
                    "level": "info",
                })
            except Exception as exc:
                import traceback
                await state.manager.broadcast({
                    "type": "agent_log",
                    "message": f"Orchestrator error: {exc}",
                    "level": "error",
                })
                await state.manager.broadcast({
                    "type": "agent_log",
                    "message": traceback.format_exc(),
                    "level": "error",
                })
                await state.manager.broadcast({
                    "type": "job_state_changed",
                    "job_id": "",
                    "new_state": "failed",
                })
                state.job_running = False

        state._job_task = asyncio.create_task(_run())
        return {"status": "submitted", "goal": goal}

    # -----------------------------------------------------------------------
    # REST — job history
    # -----------------------------------------------------------------------
    @app.get("/api/jobs")
    async def list_jobs() -> list[dict[str, Any]]:
        try:
            with sqlite3.connect(state.db_path, timeout=2.0) as conn:
                rows = conn.execute(
                    "SELECT job_id, goal, state, created_at, updated_at "
                    "FROM jobs ORDER BY updated_at DESC, created_at DESC LIMIT 100"
                ).fetchall()
            return [
                {"job_id": r[0], "goal": r[1], "state": r[2],
                 "created_at": r[3], "updated_at": r[4]}
                for r in rows
            ]
        except sqlite3.Error:
            return []

    @app.get("/api/jobs/{job_id}/tasks")
    async def list_tasks(job_id: str) -> list[dict[str, Any]]:
        try:
            with sqlite3.connect(state.db_path, timeout=2.0) as conn:
                rows = conn.execute(
                    "SELECT task_id, agent_type, state, created_at "
                    "FROM tasks WHERE job_id = ? ORDER BY created_at ASC",
                    (job_id,),
                ).fetchall()
            return [
                {"task_id": r[0], "agent_type": r[1], "state": r[2], "created_at": r[3]}
                for r in rows
            ]
        except sqlite3.Error:
            return []


    # -----------------------------------------------------------------------
    # REST — model list (proxies to Ollama server)
    # -----------------------------------------------------------------------
    @app.get("/api/models")
    async def list_models(base_url: str = "") -> dict[str, Any]:
        url = (base_url or (state.orchestrator.settings.ollama.base_url if state.orchestrator else "")).rstrip("/")
        endpoint = f"{url}/v1/models"
        try:
            req = Request(endpoint, headers={"Accept": "application/json"})
            with urlopen(req, timeout=10.0) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw)
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                ids = [str(item["id"]) for item in payload["data"] if isinstance(item, dict) and item.get("id")]
            elif isinstance(payload, dict) and isinstance(payload.get("models"), list):
                ids = [str(item.get("name") or item.get("id")) for item in payload["models"] if isinstance(item, dict)]
            elif isinstance(payload, list):
                ids = [str(item.get("id") or item.get("name") if isinstance(item, dict) else item) for item in payload]
            else:
                ids = []
            return {"models": sorted({m for m in ids if m})}
        except (HTTPError, URLError, Exception) as exc:
            return {"models": [], "error": str(exc)}

    # -----------------------------------------------------------------------
    # REST — config defaults (used by the UI on first load)
    # -----------------------------------------------------------------------
    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        if state.orchestrator is None:
            return {}
        s = state.orchestrator.settings
        return {
            "base_url": s.ollama.base_url,
            "models": {
                "planner": s.ollama.planner_model,
                "coder": s.ollama.coder_model,
                "critic": s.ollama.critic_model or s.ollama.coder_model,
                "tester": s.ollama.tester_model or s.ollama.coder_model,
                "synthesizer": s.ollama.summarizer_model,
            },
        }

    # -----------------------------------------------------------------------
    # REST — file browser: list workspace files for a job
    # -----------------------------------------------------------------------
    @app.get("/api/jobs/{job_id}/files")
    async def list_workspace_files(job_id: str) -> dict[str, Any]:
        try:
            with sqlite3.connect(state.db_path, timeout=2.0) as conn:
                row = conn.execute(
                    "SELECT workspace_path FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
        except sqlite3.Error:
            return {"error": "DB error", "files": []}
        if not row:
            return {"error": "Job not found", "files": []}
        ws_root = Path(row[0])
        if not ws_root.exists():
            return {"error": "Workspace not found", "files": []}
        files = []
        for p in sorted(ws_root.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(ws_root))
                files.append({
                    "path": rel,
                    "size": p.stat().st_size,
                    "is_source": p.suffix in (".py", ".js", ".ts", ".json", ".toml", ".md", ".txt", ".yaml", ".yml"),
                })
        return {"workspace": str(ws_root), "files": files}

    @app.get("/api/jobs/{job_id}/files/{file_path:path}")
    async def read_workspace_file(job_id: str, file_path: str) -> dict[str, Any]:
        try:
            with sqlite3.connect(state.db_path, timeout=2.0) as conn:
                row = conn.execute(
                    "SELECT workspace_path FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
        except sqlite3.Error:
            return {"error": "DB error"}
        if not row:
            return {"error": "Job not found"}
        ws_root = Path(row[0])
        target = (ws_root / file_path).resolve()
        # Path traversal guard
        if ws_root.resolve() not in target.parents and target != ws_root.resolve():
            return {"error": "Access denied"}
        if not target.exists() or not target.is_file():
            return {"error": "File not found"}
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return {"error": str(exc)}
        return {"path": file_path, "content": content, "size": target.stat().st_size}

    # -----------------------------------------------------------------------
    # Serve the single-page HTML UI
    # -----------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(content=_HTML)

    return app


# ---------------------------------------------------------------------------
# Single-page HTML/CSS/JS UI (inlined — no external dependencies)
# ---------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ollama Fleet</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --orange: #d29922;
    --planner: #58a6ff; --coder: #3fb950; --critic: #d29922;
    --tester: #f85149; --synthesizer: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 10px 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  header h1 { font-size: 15px; font-weight: 600; color: var(--accent); white-space: nowrap; }
  #goal-input { flex: 1; min-width: 200px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 6px 10px; font-size: 13px; }
  #goal-input:focus { outline: none; border-color: var(--accent); }
  button { border: none; border-radius: 6px; padding: 6px 14px; font-size: 13px; cursor: pointer; font-weight: 500; }
  #start-btn { background: var(--green); color: #000; }
  #start-btn:disabled { background: #2d4a35; color: var(--muted); cursor: not-allowed; }
  #stop-btn { background: var(--red); color: #fff; }
  #stop-btn:disabled { background: #4a1f1f; color: var(--muted); cursor: not-allowed; }
  #elapsed { color: var(--muted); font-size: 12px; white-space: nowrap; }
  #status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); flex-shrink: 0; }
  #status-dot.running { background: var(--green); animation: pulse 1.2s infinite; }
  #status-dot.failed { background: var(--red); }
  #status-dot.completed { background: var(--green); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .main { display: grid; grid-template-columns: 340px 1fr; grid-template-rows: 1fr; flex: 1; overflow: hidden; gap: 0; }
  .panel { background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
  .panel-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); padding: 8px 12px 6px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
  .right-col { display: flex; flex-direction: column; overflow: hidden; }
  /* Models row */
  #models-bar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 6px 12px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; flex-shrink: 0; }
  #models-bar label { color: var(--muted); font-size: 11px; }
  #models-bar select, #models-bar input[type=text] { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text); padding: 3px 6px; font-size: 11px; }
  #models-bar select { max-width: 220px; }
  #load-models-btn { background: var(--border); color: var(--text); padding: 3px 10px; font-size: 11px; }
  #models-status { color: var(--muted); font-size: 11px; }
  /* Agent status */
  .agent-row { display: flex; align-items: center; gap: 8px; padding: 5px 12px; border-bottom: 1px solid #1c2128; }
  .agent-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); flex-shrink: 0; }
  .agent-dot.running { background: var(--orange); animation: pulse 1s infinite; }
  .agent-dot.completed { background: var(--green); }
  .agent-dot.failed { background: var(--red); }
  .agent-name { width: 80px; font-weight: 500; }
  .agent-state { color: var(--muted); font-size: 11px; }
  /* Jobs table */
  #jobs-list { overflow-y: auto; flex: 1; }
  .job-row { display: grid; grid-template-columns: 70px 70px 1fr 60px; gap: 4px; padding: 5px 12px; border-bottom: 1px solid #1c2128; cursor: pointer; font-size: 11px; }
  .job-row:hover { background: #1c2128; }
  .job-row.selected { background: #1f2d3d; }
  .job-row .state { font-weight: 600; }
  .state-completed { color: var(--green); }
  .state-failed { color: var(--red); }
  .state-running, .state-submitted { color: var(--orange); }
  /* Tasks table */
  #tasks-section { flex-shrink: 0; max-height: 200px; border-bottom: 1px solid var(--border); overflow-y: auto; }
  .task-row { display: grid; grid-template-columns: 1fr 80px 80px; gap: 4px; padding: 4px 12px; border-bottom: 1px solid #1c2128; font-size: 11px; font-family: monospace; }
  /* Log */
  #log { flex: 1; overflow-y: auto; padding: 8px 12px; font-family: "SF Mono", "Fira Code", monospace; font-size: 11px; line-height: 1.6; }
  .log-line { white-space: pre-wrap; word-break: break-all; }
  .log-info { color: var(--accent); }
  .log-success { color: var(--green); }
  .log-warning { color: var(--orange); }
  .log-error { color: var(--red); }
  .log-debug { color: var(--muted); }
  .log-planner { color: var(--planner); }
  .log-coder { color: var(--coder); }
  .log-critic { color: var(--critic); }
  .log-tester { color: var(--tester); }
  .log-synthesizer { color: var(--synthesizer); }
  /* Chat */
  #chat { overflow-y: auto; flex: 1; padding: 8px 12px; }
  .chat-msg { margin-bottom: 8px; }
  .chat-speaker { font-weight: 600; font-size: 11px; }
  .chat-text { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .speaker-planner { color: var(--planner); }
  .speaker-coder { color: var(--coder); }
  .speaker-critic { color: var(--critic); }
  .speaker-tester { color: var(--tester); }
  .speaker-synthesizer { color: var(--synthesizer); }
  .speaker-system { color: var(--muted); }
  .speaker-error { color: var(--red); }
  footer { background: var(--surface); border-top: 1px solid var(--border); padding: 4px 16px; font-size: 11px; color: var(--muted); display: flex; gap: 16px; }
  /* File browser */
  .file-entry { display: flex; align-items: center; gap: 6px; padding: 3px 12px; cursor: pointer; font-size: 11px; font-family: monospace; border-bottom: 1px solid #1c2128; }
  .file-entry:hover { background: #1c2128; }
  .file-icon { color: var(--muted); flex-shrink: 0; }
  .file-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-size { color: var(--muted); font-size: 10px; flex-shrink: 0; }
  /* File viewer modal */
  #file-modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.7); z-index: 100; align-items: center; justify-content: center; }
  #file-modal.open { display: flex; }
  #file-modal-inner { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; width: 80vw; max-height: 80vh; display: flex; flex-direction: column; overflow: hidden; }
  #file-modal-header { display: flex; align-items: center; justify-content: space-between; padding: 10px 16px; border-bottom: 1px solid var(--border); }
  #file-modal-title { font-size: 12px; font-family: monospace; color: var(--accent); }
  #file-modal-close { background: none; border: none; color: var(--muted); font-size: 18px; cursor: pointer; padding: 0 4px; }
  #file-modal-close:hover { color: var(--text); }
  #file-modal-content { overflow: auto; padding: 12px 16px; font-family: "SF Mono","Fira Code",monospace; font-size: 11px; line-height: 1.6; white-space: pre; flex: 1; }
</style>
</head>
<body>
"""
_HTML += r"""
<header>
  <h1>⚡ Ollama Fleet</h1>
  <input id="goal-input" type="text" placeholder="Enter project goal…" value="">
  <button id="start-btn">▶ Start Job</button>
  <button id="stop-btn" disabled>⏹ Stop</button>
  <span id="status-dot"></span>
  <span id="elapsed">0s</span>
</header>

<div id="models-bar">
  <label>Ollama URL:</label>
  <input type="text" id="base-url-input" style="width:200px" value="">
  <button id="load-models-btn">Load Models</button>
  <span id="models-status">Models not loaded</span>
  <label>Planner:</label><select id="sel-planner"><option>—</option></select>
  <label>Coder:</label><select id="sel-coder"><option>—</option></select>
  <label>Critic:</label><select id="sel-critic"><option>—</option></select>
  <label>Tester:</label><select id="sel-tester"><option>—</option></select>
  <label>Synth:</label><select id="sel-synthesizer"><option>—</option></select>
</div>

<div class="main">
  <!-- Left panel -->
  <div class="panel">
    <div class="panel-title">AI Chat</div>
    <div id="chat"></div>

    <div class="panel-title" style="margin-top:4px">Agent Status</div>
    <div id="agents">
      <div class="agent-row" id="agent-planner">
        <div class="agent-dot" id="dot-planner"></div>
        <span class="agent-name">Planner</span>
        <span class="agent-state" id="state-planner">Idle</span>
      </div>
      <div class="agent-row" id="agent-coder">
        <div class="agent-dot" id="dot-coder"></div>
        <span class="agent-name">Coder</span>
        <span class="agent-state" id="state-coder">Idle</span>
      </div>
      <div class="agent-row" id="agent-critic">
        <div class="agent-dot" id="dot-critic"></div>
        <span class="agent-name">Critic</span>
        <span class="agent-state" id="state-critic">Idle</span>
      </div>
      <div class="agent-row" id="agent-tester">
        <div class="agent-dot" id="dot-tester"></div>
        <span class="agent-name">Tester</span>
        <span class="agent-state" id="state-tester">Idle</span>
      </div>
      <div class="agent-row" id="agent-synthesizer">
        <div class="agent-dot" id="dot-synthesizer"></div>
        <span class="agent-name">Synthesizer</span>
        <span class="agent-state" id="state-synthesizer">Idle</span>
      </div>
    </div>

    <div class="panel-title" style="margin-top:4px">Jobs</div>
    <div style="display:grid;grid-template-columns:70px 70px 1fr 60px;gap:4px;padding:4px 12px;font-size:10px;color:var(--muted);border-bottom:1px solid var(--border)">
      <span>ID</span><span>State</span><span>Goal</span><span>Time</span>
    </div>
    <div id="jobs-list"></div>

    <div class="panel-title" style="margin-top:4px">File Browser</div>
    <div id="file-browser" style="overflow-y:auto;flex:1;min-height:80px">
      <div id="file-list" style="padding:4px 0"></div>
    </div>
  </div>

  <!-- Right column -->
  <div class="right-col">
    <div class="panel-title" style="padding:8px 12px 6px;border-bottom:1px solid var(--border);background:var(--surface)">Task Progress</div>
    <div id="tasks-section">
      <div style="display:grid;grid-template-columns:1fr 80px 80px;gap:4px;padding:4px 12px;font-size:10px;color:var(--muted);border-bottom:1px solid var(--border)">
        <span>Task ID</span><span>Agent</span><span>State</span>
      </div>
      <div id="tasks-list"></div>
    </div>
    <div class="panel-title" style="padding:8px 12px 6px;border-bottom:1px solid var(--border);background:var(--surface)">Output Log</div>
    <div id="log"></div>
  </div>
</div>

<footer>
  <span id="footer-status">Ready</span>
  <span id="footer-job">No active job</span>
</footer>

<div id="file-modal">
  <div id="file-modal-inner">
    <div id="file-modal-header">
      <span id="file-modal-title"></span>
      <button id="file-modal-close">✕</button>
    </div>
    <div id="file-modal-content"></div>
  </div>
</div>
"""
_HTML += r"""
<script>
// ── State ──────────────────────────────────────────────────────────────────
const state = {
  jobRunning: false,
  startTime: null,
  elapsedTimer: null,
  tasks: {},       // task_id -> {agent_type, state}
  agents: {},      // agent_name -> {state}
  selectedJobId: null,
  availableModels: [],
};

// ── WebSocket ──────────────────────────────────────────────────────────────
let ws;
function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (e) => handleEvent(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connectWS, 2000);
  ws.onerror = () => ws.close();
  // keep-alive ping every 20s
  setInterval(() => { if (ws.readyState === 1) ws.send("ping"); }, 20000);
}
connectWS();

// ── Event handler ──────────────────────────────────────────────────────────
function handleEvent(ev) {
  const t = ev.type;
  if (t === "job_state_changed")   onJobStateChanged(ev);
  else if (t === "agent_log")      onAgentLog(ev);
  else if (t === "agent_output")   onAgentOutput(ev);
  else if (t === "task_state_changed") onTaskStateChanged(ev);
  else if (t === "file_written") {
    appendLog(`File created: ${ev.path}`, "success");
    if (state.selectedJobId) refreshFileBrowser(state.selectedJobId);
  }
  else if (t === "validation_result") onValidation(ev);
  else if (t === "escalation_added")  appendLog(`⚠ ESCALATION: ${ev.escalation?.reason}`, "warning");
}

function onJobStateChanged(ev) {
  const s = ev.new_state || "–";
  setFooter(`Job ${s}`, ev.job_id ? `Job: ${ev.job_id.slice(0,8)}` : "");
  setStatusDot(s);
  if (["completed","failed","stopped"].includes(s)) {
    setJobRunning(false);
  }
  appendLog(`Job ${s.toUpperCase()}`, "info");
  appendChat("System", `Job ${s}`, s === "failed" ? "error" : "system");
  refreshJobs();
  // Auto-select the active job so the file browser tracks it
  if (ev.job_id) {
    state.selectedJobId = ev.job_id;
    if (["completed","failed"].includes(s)) refreshFileBrowser(ev.job_id);
  }
}

function onAgentLog(ev) {
  const msg = ev.message || "";
  const level = ev.level || "debug";
  appendLog(msg, level);
  const ml = msg.toLowerCase();
  if (ml.includes("planner") && ml.includes("created")) {
    setAgent("planner", "completed");
    appendChat("Planner", msg, "planner");
  }
}

function onAgentOutput(ev) {
  const agent = (ev.agent_type || "?").toLowerCase();
  const out = ev.output;
  const text = typeof out === "object" ? JSON.stringify(out) : String(out);
  appendLog(`[${agent.toUpperCase()}] ${text}`, agent);
  appendChat(cap(agent), formatAgentMsg(out), agent);
  if (["planner","coder","critic","tester","synthesizer"].includes(agent)) {
    setAgent(agent, "completed");
  }
}

function onTaskStateChanged(ev) {
  const tid = ev.task_id || "?";
  const ns = ev.new_state || "–";
  const agent = (ev.agent_type || "?").toLowerCase();
  state.tasks[tid] = { agent_type: agent, state: ns };
  renderTasks();
  if (["planner","coder","critic","tester","synthesizer"].includes(agent)) {
    setAgent(agent, ns);
    if (ns === "failed" && ev.reason) appendChat(cap(agent), `Failed: ${ev.reason}`, "error");
  }
  appendLog(`Task ${tid.slice(-12)}: ${ns} (${agent})`, "debug");
}

function onValidation(ev) {
  const r = ev.validation_result || {};
  const ok = r.syntax_ok;
  appendLog(`Validation ${ok ? "✓ syntax ok" : "✗ syntax error"}`, ok ? "success" : "warning");
}

// ── Agent status helpers ───────────────────────────────────────────────────
function setAgent(name, st) {
  state.agents[name] = st;
  const dot = document.getElementById(`dot-${name}`);
  const lbl = document.getElementById(`state-${name}`);
  if (!dot || !lbl) return;
  dot.className = "agent-dot " + st;
  const icons = { running: "● Running", completed: "✓ Completed", failed: "✗ Failed" };
  lbl.textContent = icons[st] || cap(st);
}

// ── Task table ─────────────────────────────────────────────────────────────
function renderTasks() {
  const el = document.getElementById("tasks-list");
  el.innerHTML = "";
  for (const [tid, info] of Object.entries(state.tasks)) {
    const row = document.createElement("div");
    row.className = "task-row";
    row.innerHTML = `<span title="${tid}">${tid.slice(-20)}</span><span>${info.agent_type}</span><span class="state-${info.state}">${info.state}</span>`;
    el.appendChild(row);
  }
}

// ── Jobs list ──────────────────────────────────────────────────────────────
async function refreshJobs() {
  const jobs = await fetch("/api/jobs").then(r => r.json()).catch(() => []);
  const el = document.getElementById("jobs-list");
  el.innerHTML = "";
  for (const j of jobs) {
    const row = document.createElement("div");
    row.className = "job-row" + (j.job_id === state.selectedJobId ? " selected" : "");
    const ts = j.updated_at ? j.updated_at.slice(11,19) : "–";
    row.innerHTML = `<span title="${j.job_id}">${j.job_id.slice(0,8)}</span><span class="state state-${j.state}">${j.state}</span><span title="${j.goal}">${j.goal.slice(0,30)}</span><span>${ts}</span>`;
    row.onclick = () => selectJob(j.job_id);
    el.appendChild(row);
  }
}

async function selectJob(jobId) {
  state.selectedJobId = jobId;
  refreshJobs();
  const tasks = await fetch(`/api/jobs/${jobId}/tasks`).then(r => r.json()).catch(() => []);
  state.tasks = {};
  for (const t of tasks) state.tasks[t.task_id] = { agent_type: t.agent_type, state: t.state };
  renderTasks();
  refreshFileBrowser(jobId);
}

// ── File browser ───────────────────────────────────────────────────────────
async function refreshFileBrowser(jobId) {
  const el = document.getElementById("file-list");
  el.innerHTML = '<div style="padding:4px 12px;color:var(--muted);font-size:11px">Loading…</div>';
  const data = await fetch(`/api/jobs/${jobId}/files`).then(r => r.json()).catch(() => ({ files: [], error: "fetch failed" }));
  el.innerHTML = "";
  if (data.error && !data.files?.length) {
    el.innerHTML = `<div style="padding:4px 12px;color:var(--muted);font-size:11px">${data.error}</div>`;
    return;
  }
  // Group by directory
  const byDir = {};
  for (const f of (data.files || [])) {
    const parts = f.path.split("/");
    const dir = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
    if (!byDir[dir]) byDir[dir] = [];
    byDir[dir].push(f);
  }
  for (const [dir, files] of Object.entries(byDir)) {
    if (dir) {
      const hdr = document.createElement("div");
      hdr.style.cssText = "padding:3px 12px;font-size:10px;color:var(--muted);font-weight:600;background:#1c2128;";
      hdr.textContent = "📁 " + dir;
      el.appendChild(hdr);
    }
    for (const f of files) {
      const row = document.createElement("div");
      row.className = "file-entry";
      const icon = f.is_source ? "📄" : "📦";
      const name = f.path.split("/").pop();
      const size = f.size < 1024 ? `${f.size}B` : `${(f.size/1024).toFixed(1)}K`;
      row.innerHTML = `<span class="file-icon">${icon}</span><span class="file-name" title="${f.path}">${name}</span><span class="file-size">${size}</span>`;
      if (f.is_source) {
        row.onclick = () => openFile(jobId, f.path);
      }
      el.appendChild(row);
    }
  }
  if (!data.files?.length) {
    el.innerHTML = '<div style="padding:4px 12px;color:var(--muted);font-size:11px">No files yet</div>';
  }
}

async function openFile(jobId, filePath) {
  const modal = document.getElementById("file-modal");
  const title = document.getElementById("file-modal-title");
  const content = document.getElementById("file-modal-content");
  title.textContent = filePath;
  content.textContent = "Loading…";
  modal.classList.add("open");
  const data = await fetch(`/api/jobs/${jobId}/files/${filePath}`).then(r => r.json()).catch(() => ({ error: "fetch failed" }));
  content.textContent = data.error ? `Error: ${data.error}` : (data.content || "(empty)");
}

document.getElementById("file-modal-close").onclick = () => {
  document.getElementById("file-modal").classList.remove("open");
};
document.getElementById("file-modal").onclick = (e) => {
  if (e.target === document.getElementById("file-modal"))
    document.getElementById("file-modal").classList.remove("open");
};

// ── Log helpers ────────────────────────────────────────────────────────────
function appendLog(msg, cls = "debug") {
  const el = document.getElementById("log");
  const line = document.createElement("div");
  const ts = new Date().toTimeString().slice(0,8);
  line.className = `log-line log-${cls}`;
  line.textContent = `[${ts}] ${msg}`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function appendChat(speaker, msg, cls = "system") {
  if (!msg) return;
  const el = document.getElementById("chat");
  const div = document.createElement("div");
  div.className = "chat-msg";
  const ts = new Date().toTimeString().slice(0,8);
  div.innerHTML = `<div class="chat-speaker speaker-${cls}">[${ts}] ${speaker}</div><div class="chat-text">${escHtml(String(msg))}</div>`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function formatAgentMsg(out) {
  if (typeof out !== "object" || out === null) return String(out);
  if (out.summary) {
    let s = out.summary;
    if (out.files_produced?.length) s += "\nFiles: " + out.files_produced.slice(0,5).join(", ");
    return s;
  }
  if (out.tasks_created !== undefined) {
    const ms = (out.milestones || []).slice(0,5).join(", ");
    return `Created ${out.tasks_created} tasks. Milestones: ${ms}`;
  }
  return JSON.stringify(out).slice(0,200);
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function cap(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }

// ── Status helpers ─────────────────────────────────────────────────────────
function setStatusDot(state_) {
  const dot = document.getElementById("status-dot");
  dot.className = "";
  if (["running","submitted"].includes(state_)) dot.classList.add("running");
  else if (state_ === "completed") dot.classList.add("completed");
  else if (state_ === "failed") dot.classList.add("failed");
}

function setFooter(status, job) {
  document.getElementById("footer-status").textContent = status;
  if (job) document.getElementById("footer-job").textContent = job;
}

function setJobRunning(running) {
  state.jobRunning = running;
  document.getElementById("start-btn").disabled = running;
  document.getElementById("stop-btn").disabled = !running;
  if (!running) {
    clearInterval(state.elapsedTimer);
    state.elapsedTimer = null;
  }
}

// ── Elapsed timer ──────────────────────────────────────────────────────────
function startElapsed() {
  state.startTime = Date.now();
  state.elapsedTimer = setInterval(() => {
    const s = Math.floor((Date.now() - state.startTime) / 1000);
    document.getElementById("elapsed").textContent = `${s}s`;
  }, 1000);
}

// ── Model loading ──────────────────────────────────────────────────────────
async function loadModels() {
  const baseUrl = document.getElementById("base-url-input").value.trim();
  document.getElementById("models-status").textContent = "Loading…";
  const data = await fetch(`/api/models?base_url=${encodeURIComponent(baseUrl)}`).then(r => r.json()).catch(() => ({ models: [], error: "fetch failed" }));
  if (data.error) {
    document.getElementById("models-status").textContent = `Error: ${data.error}`;
    return;
  }
  state.availableModels = data.models || [];
  document.getElementById("models-status").textContent = `${state.availableModels.length} models`;
  for (const id of ["sel-planner","sel-coder","sel-critic","sel-tester","sel-synthesizer"]) {
    const sel = document.getElementById(id);
    const cur = sel.value;
    sel.innerHTML = "";
    for (const m of state.availableModels) {
      const opt = document.createElement("option");
      opt.value = opt.textContent = m;
      if (m === cur) opt.selected = true;
      sel.appendChild(opt);
    }
  }
}

// ── Job submission ─────────────────────────────────────────────────────────
async function startJob() {
  if (state.jobRunning) return;
  const goal = document.getElementById("goal-input").value.trim();
  if (!goal) { appendLog("Goal cannot be empty", "error"); return; }
  const models = {
    planner: document.getElementById("sel-planner").value,
    coder: document.getElementById("sel-coder").value,
    critic: document.getElementById("sel-critic").value,
    tester: document.getElementById("sel-tester").value,
    synthesizer: document.getElementById("sel-synthesizer").value,
  };
  const base_url = document.getElementById("base-url-input").value.trim();
  setJobRunning(true);
  startElapsed();
  // Reset agent states
  for (const a of ["planner","coder","critic","tester","synthesizer"]) setAgent(a, "idle");
  state.tasks = {};
  renderTasks();
  document.getElementById("chat").innerHTML = "";
  appendChat("System", `New job submitted: ${goal}`, "system");
  setStatusDot("submitted");
  setFooter("Submitting…", "");
  const res = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal, models, base_url }),
  }).then(r => r.json()).catch(e => ({ error: String(e) }));
  if (res.error) {
    appendLog(`Submit error: ${res.error}`, "error");
    setJobRunning(false);
  }
}

// ── Wire up buttons ────────────────────────────────────────────────────────
document.getElementById("start-btn").onclick = startJob;
document.getElementById("stop-btn").onclick = () => {
  appendLog("Stop requested (current job will finish its current task)", "warning");
  setJobRunning(false);
};
document.getElementById("load-models-btn").onclick = loadModels;
document.getElementById("goal-input").addEventListener("keydown", e => { if (e.key === "Enter") startJob(); });

// ── Init ───────────────────────────────────────────────────────────────────
(async () => {
  // Fetch config defaults from server
  const cfg = await fetch("/api/config").then(r => r.json()).catch(() => ({}));
  if (cfg.base_url) document.getElementById("base-url-input").value = cfg.base_url;
  if (cfg.models) {
    for (const [role, model] of Object.entries(cfg.models)) {
      const sel = document.getElementById(`sel-${role}`);
      if (sel) {
        sel.innerHTML = `<option value="${model}">${model}</option>`;
      }
    }
  }
  refreshJobs();
  loadModels();
})();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="web_gui.py",
        description="Ollama Fleet — web UI on http://localhost:1020",
    )
    parser.add_argument("--demo", action="store_true", help="Use demo executor (no Ollama server required)")
    parser.add_argument("--goal", default="", help="Pre-fill the goal input")
    parser.add_argument("--db-path", default="ollama_fleet.db", help="SQLite database path")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=1020, help="Bind port (default: 1020)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    state = AppState(db_path=Path(args.db_path), use_demo=args.demo)
    app = create_app(state)

    print(f"\n  ⚡ Ollama Fleet web UI → http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
