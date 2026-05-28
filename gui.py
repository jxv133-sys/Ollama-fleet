#!/usr/bin/env python3
"""Simple GUI entrypoint for Ollama Fleet.

Run with:

    python3 gui.py            # starts dashboard with demo executor
    python3 gui.py --demo     # explicit demo mode
    python3 gui.py --goal "My job"  # custom goal

This script connects the database, instantiates the orchestrator, and
launches the Textual dashboard in the main thread.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

from ollama_fleet.config import load_settings
from ollama_fleet.db.database import Database
from ollama_fleet.ui.event_bus import UIEventBus
from ollama_fleet.orchestrator.orchestrator import Orchestrator
from ollama_fleet.ui.dashboard import OllamaFleetDashboard


def _sync_connect(db: Database) -> None:
    asyncio.run(db.connect())


def _sync_close(db: Database) -> None:
    asyncio.run(db.close())


def main() -> int:
    parser = argparse.ArgumentParser(prog="gui.py")
    parser.add_argument("--demo", action="store_true", help="Use the demo executor (no Ollama server)")
    parser.add_argument("--goal", default="Demo GUI", help="High level goal for the job")
    parser.add_argument("--db-path", default="ollama_fleet.db", help="SQLite DB path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    settings = load_settings()
    db = Database(Path(args.db_path))

    # connect database before launching UI
    _sync_connect(db)

    ui_bus = UIEventBus()
    orchestrator = Orchestrator(db, settings, ui_bus=ui_bus)

    if args.demo:
        try:
            from scripts.demo_run import DummyExecutor

            orchestrator.executor = DummyExecutor(settings)
        except Exception:
            logging.exception("Failed to initialize DummyExecutor")

    dashboard = OllamaFleetDashboard(ui_bus, orchestrator=orchestrator, goal=args.goal, config={"source": "gui"})

    try:
        # Run the Textual app in the main thread — it blocks until the UI exits.
        dashboard.run()
    finally:
        _sync_close(db)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
