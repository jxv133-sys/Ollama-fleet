"""CLI entry point for Ollama Fleet."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
from pathlib import Path
from typing import Any

from ollama_fleet.config import FleetSettings, load_settings
from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.orchestrator import Orchestrator
from ollama_fleet.ui.event_bus import UIEventBus


DEFAULT_DB_PATH = "ollama_fleet.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ollama-fleet")
    parser.add_argument(
        "--goal",
        required=True,
        help="The high-level goal for the Fleet orchestrator to execute.",
    )
    parser.add_argument(
        "--config",
        help="Path to the TOML configuration file.",
        default=None,
    )
    parser.add_argument(
        "--db-path",
        help="Path to the SQLite database file.",
        default=DEFAULT_DB_PATH,
    )
    parser.add_argument(
        "--workspace-base",
        help="Base directory for generated workspaces.",
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the textual UI alongside the orchestrator.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a local demo using the bundled DummyExecutor (no Ollama server).",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    if args.config is not None:
        settings = FleetSettings.from_toml(args.config, _from_env=True)
    else:
        settings = load_settings()

    if args.workspace_base is not None:
        settings = settings.model_copy(update={"workspace": {"base_path": args.workspace_base}})

    db = Database(Path(args.db_path))
    await db.connect()

    ui_bus = UIEventBus()

    # Optionally start the Textual UI in a daemon thread so the async loop can run the orchestrator.
    if args.ui:
        try:
            from ollama_fleet.ui.dashboard import OllamaFleetDashboard

            dashboard = OllamaFleetDashboard(ui_bus)
            ui_thread = threading.Thread(target=dashboard.run_blocking, daemon=True)
            ui_thread.start()
        except Exception:
            logging.exception("Failed to start textual UI; continuing without GUI")

    orchestrator = Orchestrator(db, settings, ui_bus=ui_bus)

    # If demo flag set, swap in the DummyExecutor (imported from scripts.demo_run)
    if args.demo:
        try:
            from scripts.demo_run import DummyExecutor

            orchestrator.executor = DummyExecutor(settings)
        except Exception:
            logging.exception("Failed to initialize demo executor; continuing with configured executor")

    try:
        job_id = await orchestrator.submit_job(
            goal=args.goal,
            config={"source": "cli", "workspace_base": args.workspace_base},
        )
        logging.info("Job submitted: %s", job_id)
        print(job_id)
        return 0
    except Exception as exc:
        logging.exception("Failed to submit job")
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        await db.close()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args))
