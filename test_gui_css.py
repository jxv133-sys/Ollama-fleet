#!/usr/bin/env python3
"""Quick test to verify GUI CSS parses without errors."""

import asyncio
from pathlib import Path
from ollama_fleet.config import FleetSettings
from ollama_fleet.db.database import Database
from ollama_fleet.ui.event_bus import UIEventBus
from ollama_fleet.ui.dashboard import OllamaFleetDashboard
from ollama_fleet.orchestrator.orchestrator import Orchestrator
from scripts.demo_run import DummyExecutor


async def test_gui_css():
    """Test that GUI CSS parses without errors."""
    print("Testing GUI CSS parsing...")
    
    db_path = Path("test_gui_css.db")
    if db_path.exists():
        db_path.unlink()

    try:
        settings = FleetSettings()
        sd = settings.model_dump()
        sd["workspace"]["base_path"] = "test_gui_css_workspaces"
        settings = FleetSettings.model_validate(sd)

        db = Database(db_path)
        await db.connect()

        ui_bus = UIEventBus()
        orchestrator = Orchestrator(db, settings, ui_bus=ui_bus)
        orchestrator.executor = DummyExecutor(settings)

        # Create dashboard - this is where CSS parsing happens
        dashboard = OllamaFleetDashboard(ui_bus, orchestrator=orchestrator, goal="Test", config={})
        
        print("✓ GUI CSS parsing: SUCCESS")
        print("✓ Dashboard created without CSS errors")
        
        await db.close()
        if db_path.exists():
            db_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ CSS Parsing Error: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_gui_css())
    exit(0 if result else 1)
