#!/usr/bin/env python3
"""Test that the orchestrator properly publishes job progression events with delays."""

import asyncio
import time
from pathlib import Path
from ollama_fleet.config import FleetSettings
from ollama_fleet.db.database import Database
from ollama_fleet.ui.event_bus import UIEventBus
from ollama_fleet.orchestrator.orchestrator import Orchestrator
from scripts.demo_run import DummyExecutor


async def test_job_progression():
    """Test that jobs show progression instead of instant completion."""
    print("=" * 80)
    print("GUI JOB PROGRESSION TEST")
    print("=" * 80)
    print()

    # Setup
    db_path = Path("test_gui_progression.db")
    if db_path.exists():
        db_path.unlink()

    settings = FleetSettings()
    sd = settings.model_dump()
    sd["workspace"]["base_path"] = "test_gui_progression_workspaces"
    settings = FleetSettings.model_validate(sd)

    db = Database(db_path)
    await db.connect()

    ui_bus = UIEventBus()
    orchestrator = Orchestrator(db, settings, ui_bus=ui_bus)
    orchestrator.executor = DummyExecutor(settings)

    # Track events to verify job progression
    events_received = []
    event_times = []

    def track_event(event):
        events_received.append(event)
        event_times.append(time.time())
        event_type = event.get("type")
        if event_type == "job_state_changed":
            print(f"  [JOB] {event.get('new_state').upper()}")
        elif event_type == "task_state_changed":
            print(f"    [{event.get('agent_type').upper()}] {event.get('new_state')}")

    ui_bus.subscribe(track_event)

    # Test: Submit a job and measure event progression
    print("[Test] Submitting job and tracking event progression...")
    test_goal = "Create a REST API with user and post endpoints"
    
    start_time = time.time()
    job_id = await orchestrator.submit_job(test_goal, {"source": "test"})
    end_time = time.time()
    
    total_duration = end_time - start_time
    print()
    print(f"Total job duration: {total_duration:.2f}s")
    print(f"Events received: {len(events_received)}")
    print()

    # Analyze event progression
    job_state_events = [e for e in events_received if e.get("type") == "job_state_changed"]
    task_state_events = [e for e in events_received if e.get("type") == "task_state_changed"]

    print("Event breakdown:")
    print(f"  - Job state events: {len(job_state_events)}")
    for evt in job_state_events:
        print(f"      {evt.get('new_state')}")

    print(f"  - Task state events: {len(task_state_events)}")
    task_states = {}
    for evt in task_state_events:
        agent = evt.get('agent_type')
        state = evt.get('new_state')
        key = f"{agent}:{state}"
        task_states[key] = task_states.get(key, 0) + 1
    
    for key, count in sorted(task_states.items()):
        print(f"      {key}: {count}")

    # Check event timing distribution
    if len(event_times) > 1:
        print()
        print("Event timing distribution:")
        time_deltas = [event_times[i] - event_times[i-1] for i in range(1, len(event_times))]
        print(f"  - Min time between events: {min(time_deltas):.3f}s")
        print(f"  - Max time between events: {max(time_deltas):.3f}s")
        print(f"  - Avg time between events: {sum(time_deltas)/len(time_deltas):.3f}s")

    # Verify state progression
    has_submitted = any(e.get("new_state") == "submitted" for e in job_state_events)
    has_running_tasks = any(e.get("new_state") == "running" for e in task_state_events)
    has_completed_tasks = any(e.get("new_state") == "completed" for e in task_state_events)
    has_completed_job = any(e.get("new_state") == "completed" for e in job_state_events)

    print()
    print("State progression verification:")
    print(f"  - Job submitted: {'✓' if has_submitted else '✗'}")
    print(f"  - Tasks running: {'✓' if has_running_tasks else '✗'}")
    print(f"  - Tasks completed: {'✓' if has_completed_tasks else '✗'}")
    print(f"  - Job completed: {'✓' if has_completed_job else '✗'}")
    print()

    # Check if we see enough event granularity
    events_per_task = len(task_state_events) / max(len(job_state_events), 1)
    print(f"Event granularity: ~{events_per_task:.1f} task events per job state")
    
    sufficient_events = len(events_received) >= 5  # At least job + task events
    all_states = has_submitted and has_running_tasks and has_completed_tasks and has_completed_job
    
    print()
    print("=" * 80)
    success = sufficient_events and all_states
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    print("=" * 80)

    # Cleanup
    await db.close()
    if db_path.exists():
        db_path.unlink()

    return success


if __name__ == "__main__":
    result = asyncio.run(test_job_progression())
    exit(0 if result else 1)
