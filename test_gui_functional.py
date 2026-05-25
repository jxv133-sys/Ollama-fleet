#!/usr/bin/env python3
"""Functional test for GUI with demo mode."""

import sys
import queue
import threading
import time

print("\n" + "=" * 80)
print("GUI FUNCTIONAL TEST (Demo Mode)")
print("=" * 80)

try:
    from gui_tk import FleetTkApp, TkEventBus
    from ollama_fleet.config import load_settings
    
    print("\n[1/4] Creating GUI with demo mode...")
    q = queue.Queue()
    # Use goal "Demo GUI" to avoid auto-start
    app = FleetTkApp(q, goal="Demo GUI", use_demo=True)
    print("    ✓ GUI created successfully")
    
    print("\n[2/4] Simulating orchestrator events...")
    
    # Simulate job state change
    events = [
        {
            "type": "job_state_changed",
            "job_id": "job-abc123",
            "new_state": "running",
            "goal": "Test calculation task"
        },
        {
            "type": "agent_log",
            "message": "[PLANNER] Analyzing task requirements",
            "level": "info"
        },
        {
            "type": "task_state_changed",
            "task_id": "task-001",
            "new_state": "running",
            "agent_type": "planner"
        },
        {
            "type": "agent_output",
            "agent_type": "planner",
            "output": "Created 5 tasks for job execution"
        },
        {
            "type": "task_state_changed",
            "task_id": "task-001",
            "new_state": "completed",
            "agent_type": "planner"
        },
        {
            "type": "file_written",
            "path": "output/plan.txt"
        },
        {
            "type": "job_state_changed",
            "job_id": "job-abc123",
            "new_state": "completed"
        }
    ]
    
    # Process events with small delays
    for i, event in enumerate(events):
        q.put_nowait(event)
        time.sleep(0.1)
        print(f"    ✓ Event {i+1}/{len(events)} queued: {event['type']}")
    
    print("\n[3/4] Processing queued events...")
    
    # Manually process all queued events
    event_count = 0
    while not q.empty():
        try:
            event = q.get_nowait()
            app._process_event(event)
            event_count += 1
        except queue.Empty:
            break
    
    print(f"    ✓ Processed {event_count} events from queue")
    
    # Verify GUI state
    print("\n[4/4] Verifying GUI state...")
    
    final_job_state = app.job_state_var.get()
    final_status = app.status_var.get()
    output_content = app.raw_output.get("1.0", "end-1c")
    
    assert final_job_state == "completed", f"Expected 'completed', got {final_job_state}"
    assert len(output_content) > 0, "Output log is empty"
    assert "PLANNER" in output_content, "Planner output not logged"
    
    print(f"    ✓ Job state: {final_job_state}")
    print(f"    ✓ Status: {final_status}")
    print(f"    ✓ Output log entries: {output_content.count(chr(10))}")
    
    # Cleanup
    try:
        app.quit()
    except:
        pass
    
    try:
        app.destroy()
    except:
        pass
    
    print("\n" + "=" * 80)
    print("✓ FUNCTIONAL TEST PASSED")
    print("=" * 80)
    print("\nThe GUI is fully functional and ready for use:")
    print("  python3 gui_tk.py --demo")
    print("  python3 gui_tk.py --goal 'Your task description'")
    print("=" * 80 + "\n")

except Exception as e:
    print(f"\n✗ FUNCTIONAL TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
