#!/usr/bin/env python3
"""Test GUI components without displaying the window."""

import sys
import queue
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

print("\n" + "=" * 80)
print("GUI COMPONENT TEST")
print("=" * 80)

# Test 1: Import GUI modules
print("\n[1/5] Testing GUI imports...")
try:
    from gui_tk import TkEventBus, FleetTkApp
    print("    ✓ gui_tk imports successfully")
except Exception as e:
    print(f"    ✗ gui_tk import failed: {e}")
    sys.exit(1)

try:
    import gui
    print("    ✓ gui imports successfully")
except Exception as e:
    print(f"    ✗ gui import failed: {e}")
    sys.exit(1)

# Test 2: Test TkEventBus
print("\n[2/5] Testing TkEventBus...")
try:
    q = queue.Queue()
    bus = TkEventBus(q)
    
    # Test publishing an event
    test_event = {"type": "test", "message": "hello"}
    bus.publish(test_event)
    
    # Verify event is in queue
    received = q.get_nowait()
    assert received == test_event, f"Event mismatch: {received}"
    print("    ✓ TkEventBus works correctly")
except Exception as e:
    print(f"    ✗ TkEventBus test failed: {e}")
    sys.exit(1)

# Test 3: Test FleetTkApp initialization
print("\n[3/5] Testing FleetTkApp initialization...")
try:
    q = queue.Queue()
    # Don't display the window, just test creation
    app = FleetTkApp(q, goal="Test Goal", use_demo=True)
    
    # Verify basic attributes
    assert app.goal == "Test Goal", "Goal not set correctly"
    assert app.use_demo == True, "Demo flag not set correctly"
    assert app.event_queue is q, "Event queue not set correctly"
    
    print("    ✓ FleetTkApp initializes successfully")
    
    # Close the app to prevent display
    app.quit()
    app.destroy()
except Exception as e:
    print(f"    ✗ FleetTkApp initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test event processing
print("\n[4/5] Testing event processing...")
try:
    q = queue.Queue()
    # Use goal "Demo GUI" to avoid auto-start, and use_demo=False
    app = FleetTkApp(q, goal="Demo GUI", use_demo=False)
    
    # Manually verify the initial job state
    initial_state = app.job_state_var.get()
    print(f"    Initial job state: {initial_state}")
    
    # Test job_state_changed event by calling handler directly
    app._handle_job_state_changed({
        "job_id": "test-123",
        "new_state": "running",
        "goal": "Test Goal"
    })
    
    # Verify state was updated
    job_state = app.job_state_var.get()
    job_id = app.job_id_var.get()
    
    # Clean up before checking assertions
    try:
        app.quit()
    except:
        pass
    
    try:
        app.destroy()
    except:
        pass
    
    # Now verify
    assert job_state == "running", f"Job state handler didn't update, got {job_state}"
    assert "test-123"[:12] in job_id, f"Job ID not updated correctly, got {job_id}"
    
    print("    ✓ Event processing works correctly")
except Exception as e:
    print(f"    ✗ Event processing test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test GUI configuration module
print("\n[5/5] Testing configuration compatibility...")
try:
    from ollama_fleet.config import load_settings
    settings = load_settings()
    
    # Verify essential config fields exist
    assert hasattr(settings, 'ollama'), "Missing ollama config"
    assert hasattr(settings, 'scheduler'), "Missing scheduler config"
    
    print("    ✓ Configuration module compatible with GUI")
except Exception as e:
    print(f"    ✗ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED")
print("=" * 80)
print("\nGUI is ready for use. Start with:")
print("  python3 gui_tk.py --demo")
print("  python3 gui_tk.py --goal 'Your goal here'")
print("=" * 80 + "\n")
