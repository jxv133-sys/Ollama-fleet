#!/usr/bin/env python3
"""Comprehensive test of Ollama Fleet with detailed agent responses and project verification."""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ollama_fleet.config import load_settings
from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.orchestrator import Orchestrator
from scripts.demo_run import DummyExecutor


class DetailedEventBus:
    """Captures all events with detailed information."""
    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self.agent_responses: dict[str, list[dict]] = {
            "planner": [],
            "coder": [],
            "critic": [],
            "tester": [],
            "synthesizer": []
        }
    
    def publish(self, event: dict[str, Any]) -> None:
        """Capture event and categorize agent responses."""
        self.events.append(event)
        
        event_type = event.get("type")
        
        # Capture agent outputs
        if event_type == "agent_output":
            agent_type = event.get("agent_type", "unknown")
            if agent_type in self.agent_responses:
                self.agent_responses[agent_type].append({
                    "timestamp": datetime.now().isoformat(),
                    "output": event.get("output", {}),
                    "event": event
                })
        
        # Log for console visibility
        if event_type in ("job_state_changed", "agent_output", "agent_log"):
            print(f"[{event_type}] {event}")


async def run_full_test() -> tuple[str, dict]:
    """Run orchestrator and capture all details."""
    # Clean up old database
    db_path = Path("full_test.db")
    if db_path.exists():
        db_path.unlink()
    
    settings = load_settings()
    db = Database(db_path)
    await db.connect()
    
    # Initialize with event bus and demo executor
    event_bus = DetailedEventBus()
    orch = Orchestrator(db, settings, ui_bus=event_bus)
    orch.executor = DummyExecutor(settings)
    
    print("\n" + "="*80)
    print("OLLAMA FLEET - FULL INTEGRATION TEST")
    print("="*80 + "\n")
    
    # Submit job
    goal = "Create a comprehensive Python project with utilities module"
    print(f"[TEST] Submitting job with goal: {goal}\n")
    
    job_id = await orch.submit_job(goal=goal, config={"source": "full_test"})
    
    print(f"\n[TEST] Job completed: {job_id}\n")
    
    # Get job details
    job = await orch.job_manager.get_job(job_id)
    
    await db.close()
    
    # Prepare report data
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "job_id": job_id,
        "goal": goal,
        "status": job.state if job else "unknown",
        "workspace": str(job.workspace_path) if job else "unknown",
        "events_total": len(event_bus.events),
        "agent_responses": event_bus.agent_responses,
        "events": event_bus.events,
    }
    
    return job_id, report_data


def verify_project(workspace_path: str) -> dict[str, Any]:
    """Verify created project structure and files."""
    ws = Path(workspace_path)
    
    verification = {
        "workspace_exists": ws.exists(),
        "workspace_path": workspace_path,
        "files_created": [],
        "directories_created": [],
        "file_contents": {},
        "total_size": 0,
    }
    
    if not ws.exists():
        return verification
    
    # Scan workspace
    for item in ws.rglob("*"):
        if item.is_file() and not item.name.startswith("."):
            rel_path = str(item.relative_to(ws))
            verification["files_created"].append(rel_path)
            
            # Read file content if text
            try:
                content = item.read_text(encoding="utf-8")
                verification["file_contents"][rel_path] = content[:500]  # First 500 chars
                verification["total_size"] += len(content)
            except Exception as e:
                verification["file_contents"][rel_path] = f"[Error reading: {e}]"
        
        elif item.is_dir() and not item.name.startswith("."):
            rel_path = str(item.relative_to(ws))
            if rel_path not in verification["directories_created"]:
                verification["directories_created"].append(rel_path)
    
    return verification


async def main() -> int:
    """Run full test and generate report."""
    # Run orchestrator test
    job_id, report_data = await run_full_test()
    
    # Verify project
    workspace_path = report_data["workspace"]
    verification = verify_project(workspace_path)
    report_data["verification"] = verification
    
    # Generate report
    report_file = Path("full_test_report.txt")
    with open(report_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("OLLAMA FLEET - FULL TEST REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {report_data['timestamp']}\n")
        f.write(f"Job ID: {job_id}\n")
        f.write(f"Goal: {report_data['goal']}\n")
        f.write(f"Status: {report_data['status']}\n")
        f.write(f"Total Events: {report_data['events_total']}\n\n")
        
        # Agent responses
        f.write("="*80 + "\n")
        f.write("AGENT RESPONSES\n")
        f.write("="*80 + "\n\n")
        
        for agent_name, responses in report_data["agent_responses"].items():
            f.write(f"### {agent_name.upper()} Agent ###\n")
            if responses:
                for i, response in enumerate(responses, 1):
                    f.write(f"Response {i}:\n")
                    f.write(f"  Timestamp: {response['timestamp']}\n")
                    f.write(f"  Output: {json.dumps(response['output'], indent=4)}\n\n")
            else:
                f.write("No responses\n\n")
        
        # Event timeline
        f.write("="*80 + "\n")
        f.write("EVENT TIMELINE\n")
        f.write("="*80 + "\n\n")
        
        for i, event in enumerate(report_data["events"], 1):
            f.write(f"{i}. {event.get('type')}\n")
            if event.get("type") == "agent_output":
                f.write(f"   Agent: {event.get('agent_type')}\n")
                f.write(f"   Output: {event.get('output')}\n")
            elif event.get("type") == "task_state_changed":
                f.write(f"   Task: {event.get('task_id')}\n")
                f.write(f"   Agent: {event.get('agent_type')}\n")
                f.write(f"   State: {event.get('new_state')}\n")
            elif event.get("type") == "agent_log":
                f.write(f"   Message: {event.get('message')}\n")
            elif event.get("type") == "file_written":
                f.write(f"   File: {event.get('path')}\n")
            f.write("\n")
        
        # Project verification
        f.write("="*80 + "\n")
        f.write("PROJECT VERIFICATION\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Workspace Path: {verification['workspace_path']}\n")
        f.write(f"Workspace Exists: {verification['workspace_exists']}\n")
        f.write(f"Total Size: {verification['total_size']} bytes\n\n")
        
        f.write(f"Files Created ({len(verification['files_created'])}):\n")
        for file_path in sorted(verification["files_created"]):
            f.write(f"  - {file_path}\n")
        f.write("\n")
        
        f.write(f"Directories Created ({len(verification['directories_created'])}):\n")
        for dir_path in sorted(verification["directories_created"]):
            f.write(f"  - {dir_path}\n")
        f.write("\n")
        
        f.write("File Contents Preview:\n")
        for file_path, content in sorted(verification["file_contents"].items()):
            f.write(f"\n--- {file_path} ---\n")
            f.write(content)
            if len(content) > 400:
                f.write("\n... [truncated]")
            f.write("\n")
        
        # Summary
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        agents_responded = sum(1 for agent, responses in report_data["agent_responses"].items() if responses)
        f.write(f"Agents with responses: {agents_responded}/5\n")
        f.write(f"Files created: {len(verification['files_created'])}\n")
        f.write(f"Directories created: {len(verification['directories_created'])}\n")
        f.write(f"Total events: {report_data['events_total']}\n")
        
        # Status indicator
        if verification['workspace_exists'] and len(verification['files_created']) > 0:
            f.write("\n✓ TEST PASSED - Project created successfully\n")
        else:
            f.write("\n✗ TEST FAILED - Project not created\n")
    
    # Print summary to console
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"\nWorkspace: {verification['workspace_path']}")
    print(f"Files Created: {len(verification['files_created'])}")
    print(f"Directories: {len(verification['directories_created'])}")
    print(f"Total Size: {verification['total_size']} bytes")
    print(f"\nAgent Responses:")
    for agent_name, responses in report_data["agent_responses"].items():
        print(f"  {agent_name}: {len(responses)} response(s)")
    
    print(f"\nReport saved to: {report_file}")
    print("\n" + "="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
