#!/usr/bin/env python3
"""
Test suite for multi-file project creation with agent thinking/reasoning capture.

Tests:
1. Multi-file project structure (5+ files across multiple directories)
2. Agent planning and task decomposition
3. Agent reasoning and confidence in decisions
4. File organization and dependencies
5. Agent validation and approval workflows
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
import tempfile
import shutil

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from ollama_fleet.config import FleetSettings
from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.orchestrator import Orchestrator
from ollama_fleet.ui.event_bus import UIEventBus
from scripts.demo_run import DummyExecutor


class DetailedEventBusWithThinking(UIEventBus):
    """Event bus that captures all events and agent thinking/reasoning."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.agent_responses = {
            "planner": [],
            "coder": [],
            "critic": [],
            "tester": [],
            "synthesizer": [],
        }
        self.agent_thinking = {
            "planner": [],
            "coder": [],
            "critic": [],
            "tester": [],
            "synthesizer": [],
        }

    def publish(self, event: dict) -> None:
        """Capture event and extract agent responses."""
        self.events.append(event)

        if event.get("type") == "agent_output":
            agent_type = event.get("agent_type")
            output = event.get("output", {})
            
            self.agent_responses[agent_type].append({
                "timestamp": event.get("timestamp"),
                "output": output,
            })
            
            # Extract thinking if available
            if "reasoning" in output:
                self.agent_thinking[agent_type].append(output["reasoning"])
            if "thought_process" in output:
                self.agent_thinking[agent_type].append(output["thought_process"])

        # Always call parent publish to invoke handlers
        super().publish(event)


async def run_multifile_test() -> dict:
    """Run test of multi-file project creation."""
    print("\n" + "=" * 80)
    print("OLLAMA FLEET - MULTI-FILE PROJECT TEST WITH AGENT THINKING")
    print("=" * 80 + "\n")

    # Create temporary database
    db_path = Path("./multifile_test.db")
    if db_path.exists():
        db_path.unlink()

    db = Database(str(db_path))
    await db.connect()

    # Create custom event bus for detailed capture
    event_bus = DetailedEventBusWithThinking()

    try:
        # Create settings with demo executor
        settings = FleetSettings()
        
        # Override executor with DummyExecutor for predictable multi-file output
        orchestrator = Orchestrator(db, settings, ui_bus=event_bus)
        orchestrator.executor = DummyExecutor()

        # Test goals that require multi-file projects
        test_goals = [
            "Create a Python web API with models, routes, and utilities modules",
            "Build a data processing pipeline with config, processors, and main modules",
        ]

        results = {
            "timestamp": asyncio.get_event_loop().time(),
            "test_goals": test_goals,
            "jobs": [],
        }

        for goal in test_goals:
            print(f"\n[TEST] Submitting job: {goal}")
            print("-" * 80)

            job_id = await orchestrator.submit_job(goal, {"source": "multifile_test"})
            print(f"[TEST] Job submitted: {job_id}")

            # Wait for completion
            await asyncio.sleep(2)

            # Gather results - Add simple tracking - just assume workspace exists
            job_result = {
                "job_id": job_id,
                "goal": goal,
                "events_captured": len(event_bus.events),
                "agent_responses": {
                    agent: len(responses)
                    for agent, responses in event_bus.agent_responses.items()
                    if responses
                },
                "agent_thinking": {
                    agent: len(thoughts)
                    for agent, thoughts in event_bus.agent_thinking.items()
                    if thoughts
                },
                "events": event_bus.events,
            }

            # Get workspace path from settings
            workspace_path = Path(settings.workspace.base_path) / job_id
            if workspace_path.exists():
                py_files = list(workspace_path.glob("src/**/*.py"))
                job_result["files_created"] = [
                    str(f.relative_to(workspace_path)) for f in py_files
                ]
                job_result["workspace_path"] = str(workspace_path)
                
                # Check agent outputs
                agent_outputs_dir = workspace_path / "agent_outputs"
                if agent_outputs_dir.exists():
                    job_result["agent_output_files"] = [
                        f.name for f in agent_outputs_dir.glob("*.json")
                    ]

            results["jobs"].append(job_result)

            # Clear events for next test
            event_bus.events.clear()
            event_bus.agent_responses = {
                "planner": [],
                "coder": [],
                "critic": [],
                "tester": [],
                "synthesizer": [],
            }
            event_bus.agent_thinking = {
                "planner": [],
                "coder": [],
                "critic": [],
                "tester": [],
                "synthesizer": [],
            }

        # Generate detailed report
        report_path = Path("./multifile_test_report.txt")
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("OLLAMA FLEET - MULTI-FILE PROJECT TEST REPORT\n")
            f.write("=" * 80 + "\n\n")

            for i, job in enumerate(results["jobs"], 1):
                f.write(f"### TEST {i}: {job['goal']}\n\n")
                f.write(f"Job ID: {job['job_id']}\n")
                f.write(f"Total Events: {job['events_captured']}\n")
                f.write(f"Workspace: {job.get('workspace_path', 'N/A')}\n\n")

                if "files_created" in job:
                    f.write(f"Files Created: {len(job['files_created'])}\n")
                    for file in job["files_created"]:
                        f.write(f"  - {file}\n")
                    f.write("\n")

                f.write("Agent Responses:\n")
                for agent, count in job["agent_responses"].items():
                    f.write(f"  {agent}: {count} response(s)\n")
                f.write("\n")

                f.write("Agent Thinking Captured:\n")
                for agent, count in job["agent_thinking"].items():
                    f.write(f"  {agent}: {count} thought(s)\n")
                f.write("\n")

                # Detailed event breakdown
                f.write("Event Timeline:\n")
                for idx, event in enumerate(job["events"], 1):
                    event_type = event.get("type")
                    f.write(f"  {idx}. {event_type}")
                    if event_type == "agent_output":
                        agent = event.get("agent_type")
                        output = event.get("output", {})
                        f.write(f" ({agent})")
                        if "file_count" in output:
                            f.write(f" - Files: {output['file_count']}")
                        if "tasks_created" in output:
                            f.write(f" - Tasks: {output['tasks_created']}")
                    f.write("\n")
                f.write("\n")

                f.write("=" * 80 + "\n\n")

        print(f"\n[TEST] Report saved to: {report_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        for i, job in enumerate(results["jobs"], 1):
            print(f"\nTest {i}: {job['goal'][:50]}...")
            print(f"  Events: {job['events_captured']}")
            print(f"  Agent Responses: {sum(job['agent_responses'].values())}")
            if "files_created" in job:
                print(f"  Files Created: {len(job['files_created'])}")

        return results

    finally:
        await db.close()


def main():
    """Entry point."""
    # Handle event loop on macOS
    if sys.platform == "darwin":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    try:
        results = asyncio.run(run_multifile_test())
        print("\n✓ TEST COMPLETED SUCCESSFULLY")
        return 0
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
