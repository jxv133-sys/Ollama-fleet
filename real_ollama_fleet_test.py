"""
Real Ollama Fleet Orchestration Test

This test uses actual Ollama agents (via the real OllamaClient) instead of
dummy responses. It submits real goals and collects LLM-generated responses.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ollama_fleet.config import load_settings
from ollama_fleet.db.database import Database
from ollama_fleet.ollama.client import OllamaClient, OllamaConnectionError
from ollama_fleet.orchestrator.orchestrator import Orchestrator
from ollama_fleet.ui.event_bus import UIEventBus


class RealOllamaTestObserver(UIEventBus):
    """Observer that captures all events from real Ollama agent execution."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.agent_responses = {}
        self.start_time = time.time()
        self.job_timings = {}

    def publish(self, event: dict) -> None:
        elapsed_ms = round((time.time() - self.start_time) * 1000, 2)
        event_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": elapsed_ms,
            "type": event.get("type"),
            "job_id": event.get("job_id"),
        }

        # Track agent responses
        if event.get("type") == "agent_output":
            agent_type = event.get("agent_type")
            job_id = event.get("job_id")

            if job_id not in self.agent_responses:
                self.agent_responses[job_id] = {}
            if agent_type not in self.agent_responses[job_id]:
                self.agent_responses[job_id][agent_type] = []

            output = event.get("output", {})
            self.agent_responses[job_id][agent_type].append(
                {
                    "elapsed_ms": elapsed_ms,
                    "output_keys": list(output.keys()) if isinstance(output, dict) else "non-dict",
                    "summary": self._summarize_output(agent_type, output),
                }
            )

        # Track job completion
        if event.get("type") == "job_state_changed":
            new_state = event.get("new_state")
            job_id = event.get("job_id")
            if new_state == "submitted":
                self.job_timings[job_id] = {"start": elapsed_ms}
            elif new_state == "completed":
                if job_id in self.job_timings:
                    self.job_timings[job_id]["end"] = elapsed_ms
                    self.job_timings[job_id]["duration_ms"] = (
                        elapsed_ms - self.job_timings[job_id]["start"]
                    )

        self.events.append(event_record)
        super().publish(event)

    def _summarize_output(self, agent_type: str, output: dict) -> str:
        """Create a brief summary of agent output for logging."""
        if agent_type == "planner":
            tasks = output.get("tasks", [])
            return f"{len(tasks)} tasks planned"
        elif agent_type == "coder":
            files = output.get("file_modifications", [])
            return f"{len(files)} files modified"
        elif agent_type == "critic":
            approved = output.get("approved", False)
            issues = len(output.get("issues", []))
            return f"{'approved' if approved else 'rejected'}, {issues} issues"
        elif agent_type == "tester":
            return "test results generated"
        elif agent_type == "synthesizer":
            return "summary generated"
        return "unknown agent type"

    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        lines = []
        lines.append("=" * 80)
        lines.append("REAL OLLAMA FLEET ORCHESTRATION TEST REPORT")
        lines.append("=" * 80)
        lines.append(f"\nTest Execution Date: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"Total Duration: {(time.time() - self.start_time):.2f} seconds")
        lines.append(f"Total Events Captured: {len(self.events)}")
        lines.append(f"Total Jobs: {len(self.job_timings)}")

        # Job timings
        lines.append("\n" + "-" * 80)
        lines.append("JOB EXECUTION TIMINGS")
        lines.append("-" * 80)
        for job_id, timing in self.job_timings.items():
            duration = timing.get("duration_ms", 0)
            lines.append(f"\nJob: {job_id}")
            lines.append(f"  Duration: {duration:.2f}ms")

            # Agent responses for this job
            if job_id in self.agent_responses:
                for agent_type, responses in self.agent_responses[job_id].items():
                    for resp in responses:
                        lines.append(
                            f"  {agent_type.capitalize()} @ {resp['elapsed_ms']}ms: {resp['summary']}"
                        )

        # Event timeline
        lines.append("\n" + "-" * 80)
        lines.append("EVENT TIMELINE")
        lines.append("-" * 80)
        for idx, event in enumerate(self.events, 1):
            job_id = event.get("job_id", "N/A")
            job_short = job_id[:8] if job_id != "N/A" else job_id
            lines.append(
                f"{idx:2d}. [{event['elapsed_ms']:7.2f}ms] {event['type']:20s} ({job_short})"
            )

        # Ollama connectivity info
        lines.append("\n" + "-" * 80)
        lines.append("OLLAMA CONFIGURATION")
        lines.append("-" * 80)
        lines.append(f"Server: Using real Ollama agents via OllamaClient")
        lines.append(f"Model Pipeline: Planner → Coder → Critic → (Tester/Synthesizer)")

        lines.append("\n" + "-" * 80)
        lines.append("SUMMARY")
        lines.append("-" * 80)
        total_duration = sum(t.get("duration_ms", 0) for t in self.job_timings.values())
        lines.append(f"✓ Completed {len(self.job_timings)} jobs with real Ollama agents")
        lines.append(f"✓ Total execution time: {total_duration:.2f}ms")
        lines.append(f"✓ Event throughput: {len(self.events) / (time.time() - self.start_time):.1f} events/sec")
        if self.job_timings:
            avg_job_time = total_duration / len(self.job_timings)
            lines.append(f"✓ Average job time: {avg_job_time:.2f}ms")

        return "\n".join(lines)


async def verify_ollama_connection(base_url: str) -> bool:
    """Verify that Ollama is reachable before running tests."""
    client = OllamaClient(base_url)
    try:
        # Try a simple generate with short timeout to verify connection
        result = await client.generate(
            model="hf.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M",
            prompt="Respond with just 'ok'",
            timeout=10.0,
        )
        return True
    except OllamaConnectionError as e:
        print(f"✗ Cannot connect to Ollama at {base_url}")
        print(f"  Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Ollama connection test failed: {e}")
        return False


async def run_real_ollama_test():
    """Run comprehensive test with real Ollama agents."""
    print("\n" + "=" * 80)
    print("REAL OLLAMA FLEET TEST - STARTING")
    print("=" * 80)

    # Load settings
    settings = load_settings()
    print(f"\n✓ Loaded configuration")
    print(f"  Ollama Server: {settings.ollama.base_url}")
    print(f"  Planner Model: {settings.ollama.planner_model}")
    print(f"  Coder Model: {settings.ollama.coder_model}")
    print(f"  Critic Model: {settings.ollama.critic_model}")

    # Verify Ollama connection
    print(f"\n✓ Verifying Ollama connection...")
    if not await verify_ollama_connection(settings.ollama.base_url):
        print("✗ Cannot proceed without Ollama")
        return

    # Create database and orchestrator
    db_path = Path("real_ollama_test.db")
    async with Database(db_path) as db:
        observer = RealOllamaTestObserver()
        orchestrator = Orchestrator(db, settings, ui_bus=observer)

        # Test cases with real goals
        test_cases = [
            {
                "name": "Simple Python Module",
                "goal": "Create a simple Python module with a function that calculates the sum of two numbers. Include docstrings and type hints.",
            },
            {
                "name": "Data Processor",
                "goal": "Create a Python module with a class that processes and filters data. Include error handling and validation.",
            },
        ]

        print(f"\n{'=' * 80}")
        print("STARTING TEST CASES WITH REAL AGENTS")
        print("=" * 80)

        for idx, test_case in enumerate(test_cases, 1):
            print(f"\n[{idx}/{len(test_cases)}] {test_case['name']}")
            print(f"Goal: {test_case['goal'][:70]}...")

            try:
                job_id = await orchestrator.submit_job(
                    goal=test_case["goal"],
                    config={"test": True, "case": test_case["name"]},
                )
                print(f"✓ Job submitted: {job_id}")

                # Wait for job to complete (with timeout)
                timeout = 900  # 15 minutes max per job for real LLM
                start = time.time()
                while time.time() - start < timeout:
                    # Check job state
                    job = await db.query_one(
                        "SELECT state FROM jobs WHERE id = ?", (job_id,)
                    )
                    if job and job[0] in ("completed", "failed", "cancelled"):
                        print(f"✓ Job {job_id[:8]}: {job[0].upper()}")
                        break
                    await asyncio.sleep(5)  # Poll every 5 seconds

            except Exception as e:
                print(f"✗ Test case failed: {e}")
                import traceback
                traceback.print_exc()

        # Generate and save report
        report = observer.generate_report()
        print(f"\n{report}")

        # Save report to file
        report_path = Path("real_ollama_test_observations.txt")
        report_path.write_text(report)
        print(f"\n✓ Report saved to {report_path}")

    # Cleanup
    if db_path.exists():
        db_path.unlink()

    print(f"\n{'=' * 80}")
    print("REAL OLLAMA FLEET TEST - COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_real_ollama_test())
