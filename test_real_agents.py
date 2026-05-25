"""
Real Ollama Agent Test - Direct Agent Execution

This test directly executes real agents with Ollama to demonstrate
actual LLM responses for orchestration tasks.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ollama_fleet.agents.executor import AgentExecutor
from ollama_fleet.agents.schemas import AgentType
from ollama_fleet.config import load_settings
from ollama_fleet.ollama.client import OllamaClient


async def test_real_agents():
    """Test real agents with Ollama directly"""
    print("\n" + "=" * 80)
    print("REAL OLLAMA AGENT TEST - DIRECT EXECUTION")
    print("=" * 80)

    # Load settings
    settings = load_settings()
    print(f"\n✓ Configuration Loaded")
    print(f"  Ollama Server: {settings.ollama.base_url}")
    print(f"  Planner Model: {settings.ollama.planner_model}")
    print(f"  Timeout: {settings.ollama.timeout}s")

    # Create executor
    client = OllamaClient(settings.ollama.base_url)
    executor = AgentExecutor(client, settings)

    # Test cases
    test_cases = [
        {
            "name": "Planner: Simple Python Module",
            "agent_type": AgentType.PLANNER,
            "task": {
                "goal": "Create a Python module with a function that calculates the sum of two numbers",
                "task_id": "planner-test-1",
            },
            "extra_context": {"architecture_notes": "Simple module with basic functions"},
        },
        {
            "name": "Coder: Implement Function",
            "agent_type": AgentType.CODER,
            "task": {
                "description": "Implement a Python function that takes two numbers and returns their sum. Include docstring and type hints.",
                "task_id": "coder-test-1",
            },
            "extra_context": {
                "active_files": [],
                "episodic_summaries": [],
            },
        },
    ]

    results = []

    for test_case in test_cases:
        print(f"\n{'=' * 80}")
        print(f"Testing: {test_case['name']}")
        print(f"{'=' * 80}")

        agent_type = test_case["agent_type"]
        task = test_case["task"]
        extra_context = test_case.get("extra_context", {})

        start_time = time.time()
        try:
            print(f"  Sending request to Ollama ({settings.ollama.timeout}s timeout)...")
            print(f"  Agent Type: {agent_type.value}")
            print(f"  Task ID: {task['task_id']}")

            # Execute agent with proper asyncio timeout
            try:
                output = await asyncio.wait_for(
                    executor.execute(task, agent_type, extra_context),
                    timeout=settings.ollama.timeout + 30,  # Add 30s buffer
                )
                elapsed = time.time() - start_time

                print(f"\n✓ Agent Response Received ({elapsed:.2f}s)")
                print(f"  Output Type: {type(output).__name__}")
                print(f"  Keys: {list(output.model_dump().keys())}")

                # Store result
                result = {
                    "test": test_case["name"],
                    "status": "success",
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "output_type": type(output).__name__,
                    "output": output.model_dump(),
                }

                # Print sample output
                output_dict = output.model_dump()
                if agent_type == AgentType.PLANNER:
                    tasks = output_dict.get("tasks", [])
                    print(f"  Tasks Created: {len(tasks)}")
                    if tasks:
                        print(f"    First Task: {tasks[0].get('title', 'N/A')}")
                        print(f"    Description: {tasks[0].get('description', 'N/A')[:80]}")
                    milestones = output_dict.get("milestones", [])
                    print(f"  Milestones: {milestones}")

                elif agent_type == AgentType.CODER:
                    files = output_dict.get("file_modifications", [])
                    print(f"  Files Modified: {len(files)}")
                    if files:
                        print(f"    First File: {files[0].get('file_path', 'N/A')}")
                    print(f"  Confidence: {output_dict.get('confidence_score', 'N/A')}")
                    print(f"  Summary: {output_dict.get('summary', 'N/A')[:80]}")

                results.append(result)

            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                print(f"✗ Timeout after {elapsed:.2f}s")
                results.append(
                    {
                        "test": test_case["name"],
                        "status": "timeout",
                        "elapsed_ms": round(elapsed * 1000, 2),
                    }
                )

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ Error: {type(e).__name__}: {e}")
            results.append(
                {
                    "test": test_case["name"],
                    "status": "error",
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "error": str(e),
                }
            )

    # Generate report
    print(f"\n{'=' * 80}")
    print("TEST RESULTS SUMMARY")
    print(f"{'=' * 80}\n")

    for result in results:
        status_symbol = (
            "✓"
            if result["status"] == "success"
            else "✗"
        )
        print(
            f"{status_symbol} {result['test']}"
        )
        print(f"   Status: {result['status']}")
        print(f"   Elapsed: {result['elapsed_ms']}ms")
        if "error" in result:
            print(f"   Error: {result['error']}")
        print()

    # Save report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_type": "Real Ollama Agent Direct Execution",
        "ollama_server": settings.ollama.base_url,
        "results": results,
        "summary": {
            "total": len(results),
            "success": len([r for r in results if r["status"] == "success"]),
            "timeout": len([r for r in results if r["status"] == "timeout"]),
            "error": len([r for r in results if r["status"] == "error"]),
        },
    }

    report_path = Path("real_ollama_agent_test_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"✓ Report saved to {report_path}")

    print(f"\n{'=' * 80}")
    print("REAL OLLAMA AGENT TEST COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    asyncio.run(test_real_agents())
