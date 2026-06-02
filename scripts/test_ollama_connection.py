#!/usr/bin/env python3
"""Test connectivity to an Ollama server and verify AgentExecutor can call it.

Usage:
    PYTHONPATH=. python3 scripts/test_ollama_connection.py

This will attempt a small generation using the configured planner model and
then call the AgentExecutor for a planner task to ensure end-to-end flow.
"""
from __future__ import annotations

import asyncio
import os
import sys
from ollama_fleet.config import load_settings
from ollama_fleet.ollama.client import OllamaClient, OllamaConnectionError, OllamaHTTPError, OllamaTimeoutError
from ollama_fleet.agents.executor import AgentExecutor
from ollama_fleet.agents.schemas import AgentType


async def main() -> int:
    settings = load_settings()
    base = settings.ollama.base_url
    model = settings.ollama.planner_model
    timeout = min(60.0, settings.ollama.timeout)

    print(f"Using Ollama base URL: {base}")
    print(f"Using planner model: {model}")

    client = OllamaClient(base)

    # List available models and pick a suitable one for a quick test if possible
    try:
        models = await client.list_models()
        print(f"Server returned {len(models)} model(s)")
        if models:
            # models entries may include keys like 'name' or 'model'
            first = models[0]
            candidate = first.get("name") or first.get("model") or None
            print("Using detected model:", candidate)
            model_to_use = candidate or model
        else:
            model_to_use = model
    except Exception as e:
        print("Could not list models, falling back to configured planner_model:", e)
        model_to_use = model

    test_prompt = "Say hello in one sentence."
    try:
        print("Attempting a direct generate() call using model:", model_to_use)
        resp = await client.generate(model=model_to_use, prompt=test_prompt, timeout=timeout)
        print("Generate response (truncated):\n", resp[:500])
    except OllamaConnectionError as e:
        print("Connection error:", e)
        return 2
    except OllamaHTTPError as e:
        print("HTTP error:", e.status_code, e.body)
        return 3
    except OllamaTimeoutError as e:
        print("Timeout:", e)
        return 4
    except Exception as e:
        print("Unexpected error during generate():", e)
        return 5

    # Now test AgentExecutor end-to-end for Planner
    executor = AgentExecutor(client, settings)
    task = {"task_id": "test-planner-1", "goal": "Connectivity test", "description": "Return a valid minimal planner JSON response."}
    try:
        print("Running AgentExecutor.execute for Planner (may take longer)...")
        out, prompt = await executor.execute(task, AgentType.PLANNER, extra_context={})
        print("Planner output type:", type(out), "\nPrompt (truncated):", repr(prompt[:200]))
    except Exception as e:
        print("AgentExecutor failed:", e)
        return 6

    print("OK: Ollama reachable and AgentExecutor executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
