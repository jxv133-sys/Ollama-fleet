#!/usr/bin/env python3
"""Debug test for Planner agent with improved prompts."""

import sys
sys.path.insert(0, '/Users/jonahvaira/Documents/GitHub/Ollama-fleet/Ollama-fleet-')

from ollama_fleet.ollama.client import OllamaClient
from ollama_fleet.agents.planner import build_planner_prompt
from ollama_fleet.agents.executor import AgentExecutor
from ollama_fleet.agents.schemas import AgentType
from ollama_fleet.config import load_settings
import asyncio

print("=" * 80)
print("DEBUG: Planner Agent Test")
print("=" * 80)

async def test():
    try:
        # Load config
        print("\n1. Loading config...")
        settings = load_settings()
        print(f"   Config loaded: {settings.ollama.base_url}")
    
    # Create executor
    print("\n2. Creating executor...")
    executor = AgentExecutor()
    print("   Executor created")
    
    # Build prompt
    print("\n3. Building planner prompt...")
    prompt = build_planner_prompt(
        goal="Build a simple Python CLI tool",
        architecture="Modular with separate files"
    )
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   Prompt preview: {prompt[:200]}...")
    
    # Execute planner
    print("\n4. Executing planner agent...")
    print("   Sending request to Ollama...")
    result = executor.execute(
        agent_type="planner",
        task_id="debug-test-1",
        goal="Build a simple Python CLI tool",
        context="Modular architecture"
    )
    
    print("\n5. Result:")
    print(f"   Success: {result.get('success')}")
    print(f"   Output type: {type(result.get('output'))}")
    if result.get('output'):
        print(f"   Output keys: {result['output'].keys() if hasattr(result['output'], 'keys') else 'Not dict'}")
        print(f"   Output: {result.get('output')}")
    if result.get('error'):
        print(f"   Error: {result['error']}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debug test complete")
print("=" * 80)
