#!/usr/bin/env python3
"""Simple synchronous test for Planner agent with improved prompts."""

import sys
sys.path.insert(0, '/Users/jonahvaira/Documents/GitHub/Ollama-fleet/Ollama-fleet-')

from ollama_fleet.ollama.client import OllamaClient
from ollama_fleet.agents.executor import AgentExecutor
from ollama_fleet.agents.schemas import AgentType
from ollama_fleet.config import load_settings

print("=" * 80)
print("SIMPLE PLANNER DEBUG TEST")
print("=" * 80)

try:
    # Load config
    print("\n1. Loading config...")
    settings = load_settings()
    print(f"   ✓ Server: {settings.ollama.base_url}")
    print(f"   ✓ Planner Model: {settings.ollama.planner_model}")
    print(f"   ✓ Timeout: {settings.ollama.timeout}s")
    
    # Create client and executor
    print("\n2. Creating client and executor...")
    client = OllamaClient(settings.ollama.base_url)
    executor = AgentExecutor(client, settings)
    print("   ✓ Ready to execute")
    
    # Execute planner synchronously
    print("\n3. Executing planner agent...")
    print("   Calling executor.execute() directly...")
    
    result = executor.execute(
        {
            "goal": "Create a simple Python calculator",
            "task_id": "test-1",
        },
        AgentType.PLANNER,
        {"architecture_notes": "Simple modular design"}
    )
    
    print(f"\n4. Result received:")
    print(f"   Type: {type(result)}")
    print(f"   Keys: {list(result.keys()) if hasattr(result, 'keys') else 'N/A'}")
    
    if isinstance(result, dict):
        if 'output' in result:
            print(f"\n   Output type: {type(result['output'])}")
            print(f"   Output: {result['output']}")
        if 'error' in result:
            print(f"\n   Error: {result['error']}")
        if 'success' in result:
            print(f"\n   Success: {result['success']}")
            
    print("\n✓ Test completed")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
