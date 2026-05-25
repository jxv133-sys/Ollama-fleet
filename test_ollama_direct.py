"""
Direct test of Ollama client to diagnose issues
"""
import asyncio
import time
from ollama_fleet.ollama.client import OllamaClient

async def test_direct():
    """Test Ollama client directly"""
    print("=" * 80)
    print("DIRECT OLLAMA CLIENT TEST")
    print("=" * 80)
    
    client = OllamaClient("http://192.168.50.142:7869")
    
    # Test 1: Simple prompt
    print("\nTest 1: Simple 'ok' prompt (15s timeout)")
    start = time.time()
    try:
        result = await asyncio.wait_for(
            client.generate(
                model="hf.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M",
                prompt="Respond with just: ok",
                timeout=300.0,  # 5 min timeout for Ollama
            ),
            timeout=15.0  # 15s total timeout for this test
        )
        elapsed = time.time() - start
        print(f"✓ Success in {elapsed:.2f}s")
        print(f"  Response length: {len(result)} chars")
        print(f"  First 100 chars: {result[:100]}")
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"✗ Timeout after {elapsed:.2f}s (asyncio timeout)")
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Error after {elapsed:.2f}s: {type(e).__name__}: {e}")
    
    # Test 2: Longer timeout
    print("\nTest 2: Same prompt (60s timeout)")
    start = time.time()
    try:
        result = await asyncio.wait_for(
            client.generate(
                model="hf.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M",
                prompt="Respond with just: ok",
                timeout=300.0,  # 5 min timeout for Ollama
            ),
            timeout=60.0  # 60s total timeout
        )
        elapsed = time.time() - start
        print(f"✓ Success in {elapsed:.2f}s")
        print(f"  Response length: {len(result)} chars")
        print(f"  First 100 chars: {result[:100]}")
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"✗ Timeout after {elapsed:.2f}s (asyncio timeout)")
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Error after {elapsed:.2f}s: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_direct())
