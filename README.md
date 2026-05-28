# Ollama-fleet-

This repository now includes an agent pipeline scaffold in `ollama_fleet/agents/pipeline.py`.

- Central prompt builder for all agent types
- Model selection with critic/tester fallback to coder model
- Response validation against Pydantic agent schemas
- End-to-end runner with `AgentPipeline.run_agent()`
