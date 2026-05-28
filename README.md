# Ollama-fleet-

This repository now includes an agent pipeline scaffold in `ollama_fleet/agents/pipeline.py`.

- Central prompt builder for all agent types
- Planner asks clarifying questions, captures technical requirements, and returns numbered task list
- Coding stage executes tasks one at a time in step order
- Scoring stage runs critic and tester agents after coding
- Response validation against Pydantic agent schemas
- End-to-end runner with `AgentPipeline.run_agent()`
