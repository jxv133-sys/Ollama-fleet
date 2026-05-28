# Implementation Plan: Ollama Fleet

## Overview

Implement Ollama Fleet as a four-phase incremental build. Each phase produces a runnable system before the next phase begins. Phase 1 delivers a working Planner → Coder → Tester pipeline. Phase 2 adds the critique/revision loop and validation. Phase 3 adds crash resilience, dependency tracking, and episodic memory. Phase 4 adds advanced context compression, confidence-score routing, and long-term memory.

All implementation is in Python using `asyncio`, `httpx`, `pydantic v2`, `textual`, `hypothesis`, and `sqlite3`.

---

## Tasks

## Phase 1: Core Pipeline (Planner → Coder → Tester)

- [x] 1. Implement the configuration system
  - Implement `ollama_fleet/config.py` with all Pydantic settings models
  - Support `OLLAMA_FLEET_CONFIG`; default to `./config.toml`
  - Enforce validation constraints and exit non-zero on invalid config

- [x] 2. Implement the database layer
  - Implement `ollama_fleet/db/database.py` with `aiosqlite` and migration runner
  - Add `db/migrations/001_initial.sql` with `jobs`, `tasks`, and `escalations`

- [x] 3. Implement the Ollama client
  - Implement `ollama_fleet/ollama/client.py` with async streaming and typed errors

- [x] 4. Implement agent output schemas
  - Implement `ollama_fleet/agents/schemas.py` for Planner, Coder, Critic, Tester, and Synthesizer outputs

- [x] 5. Implement the Workspace Manager
  - Implement workspace creation, atomic file writes, and execution history logging

- [x] 6. Implement the Tool Runtime
  - Implement file tools, shell tools, git tools, and the central runtime dispatcher

- [x] 7. Implement the Task Scheduler
  - Implement task enqueueing, ready-task selection, state transitions, retry counting, and cancellation

- [x] 8. Implement the Agent Executor and prompt builders
  - Implement `ollama_fleet/agents/executor.py` with prompt assembly, validation retry, timeout handling, and logging
  - Add Planner, Coder, and Tester prompt builders

- [x] 9. Implement the Phase 1 Orchestrator and job lifecycle
  - Implement job CRUD, submission, planner invocation, task enqueueing, and the dispatch loop
  - Add error handling for planner failures and task recoveries

- [x] 10. Implement the CLI entry point
  - Implement `ollama_fleet/main.py` with argument parsing, config loading, database startup, and orchestrator submission

- [ ] 11. Phase 1 checkpoint
  - Verify a submitted job runs Planner → Coder → Tester
  - Verify SQLite persistence and basic command-line execution

---

## Phase 2: Critique Loop + Validation + Synthesizer

- [x] 12. Implement validation
  - Implement `ollama_fleet/validation/validator.py` with syntax checking and lint parsing
  - Ensure syntax failures can requeue code generation without consuming schema retries

- [x] 13. Implement Critic and Synthesizer agents
  - Add Critic and Synthesizer prompt builders and schema support

- [ ] 14. Extend the Orchestrator with the critique/revision loop
  - Add critic result handling, revision task creation, and loop limits

- [ ] 15. Implement escalation support
  - Add escalation persistence and loop-detection logic in `ollama_fleet/orchestrator/escalation.py`

- [ ] 16. Add UI panels for validation and escalation
  - Add validation and escalation rendering in `ollama_fleet/ui/panels.py`

- [ ] 17. Add episodic memory migration
  - Add `db/migrations/002_episodic_memory.sql`

- [ ] 18. Phase 2 checkpoint
  - Verify full Planner → Coder → Critic → Revision → Tester → Synthesizer flow

---

## Phase 3: Crash Resilience + Dependency Tracking + Episodic Memory

- [ ] 19. Add crash recovery
  - Recover running tasks as pending on startup

- [ ] 20. Add dependency resolution
  - Implement `ollama_fleet/scheduler/dependency_resolver.py`
  - Wire it into the orchestrator dispatch loop

- [ ] 21. Implement episodic memory persistence
  - Add `ollama_fleet/memory/episodic.py`
  - Add `ollama_fleet/memory/memory_system.py`

- [ ] 22. Add stall detection
  - Implement background stall detection and escalation

- [ ] 23. Phase 3 checkpoint
  - Verify crash recovery, dependency-ordered execution, and episodic memory injection

---

## Phase 4: Advanced Context + Confidence Routing + Long-Term Memory

- [ ] 24. Add long-term memory and context compression
  - Add `db/migrations/003_long_term_memory.sql`
  - Add `ollama_fleet/memory/long_term.py`

- [ ] 25. Add confidence-score routing
  - Route low-confidence coder outputs through additional critic review

- [ ] 26. Verify migration compatibility
  - Ensure migrations are idempotent and backward-compatible

- [ ] 27. Phase 4 checkpoint
  - Verify advanced compression, low-confidence review routing, and searchable long-term memory
