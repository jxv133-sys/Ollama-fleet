# Implementation Plan: Ollama Fleet

## Overview

Implement Ollama Fleet as a four-phase incremental build. Each phase produces a runnable system before the next phase begins. Phase 1 delivers a working Planner → Coder → Tester pipeline. Phase 2 adds the critique/revision loop and validation. Phase 3 adds crash resilience, dependency tracking, and episodic memory. Phase 4 adds advanced context compression, confidence-score routing, and long-term memory.

All implementation is in Python using `asyncio`, `httpx`, `pydantic v2`, `textual`, `hypothesis`, and `sqlite3`.

---

## Tasks

## Phase 1: Core Pipeline (Planner → Coder → Tester)

- [x] 1. Scaffold project structure and tooling
  - [x] 1.1 Create `pyproject.toml` with all dependencies and dev tooling
    - Add `httpx[asyncio]`, `pydantic>=2`, `pydantic-settings`, `textual`, `rich`, `tomllib`/`tomli`, `hypothesis`, `pytest`, `pytest-asyncio`, `ruff`, `aiosqlite`
    - Configure `[tool.pytest.ini_options]` with `asyncio_mode = "auto"`
    - Configure `[tool.hypothesis]` with `max_examples = 100`
    - Configure `[tool.ruff]` linting rules
    - _Requirements: 14.1_
  - [x] 1.2 Create the full `ollama_fleet/` directory tree with `__init__.py` files
    - Create all subdirectories: `orchestrator/`, `scheduler/`, `agents/`, `ollama/`, `tools/`, `validation/`, `memory/`, `workspace/`, `ui/`, `db/migrations/`
    - Add `__init__.py` to each package directory
    - _Requirements: 14.1_

- [x] 2. Implement the configuration system
  - [x] 2.1 Implement `ollama_fleet/config.py` with all Pydantic settings models
    - Define `OllamaConfig`, `SchedulerConfig`, `MemoryConfig`, `WorkspaceConfig`, `UIConfig`, `ToolsConfig`, `FleetSettings`
    - Implement `FleetSettings.from_toml(path)` using `tomllib`/`tomli`
    - Apply all field constraints: `timeout` in [300, 3600], `retry_limit` in [1, 10], `max_concurrent_tasks` in [1, 32], `max_context_tokens` in [1024, 131072], `refresh_rate` in [0.1, 10.0], `command_timeout` in [1, 3600]
    - Read config path from `OLLAMA_FLEET_CONFIG` env var, fall back to `./config.toml`
    - On validation error: write field name + expected range to stderr, exit non-zero
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  - [x] 2.2 Write property test for config validation error completeness
    - **Property 23: Config Validation Error Completeness**
    - **Validates: Requirements 12.3, 12.4**
    - Use `st.fixed_dictionaries(...)` with out-of-range values for each field
    - Assert stderr contains field name and expected range; assert exit code is non-zero

- [x] 3. Implement the database layer
  - [x] 3.1 Implement `ollama_fleet/db/database.py` with SQLite connection management
    - Use `aiosqlite` for async access
    - Implement `Database.connect()`, `Database.close()`, `Database.execute()`, `Database.fetchall()`
    - Implement migration runner: scan `db/migrations/` for `*.sql` files in lexicographic order, execute each idempotently using a `schema_migrations` tracking table
    - _Requirements: 2.1, 10.4_
  - [x] 3.2 Write `db/migrations/001_initial.sql` with `jobs` and `tasks` tables
    - Create `jobs` table with all columns including `version` and `CHECK` constraints on `state`
    - Create `tasks` table with all columns including `version`, `dependencies` (JSON), `CHECK` constraints on `state` and `agent_type`
    - Create all indexes: `idx_jobs_state`, `idx_tasks_job_id`, `idx_tasks_state`, `idx_tasks_job_state`
    - Create `escalations` table with all columns and `idx_escalations_job_id`
    - _Requirements: 2.1, 13.2_
  - [x] 3.3 Write property test for task state persistence round-trip
    - **Property 5: Task State Persistence Round-Trip**
    - **Validates: Requirements 2.1**
    - Use `st.sampled_from(['pending','running','completed','failed','blocked','cancelled'])`
    - Write task to SQLite, read back, assert state is identical

- [ ] 4. Implement the Ollama client
  - [x] 4.1 Implement `ollama_fleet/ollama/client.py` with async httpx streaming
    - Define `OllamaConnectionError`, `OllamaHTTPError` (with `status_code` and `body`), `OllamaTimeoutError`
    - Implement `OllamaClient.generate(model, prompt, timeout)` using `httpx.AsyncClient.stream()`
    - Accumulate streamed line-delimited JSON chunks; stop when `"done": true`
    - Map connection errors → `OllamaConnectionError`, 4xx/5xx → `OllamaHTTPError`, timeout → `OllamaTimeoutError`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.6_
  - [ ]* 4.2 Write property test for streaming accumulation correctness
    - **Property 12: Streaming Accumulation Correctness**
    - **Validates: Requirements 4.3**
    - Use `st.text()` + random partition into N chunks; mock `httpx` stream
    - Assert accumulated result equals concatenation of all `response` fields before `"done": true`
  - [ ]* 4.3 Write property test for HTTP error typed exception mapping
    - **Property 13: HTTP Error Typed Exception Mapping**
    - **Validates: Requirements 4.4**
    - Use `st.integers(min_value=400, max_value=599)` for status codes
    - Assert `OllamaHTTPError` raised with correct `status_code`; assert `OllamaConnectionError` for connection failures

- [ ] 5. Implement Pydantic agent output schemas
  - [x] 5.1 Implement `ollama_fleet/agents/schemas.py` with all five agent output schemas
    - Define `PlannerTask`, `PlannerOutput` (tasks, milestones, architecture_notes)
    - Define `FileModification`, `CoderOutput` (file_modifications, summary, confidence_score in [0.0, 1.0])
    - Define `CriticIssue`, `CriticOutput` (approved, issues, overall_assessment)
    - Define `TestFailure`, `TesterOutput` (tests_passed, tests_failed, failures, ready_for_review)
    - Define `SynthesizerOutput` (summary, changelog, files_produced, next_steps)
    - Define `AgentOutput` union type and `AgentType` enum
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [ ]* 5.2 Write property test for agent output schema round-trip
    - **Property 14: Agent Output Schema Round-Trip**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
    - Use `st.builds()` for each of the five schema types
    - Assert `model_validate_json(model_dump_json(instance)) == instance` for all five types

- [ ] 6. Implement the Workspace Manager
  - [ ] 6.1 Implement `ollama_fleet/workspace/manager.py` with directory creation and atomic writes
    - Implement `WorkspaceManager.create_workspace(job_id)`: create `src/`, `tests/`, `logs/`, `agent_outputs/`, `validation/`, `summaries/`, `metadata/` subdirectories
    - Write `metadata/job.json` at creation time with job_id, goal, created_at, config
    - Implement `write_file(rel_path, content)` using write-to-temp-then-rename atomic pattern
    - Implement `_validate_path(rel_path)` to reject paths outside workspace root (return structured `PathTraversalError`)
    - Implement `append_execution_history(event)` writing to `logs/execution_history.jsonl`
    - On workspace creation failure: raise `WorkspaceCreationError`
    - On atomic write failure: delete temp file, raise `AtomicWriteError`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_
  - [ ]* 6.2 Write property test for atomic write integrity
    - **Property 22: Atomic Write Integrity**
    - **Validates: Requirements 10.3**
    - Use `st.text()` for file contents; inject mock rename failure
    - Assert target file contains new content on success, or pre-write content on failure; assert temp file is deleted on failure
  - [ ]* 6.3 Write property test for path traversal rejection
    - **Property 17: Path Traversal Rejection**
    - **Validates: Requirements 7.2, 10.5**
    - Use `st.text()` including `../` sequences
    - Assert any path resolving outside workspace root returns `error_type: "path_traversal"` and no filesystem operation is performed

- [ ] 7. Implement the Tool Runtime
  - [ ] 7.1 Implement `ollama_fleet/tools/file_tools.py` with `read_file`, `write_file`, `list_files`, `search_code`
    - `read_file`: validate path, return content or structured error (`file_not_found`, `permission_denied`)
    - `write_file`: validate path, delegate to `WorkspaceManager.write_file`
    - `list_files`: validate path, return directory listing
    - `search_code`: validate path, run regex search across workspace files
    - _Requirements: 7.1, 7.2, 7.10_
  - [ ] 7.2 Implement `ollama_fleet/tools/shell_tools.py` with `run_command` and `run_tests`
    - `run_command`: validate args against `SHELL_METACHARACTERS` set `{; & | > < $ \` ( ) { } [ ] * ? ! ~}`; execute subprocess with configurable timeout; return stdout, stderr, exit_code
    - On timeout: terminate subprocess, return `error_type: "timeout"` with `timeout_seconds`, partial stdout/stderr
    - `run_tests`: execute test suite, return structured result with pass_count, fail_count, per-failure details
    - _Requirements: 7.1, 7.3, 7.4, 7.6, 7.7_
  - [ ] 7.3 Implement `ollama_fleet/tools/git_tools.py` with `git_diff` and `git_commit`
    - `git_diff`: run `git diff`, return structured output
    - `git_commit`: stage and commit with provided message
    - Return `tool_unavailable` error when git is disabled in config
    - _Requirements: 7.1, 7.8, 7.9_
  - [ ] 7.4 Implement `ollama_fleet/tools/runtime.py` as the central tool dispatcher
    - Implement `ToolRuntime.invoke(tool_name, args, task_id)` dispatching to the correct tool function
    - Log every invocation: tool name, arguments, result, duration, task_id
    - _Requirements: 7.5_
  - [ ]* 7.5 Write property test for shell metacharacter rejection
    - **Property 18: Shell Metacharacter Rejection**
    - **Validates: Requirements 7.7**
    - Use `st.text()` seeded with characters from the metacharacter set
    - Assert `run_command` returns validation error and no subprocess is spawned

- [ ] 8. Implement the Task Scheduler (Phase 1 — basic queue, no dependency resolution)
  - [ ] 8.1 Implement `ollama_fleet/scheduler/task_scheduler.py` with state machine and atomic transitions
    - Implement `TaskScheduler.enqueue_tasks(tasks)`: insert task records with `pending` state
    - Implement `get_ready_tasks(job_id)`: return tasks in `pending` state
    - Implement `transition(task_id, new_state, reason)` using `BEGIN IMMEDIATE` + optimistic locking via `version` column: `UPDATE tasks SET state=?, version=version+1 WHERE task_id=? AND state=? AND version=?`; return `False` if 0 rows updated
    - Implement `increment_retry(task_id)`: increment `retry_count`, return new value
    - Implement `cancel_task(task_id)`: transition `pending`/`blocked` → `cancelled`
    - _Requirements: 2.1, 2.3, 2.4, 2.5, 2.6, 2.7_
  - [ ]* 8.2 Write property test for atomic dispatch exclusivity
    - **Property 8: Atomic Dispatch Exclusivity**
    - **Validates: Requirements 2.5**
    - Use `asyncio.gather` with N=10–50 concurrent coroutines all attempting to transition the same task to `running`
    - Assert exactly one succeeds; assert `version` incremented by exactly 1
  - [ ]* 8.3 Write property test for retry counter monotonicity
    - **Property 7: Retry Counter Monotonicity**
    - **Validates: Requirements 2.3**
    - Use `st.integers(min_value=0)` for initial retry counts
    - Assert counter increments by exactly 1 on each failure; assert counter never decreases; assert counter never exceeds `retry_limit`
  - [ ]* 8.4 Write property test for dispatch record completeness
    - **Property 9: Dispatch Record Completeness**
    - **Validates: Requirements 2.6**
    - Use `st.sampled_from(AgentType)` for all agent types
    - Assert `dispatched_at` is non-null and `agent_type` is correct immediately after dispatch

- [ ] 9. Implement the Agent Executor and Phase 1 agent prompt builders
  - [ ] 9.1 Implement `ollama_fleet/agents/executor.py` with prompt assembly, retry, timeout, and logging
    - Implement `AgentExecutor.execute(task, agent_type, extra_context)` following the 8-step execution flow
    - Assemble system + user messages; call `OllamaClient.generate()` with configured timeout
    - On `PydanticValidationError`: build error-correction prompt including validation error details, retry up to `retry_limit` (1–5); log raw response and error before each retry
    - On `OllamaTimeoutError`: mark task failed with `reason=invocation_timeout`; do NOT increment schema-validation retry counter
    - Log full prompt, raw response, parsed output, and duration for every call
    - _Requirements: 3.1, 3.2, 3.3, 3.6, 3.7, 3.8, 5.6_
  - [ ]* 9.2 Write property test for schema validation retry with logging
    - **Property 10: Schema Validation Retry with Logging**
    - **Validates: Requirements 3.3, 5.6**
    - Use `st.text()` + `st.binary()` to generate malformed JSON responses
    - Assert raw response and validation error are logged; assert retry prompt includes validation error; assert retry counter increments by exactly 1 per schema failure
  - [ ]* 9.3 Write property test for timeout independence from schema retries
    - **Property 11: Timeout Independence from Schema Retries**
    - **Validates: Requirements 3.8**
    - Use `st.lists(st.sampled_from(['timeout','schema']))` for mixed failure sequences
    - Assert schema-validation retry counter is incremented only by schema failures, not by timeouts
  - [ ] 9.4 Implement `ollama_fleet/agents/planner.py` — Planner_Agent prompt builder
    - Build system prompt defining the Planner role and output schema
    - Build user prompt injecting the job goal and `architecture_notes` context
    - Return assembled messages dict for `AgentExecutor`
    - _Requirements: 1.3, 3.4, 5.1_
  - [ ] 9.5 Implement `ollama_fleet/agents/coder.py` — Coding_Agent prompt builder
    - Build system prompt defining the Coder role and `CoderOutput` schema
    - Build user prompt injecting task description, active context files, and episodic summaries
    - _Requirements: 3.4, 5.2_
  - [ ] 9.6 Implement `ollama_fleet/agents/tester.py` — Tester_Agent prompt builder
    - Build system prompt defining the Tester role and `TesterOutput` schema
    - Build user prompt injecting workspace state and test results
    - _Requirements: 3.4, 5.4_

- [ ] 10. Implement the Phase 1 Orchestrator and job lifecycle
  - [ ] 10.1 Implement `ollama_fleet/orchestrator/job_manager.py` with job CRUD and state transitions
    - Implement `JobManager.create_job(goal, config)`: generate unique job_id (UUID4), persist to `jobs` table, return job_id
    - Implement `get_job(job_id)`, `update_job_state(job_id, new_state)`, `list_jobs_by_state(state)`
    - _Requirements: 1.1, 1.2_
  - [ ] 10.2 Write property test for job ID uniqueness
    - **Property 1: Job ID Uniqueness**
    - **Validates: Requirements 1.2**
    - Use `st.lists(st.text(), min_size=2)` for goal strings
    - Submit N jobs; assert all returned job_ids are distinct
  - [ ] 10.3 Implement `ollama_fleet/orchestrator/orchestrator.py` — Phase 1 dispatch loop
    - Implement `submit_job(goal, config)`: create workspace, persist job, invoke Planner_Agent, enqueue tasks, start dispatch loop
    - Implement `dispatch_loop(job_id)`: poll `get_ready_tasks` at ≤5s intervals; dispatch each ready task via `AgentExecutor`; check terminal condition
    - Implement `cancel_job(job_id)`, `pause_job(job_id)`, `resume_job(job_id)`
    - Implement `_dispatch_task(task)`: route to correct agent type; apply file modifications via `WorkspaceManager`; transition task state
    - On workspace creation failure: transition job to `failed`, write `metadata/job.json`
    - On Planner_Agent failure after retry_limit: transition job to `failed`
    - Phase 1: no Critic_Agent or Synthesizer_Agent; import them conditionally with `CRITIC_AVAILABLE` guard
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.9, 3.5, 14.1, 14.6_
  - [ ] 10.4 Write property test for job terminal state convergence
    - **Property 2: Job Terminal State Convergence**
    - **Validates: Requirements 1.5**
    - Use `st.lists(st.sampled_from(['completed','failed','cancelled']))` for task state sets
    - Assert job transitions to terminal state when all tasks are terminal

- [ ] 11. Implement the Phase 1 Terminal UI
  - [ ] 11.1 Implement `ollama_fleet/ui/event_bus.py` with async publish/subscribe
    - Implement `UIEventBus.publish(event)` and `subscribe(handler)`
    - Define event types: `TaskStateChanged`, `AgentStarted`, `AgentCompleted`, `ValidationResult`, `EscalationAdded`, `JobStateChanged`
    - _Requirements: 11.1, 11.2_
  - [ ] 11.2 Implement `ollama_fleet/ui/dashboard.py` with Textual layout and keyboard shortcuts
    - Implement main `FleetDashboard` Textual app with the 5-panel layout: header (job name, progress %), task queue panel, agent activity panel, logs panel, idle state panel
    - Subscribe to `UIEventBus`; update task queue within 1 second of `TaskStateChanged` events
    - Display per-agent status: agent type, task title, invocation duration (1 decimal), model name
    - Implement keyboard shortcuts: `p` to pause, `q` to quit
    - Display idle state panel when no job is active (last completed job name + terminal status)
    - _Requirements: 11.1, 11.2, 11.3, 11.5, 11.7_
  - [ ] 11.3 Implement `ollama_fleet/main.py` as the CLI entry point
    - Parse CLI arguments: `submit <goal>`, `cancel <job_id>`, `status <job_id>`
    - Load `FleetSettings` from config; exit non-zero on validation error
    - Initialize `Database`, `WorkspaceManager`, `TaskScheduler`, `Orchestrator`, `UIEventBus`, `FleetDashboard`
    - Run `FleetDashboard` with `asyncio` event loop
    - _Requirements: 12.1, 12.3, 12.4, 12.5_

- [ ] 12. Phase 1 checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Verify a submitted job runs Planner → Coder → Tester, persists all state to SQLite, and displays progress in Terminal_UI


---

## Phase 2: Critique Loop + Validation + Synthesizer

- [ ] 13. Implement the Validation Layer
  - [ ] 13.1 Implement `ollama_fleet/validation/validator.py` with syntax checking and linting
    - Implement `ValidationLayer.validate(modified_files, workspace)` running the 5-step pipeline
    - Syntax check: `ast.parse()` for Python files; record `SyntaxValidationError` per file
    - Lint: invoke `ruff` subprocess on modified files; parse output into `LintIssue` list; if `ruff` binary not found, record `linter_unavailable` warning and proceed
    - Write results to `validation/validation_<ISO8601_UTC>.json`
    - On syntax failure: re-queue Coding_Agent task WITHOUT incrementing retry counter
    - Return `ValidationResult(syntax_ok, lint_results, linter_available, timestamp)`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  - [ ] 13.2 Write property test for syntax failure does not consume retry
    - **Property 19: Syntax Failure Does Not Consume Retry**
    - **Validates: Requirements 8.3**
    - Use `st.text()` to generate syntactically invalid Python
    - Assert `retry_count` is identical before and after the validation failure re-queue

- [ ] 14. Implement Critic and Synthesizer agents
  - [ ] 14.1 Implement `ollama_fleet/agents/critic.py` — Critic_Agent prompt builder
    - Build system prompt defining the Critic role and `CriticOutput` schema
    - Build user prompt injecting exact file paths and contents of all files modified by the preceding Coding_Agent (within Active_Context token limit), plus lint results from Validation_Layer
    - _Requirements: 3.4, 5.3, 6.6_
  - [ ] 14.2 Implement `ollama_fleet/agents/synthesizer.py` — Synthesizer_Agent prompt builder
    - Build system prompt defining the Synthesizer role and `SynthesizerOutput` schema
    - Build user prompt injecting job goal, completed task summaries, and files produced
    - _Requirements: 1.6, 3.4, 5.5_
  - [ ] 14.3 Extend `ollama_fleet/agents/schemas.py` with `CriticOutput` and `SynthesizerOutput`
    - Add `CriticIssue`, `CriticOutput`, `SynthesizerOutput` to schemas
    - Update `AgentOutput` union type
    - _Requirements: 5.3, 5.5_

- [ ] 15. Extend the Orchestrator with the critique/revision loop
  - [ ] 15.1 Implement `_handle_critic_output` in `orchestrator.py` with revision task creation
    - When `CriticOutput.approved == False`: create new Coding_Agent revision task injecting the complete issue list (all severities) into the revision prompt
    - When `CriticOutput.approved == True`: proceed to Tester_Agent task
    - Enforce `max_critique_revision_loops` per coding task (valid range 1–10, default 3)
    - When loop limit reached without approval: write escalation record to `metadata/escalations.json` and `escalations` SQLite table; mark task `failed`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [ ] 15.2 Write property test for critique revision issue injection
    - **Property 15: Critique Revision Issue Injection**
    - **Validates: Requirements 6.2**
    - Use `st.lists(st.builds(CriticIssue, ...))` for issue lists
    - Assert every issue from `CriticOutput` appears in the revision prompt; assert no issue is silently dropped
  - [ ] 15.3 Write property test for critique loop termination
    - **Property 16: Critique Loop Termination**
    - **Validates: Requirements 6.4**
    - Use `st.integers(min_value=1, max_value=10)` for `max_critique_revision_loops`
    - Assert Coding_Agent is invoked at most `max_critique_revision_loops` times when Critic always returns `approved: false`

- [ ] 16. Implement escalation writing and UI panels
  - [ ] 16.1 Implement `ollama_fleet/orchestrator/escalation.py` with escalation record writing
    - Implement `write_escalation(task_id, job_id, reason, retry_count)`: append to `metadata/escalations.json` JSON array and insert into `escalations` SQLite table
    - Escalation record fields: `task_id`, `job_id`, `reason`, `retry_count`, `timestamp` (ISO 8601)
    - Implement identical-output loop detection: compare current `CoderOutput.file_modifications` byte-for-byte with previous invocation; if identical, mark task `failed` without re-queuing
    - _Requirements: 13.1, 13.2_
  - [ ] 16.2 Write property test for escalation record field completeness
    - **Property 25: Escalation Record Field Completeness**
    - **Validates: Requirements 13.2**
    - Use `st.builds(Task, ...)` for escalated tasks
    - Assert all five fields (`task_id`, `job_id`, `reason`, `retry_count`, `timestamp`) are present and non-null in both `escalations.json` and the SQLite table
  - [ ] 16.3 Write property test for identical output loop detection
    - **Property 24: Identical Output Loop Detection**
    - **Validates: Requirements 13.1**
    - Use `st.builds(CoderOutput, ...)` for identical outputs
    - Assert task is marked `failed` (not re-queued) when `file_modifications` are byte-for-byte identical to the preceding invocation
  - [ ] 16.4 Implement `ollama_fleet/ui/panels.py` with validation results and escalation panels
    - Implement `ValidationPanel`: display lint errors and test failures from `ValidationResult` events
    - Implement `EscalationPanel`: display `task_id`, `reason`, `timestamp`; persist until user presses `d` to dismiss
    - Implement critique loop iteration counter display for tasks in revision loop
    - Wire panels into `FleetDashboard` via `UIEventBus`
    - _Requirements: 11.4, 11.6, 13.3_
  - [ ] 16.5 Write `db/migrations/002_episodic_memory.sql` with episodic memory table
    - Create `episodic_memory` table with all columns and indexes
    - Migration must be idempotent and use `CREATE TABLE IF NOT EXISTS`
    - _Requirements: 9.2, 14.5_

- [ ] 17. Phase 2 checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Verify full Planner → Coder → Critic → Revision → Tester → Synthesizer pipeline with validation


---

## Phase 3: Crash Resilience + Dependency Tracking + Episodic Memory

- [ ] 18. Implement crash recovery in the Orchestrator
  - [ ] 18.1 Implement crash recovery startup logic in `orchestrator.py`
    - On startup: query SQLite for jobs in `running` state
    - For each such job: re-queue all tasks in `running` state as `pending` (reset `dispatched_at`); resume dispatch loop
    - Tasks already in terminal state must remain unchanged
    - _Requirements: 1.7_
  - [ ] 18.2 Write property test for crash recovery consistency
    - **Property 3: Crash Recovery Consistency**
    - **Validates: Requirements 1.7**
    - Use `st.fixed_dictionaries(...)` with running tasks at simulated crash time
    - Assert all `running` tasks become `pending` after restart; assert terminal tasks remain unchanged

- [ ] 19. Implement dependency resolution in the Task Scheduler
  - [ ] 19.1 Implement `ollama_fleet/scheduler/dependency_resolver.py` with DAG traversal
    - Implement `DependencyResolver.resolve(job_id)`: scan `blocked` tasks; for each, check if all dependencies are `completed`; if so, transition to `pending`
    - Implement cascade failure: when any dependency reaches `failed` or `cancelled`, transition all `blocked` dependents to `failed` with blocking dependency ID as `failure_reason`
    - _Requirements: 2.2, 2.8_
  - [ ] 19.2 Wire `DependencyResolver` into `TaskScheduler.resolve_dependencies(job_id)` and call it from the Orchestrator dispatch loop
    - Call `resolve_dependencies` on each dispatch loop iteration before `get_ready_tasks`
    - _Requirements: 2.2_
  - [ ] 19.3 Write property test for dependency unblocking invariant
    - **Property 6: Dependency Unblocking Invariant**
    - **Validates: Requirements 2.2, 2.8**
    - Use a custom `st.composite` DAG generator to produce random dependency graphs
    - Assert `blocked` task transitions to `pending` when all deps complete; assert `blocked` task transitions to `failed` when any dep fails/cancels with correct `failure_reason`
  - [ ] 19.4 Write property test for cancellation completeness
    - **Property 4: Cancellation Completeness**
    - **Validates: Requirements 1.8**
    - Use `st.lists(st.sampled_from(ALL_STATES))` for mixed task state distributions
    - Assert all non-terminal tasks transition to `cancelled`; assert terminal tasks remain unchanged

- [ ] 20. Implement the Memory System (Phase 3 — episodic memory)
  - [ ] 20.1 Implement `ollama_fleet/memory/episodic.py` with episodic memory CRUD
    - Implement `EpisodicMemory.save(entry: EpisodicEntry)`: insert into `episodic_memory` table
    - Implement `get_recent(job_id, n)`: return the most recent N entries ordered by timestamp descending
    - `EpisodicEntry` fields: `agent_type`, `task_id`, `outcome`, `files_modified`, `summary_text`, `timestamp`
    - _Requirements: 9.2, 9.5_
  - [ ] 20.2 Implement `ollama_fleet/memory/memory_system.py` with Active Context assembly
    - Implement `MemorySystem.assemble_context(task, job_id)`: build `ActiveContext` following the 6-step assembly process
    - Include task description (always), referenced files, most recent N episodic summaries (default N=5)
    - Estimate token count as `len(text) / 4`
    - Implement `_truncate_to_budget(context, max_tokens)`: remove oldest file contents first, then oldest episodic summaries, until under budget
    - Never inject full file tree
    - Implement `save_episodic(entry)` delegating to `EpisodicMemory`
    - _Requirements: 9.1, 9.3, 9.4, 9.6_
  - [ ] 20.3 Write property test for active context episodic window
    - **Property 20: Active Context Episodic Window**
    - **Validates: Requirements 9.1**
    - Use `st.integers(min_value=0, max_value=20)` × 2 for history size M and window N
    - Assert assembled context contains exactly `min(M, N)` episodic summaries, always the most recent N
  - [ ] 20.4 Write property test for active context token budget truncation order
    - **Property 21: Active Context Token Budget Truncation Order**
    - **Validates: Requirements 9.3**
    - Use `st.lists(st.builds(FileContent, ...))` for context compositions exceeding the token budget
    - Assert file contents are removed before episodic summaries; assert oldest files are removed first

- [ ] 21. Implement stall detection in the Orchestrator
  - [ ] 21.1 Implement `_check_stall` background coroutine in `orchestrator.py`
    - Run every 60 seconds; compare `now - last_task_state_transition_timestamp` against `stall_timeout` (default 600s)
    - On stall: write escalation record to `metadata/escalations.json` and `escalations` table; mark job `failed`
    - _Requirements: 13.4_

- [ ] 22. Phase 3 checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Verify jobs resume correctly after simulated crash; dependency-ordered task graphs execute correctly; episodic memory is injected into agent prompts


---

## Phase 4: Advanced Context + Confidence Routing + Long-Term Memory

- [ ] 23. Implement advanced context compression and long-term memory
  - [ ] 23.1 Extend `memory_system.py` with Phase 4 context compression
    - When raw context exceeds `max_context_tokens`, apply summarization to reduce to ≤50% of the limit
    - Implement summarization by invoking the `summarizer_model` via `OllamaClient` to produce condensed summaries of truncated file contents
    - _Requirements: 14.4_
  - [ ] 23.2 Write `db/migrations/003_long_term_memory.sql` with long-term memory table
    - Create `long_term_memory` table with all columns and `idx_ltm_job_id` index
    - Migration must be idempotent
    - _Requirements: 9.5, 14.5_
  - [ ] 23.3 Implement `ollama_fleet/memory/long_term.py` with searchable long-term memory
    - Implement `LongTermMemory.save(entry)`: insert into `long_term_memory` table
    - Implement `search(job_id, query)`: case-insensitive substring search via `WHERE summary_text LIKE '%query%' COLLATE NOCASE`
    - Implement `MemorySystem.search_long_term(job_id, query)` delegating to `LongTermMemory`
    - _Requirements: 9.5_

- [ ] 24. Implement confidence-score-based routing
  - [ ] 24.1 Implement confidence-score routing in `orchestrator.py`
    - After each `CoderOutput`: if `confidence_score < threshold` (default 0.4), route task through an additional Critic_Agent review before advancing to Tester_Agent
    - This routing applies regardless of whether the task was already scheduled for critic review
    - Subject to the loop rules in Requirement 6
    - _Requirements: 13.5_
  - [ ] 24.2 Write property test for low-confidence routing
    - **Property 26: Low-Confidence Routing**
    - **Validates: Requirements 13.5**
    - Use `st.floats(min_value=0.0, max_value=0.39)` for confidence scores below threshold
    - Assert Critic_Agent review is always triggered before Tester_Agent for any `confidence_score < threshold`

- [ ] 25. Verify migration backward compatibility
  - [ ] 25.1 Write and run migration compatibility tests
    - Test that all three migration scripts are idempotent (safe to run twice)
    - Test that existing job/task records created under migration 001 are readable after applying migrations 002 and 003
    - Use `ALTER TABLE ... ADD COLUMN` with defaults; never drop columns
    - _Requirements: 14.5_

- [ ] 26. Phase 4 checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Verify active context stays within token budget via summarization; low-confidence outputs route through additional critic review; long-term memory is searchable


---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability
- Checkpoints at the end of each phase ensure incremental validation before proceeding
- Property tests validate universal correctness guarantees using Hypothesis; unit tests validate specific examples and edge cases
- Phase N+1 components are imported conditionally using `try/except ImportError` guards to satisfy Requirement 14.6
- All migrations must be idempotent and use `CREATE TABLE IF NOT EXISTS` / `ALTER TABLE ... ADD COLUMN` patterns
- The `version` column on `jobs` and `tasks` tables enables optimistic locking for atomic state transitions (Property 8)
- All file writes use the write-to-temp-then-rename atomic pattern (Property 22)
- Token estimation uses `len(text) / 4` characters-per-token approximation throughout

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "3.1", "3.2"] },
    { "id": 2, "tasks": ["2.2", "3.3", "4.1", "5.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "5.2", "6.1"] },
    { "id": 4, "tasks": ["6.2", "6.3", "7.1", "7.2", "7.3"] },
    { "id": 5, "tasks": ["7.4", "7.5", "8.1"] },
    { "id": 6, "tasks": ["8.2", "8.3", "8.4", "9.1"] },
    { "id": 7, "tasks": ["9.2", "9.3", "9.4", "9.5", "9.6"] },
    { "id": 8, "tasks": ["10.1", "11.1"] },
    { "id": 9, "tasks": ["10.2", "10.3", "11.2"] },
    { "id": 10, "tasks": ["10.4", "11.3"] },
    { "id": 11, "tasks": ["13.1"] },
    { "id": 12, "tasks": ["13.2", "14.1", "14.2"] },
    { "id": 13, "tasks": ["14.3"] },
    { "id": 14, "tasks": ["15.1"] },
    { "id": 15, "tasks": ["15.2", "15.3", "16.1"] },
    { "id": 16, "tasks": ["16.2", "16.3", "16.4", "16.5"] },
    { "id": 17, "tasks": ["18.1"] },
    { "id": 18, "tasks": ["18.2", "19.1"] },
    { "id": 19, "tasks": ["19.2"] },
    { "id": 20, "tasks": ["19.3", "19.4", "20.1"] },
    { "id": 21, "tasks": ["20.2"] },
    { "id": 22, "tasks": ["20.3", "20.4", "21.1"] },
    { "id": 23, "tasks": ["23.1", "23.2"] },
    { "id": 24, "tasks": ["23.3"] },
    { "id": 25, "tasks": ["24.1"] },
    { "id": 26, "tasks": ["24.2", "25.1"] }
  ]
}
```
