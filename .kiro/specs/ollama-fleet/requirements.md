# Requirements Document

## Introduction

Ollama Fleet is a production-grade local AI orchestration platform built in Python. It is an asynchronous, autonomous workflow engine designed for long-running AI-assisted software engineering tasks. Users submit a high-level software project goal (e.g., "Build a Flask dashboard with authentication"), and the system autonomously decomposes the goal into a plan, delegates subtasks to specialized AI agents backed by locally-hosted Ollama LLMs, executes tools (file I/O, test runners, shell commands), iterates through critique/revision loops, and delivers completed artifacts with progress summaries — all without human intervention during execution.

The system is explicitly NOT a chatbot. It is designed around asynchronous execution, persistent job queues, structured agent outputs, crash-resilient resumability, and a terminal-based CI/CD-style dashboard UI.

---

## Glossary

- **Orchestrator**: The central coordination component that manages the lifecycle of all jobs, tasks, and agents.
- **Job**: A top-level unit of work submitted by the user, representing a complete software project goal.
- **Task**: An atomic unit of work within a Job, assigned to a single agent for execution.
- **Agent**: A specialized AI-backed worker that receives a structured prompt and returns a structured JSON output. Agents are stateless; all context is injected by the Orchestrator.
- **Planner_Agent**: The agent responsible for decomposing a Job into Tasks, generating milestones, and identifying dependencies.
- **Coding_Agent**: The agent responsible for writing code, modifying files, and implementing features.
- **Critic_Agent**: The agent responsible for reviewing code outputs, identifying flaws, and producing structured critique reports.
- **Tester_Agent**: The agent responsible for running tests, analyzing failures, and generating debugging information.
- **Synthesizer_Agent**: The agent responsible for summarizing progress, generating changelogs, and producing user-facing reports.
- **Ollama_Client**: The component that communicates with the remote Ollama REST API server on the local network.
- **Task_Scheduler**: The component that manages the persistent task queue, task state transitions, dependency resolution, and retry logic.
- **Tool_Runtime**: The sandboxed execution environment for agent-invoked tools (file I/O, shell commands, test runners).
- **Validation_Layer**: The component that performs syntax checks, linting, and test execution on agent outputs before they are accepted.
- **Memory_System**: The component managing Active Context, Episodic Memory, and Long-Term Memory for each Job.
- **Workspace_Manager**: The component that manages the on-disk directory structure for each project's source files, logs, and artifacts.
- **Terminal_UI**: The Rich/Textual-based terminal dashboard displaying real-time job status, agent activity, task queue, and logs.
- **Active_Context**: The in-memory context window assembled for a single Task execution, containing only relevant information.
- **Episodic_Memory**: Persistent summaries of completed tasks and agent decisions stored per Job.
- **Long_Term_Memory**: Searchable, persistent project history stored in SQLite.
- **Structured_Output**: A strict JSON payload conforming to a Pydantic schema, returned by every Agent invocation.
- **Critique_Revision_Loop**: The iterative workflow where Critic_Agent output triggers a Coding_Agent revision until acceptance criteria are met or retry limits are exhausted.
- **Retry_Limit**: The maximum number of times a failed Task may be re-attempted before escalation.
- **Tool**: A discrete, validated, logged operation available to agents (e.g., `read_file`, `write_file`, `run_tests`).
- **Workspace**: The on-disk directory tree for a single Job, containing source files, metadata, logs, agent outputs, and execution history.

---

## Requirements

### Requirement 1: Job Submission and Lifecycle Management

**User Story:** As a user, I want to submit a high-level software project goal and have the system manage the full execution lifecycle autonomously, so that I receive completed software artifacts without manual intervention.

#### Acceptance Criteria

1. THE Orchestrator SHALL accept a job submission containing a natural-language project goal string and an optional configuration object specifying model assignments and workspace path.
2. WHEN a job is submitted, THE Orchestrator SHALL assign a unique Job ID, create a Workspace for the job, and persist the job record to the database before returning the Job ID to the caller.
3. WHEN a job is submitted, THE Orchestrator SHALL invoke the Planner_Agent to decompose the goal into an initial set of Tasks and persist those Tasks to the Task_Scheduler queue.
4. WHILE a job is in the `running` state, THE Orchestrator SHALL poll the Task_Scheduler for ready Tasks at intervals of no more than 5 seconds and dispatch them to the appropriate Agent.
5. WHEN all Tasks for a job reach a terminal state (`completed` or `failed`), THE Orchestrator SHALL transition the job to a terminal state.
6. WHEN a job transitions to a terminal state, THE Orchestrator SHALL invoke the Synthesizer_Agent to produce a final summary report.
7. IF a job is interrupted by a process crash, THEN THE Orchestrator SHALL resume the job from its last database-committed task/job state upon restart, re-queuing any Tasks that were in the `running` state at the time of the crash as `pending`.
8. WHEN a job cancellation is requested, THE Orchestrator SHALL transition the job and all non-terminal Tasks to the `cancelled` state.
9. IF the Planner_Agent fails to return a valid decomposition after the configured Retry_Limit, THEN THE Orchestrator SHALL transition the job to `failed` state and write a failure record to `metadata/job.json`.

---

### Requirement 2: Task Scheduling and State Management

**User Story:** As a system operator, I want tasks to be reliably queued, tracked, and retried with dependency awareness, so that the workflow progresses correctly even when individual tasks fail.

#### Acceptance Criteria

1. THE Task_Scheduler SHALL persist all Tasks to SQLite with the following states: `pending`, `running`, `completed`, `failed`, `blocked`, `cancelled`.
2. WHEN a Task's dependencies are all in the `completed` state (none in `failed` or `cancelled` state), THE Task_Scheduler SHALL transition that Task from `blocked` to `pending`.
3. WHEN a Task fails, THE Task_Scheduler SHALL increment the Task's retry counter and re-queue the Task as `pending` if the retry counter is below the configured Retry_Limit.
4. WHEN a Task's retry counter reaches the Retry_Limit, THE Task_Scheduler SHALL transition the Task to `failed` and write an escalation record to `metadata/escalations.json`, which the Orchestrator reads on its next poll.
5. THE Task_Scheduler SHALL support atomic state transitions to prevent race conditions during concurrent task dispatch.
6. WHEN a Task is dispatched to an Agent, THE Task_Scheduler SHALL record the dispatch timestamp and the assigned Agent type.
7. WHEN a task cancellation is requested, THE Task_Scheduler SHALL transition a `pending` or `blocked` Task to `cancelled` within 1 second.
8. WHEN a Task's dependency reaches `failed` or `cancelled` state, THE Task_Scheduler SHALL transition any `blocked` Tasks that depend on it to `failed` state and record the blocking dependency ID as the failure reason.

---

### Requirement 3: Agent Execution Framework

**User Story:** As a system architect, I want each agent to be a stateless, specialized worker that receives full context and returns structured JSON, so that agent behavior is predictable, testable, and composable.

#### Acceptance Criteria

1. THE Orchestrator SHALL inject all required context (task description, relevant file contents, episodic memory summaries, tool results) into each Agent invocation prompt; no Agent SHALL rely on conversational history.
2. WHEN an Agent is invoked, THE Agent_Executor SHALL send a structured prompt to the Ollama_Client and await a Structured_Output response conforming to the Agent's Pydantic output schema.
3. IF an Agent returns a response that fails Pydantic schema validation, THEN THE Agent_Executor SHALL retry the invocation with an error-correction prompt that includes the Pydantic validation error details from the failed attempt, up to a Retry_Limit bounded between 1 and 5 attempts, before marking the Task as `failed`.
4. THE system SHALL support the following specialized agents: Planner_Agent, Coding_Agent, Critic_Agent, Tester_Agent, and Synthesizer_Agent, each with a distinct Pydantic output schema.
5. THE Orchestrator SHALL route each Task to the Agent type specified in the Task record, using the model assignment configuration to select the correct Ollama model for that Agent type.
6. WHILE an Agent invocation is in progress, THE Agent_Executor SHALL enforce a configurable per-invocation timeout of no less than 300 seconds and no more than 3600 seconds to accommodate large model response times; WHEN the timeout fires, THE Agent_Executor SHALL mark the Task as `failed` with reason `invocation_timeout`.
7. THE Agent_Executor SHALL log the full prompt, raw response, parsed output, and invocation duration for every Agent call.
8. A timeout expiry SHALL be treated as a distinct failure mode from schema validation failure and SHALL NOT consume a schema-validation retry attempt.

---

### Requirement 4: Ollama Client Integration

**User Story:** As a developer, I want the system to communicate reliably with a remote Ollama server over the local network, so that I can leverage large locally-hosted models without cloud dependencies.

#### Acceptance Criteria

1. THE Ollama_Client SHALL communicate with the Ollama REST API using the base URL specified in the system configuration, defaulting to `http://localhost:11434`.
2. WHEN a generation request is made, THE Ollama_Client SHALL send an HTTP POST request to the `/api/generate` endpoint with the model name, prompt, and a `format: "json"` parameter to enforce structured output.
3. WHEN a streaming response is received, THE Ollama_Client SHALL accumulate all streamed JSON chunks until the Ollama API sends `"done": true`, then return the concatenated full response body to the Agent_Executor.
4. IF the Ollama server is unreachable (connection refused, DNS resolution failure, or TCP timeout) or returns an HTTP status code in the 4xx or 5xx range, THEN THE Ollama_Client SHALL raise a typed exception containing the status code and error body, which the Agent_Executor SHALL handle as a Task failure.
5. THE Ollama_Client SHALL support configuring separate model names for each Agent type via the system configuration file.
6. WHEN the per-request timeout is exceeded, THE Ollama_Client SHALL cancel the in-flight HTTP request and raise a typed `OllamaTimeoutError`.
7. THE system configuration SHALL support at minimum three model role assignments: `planner_model`, `coder_model`, and `summarizer_model`, with `critic_model` and `tester_model` defaulting to `coder_model` if not specified.

---

### Requirement 5: Structured Agent Output Schemas

**User Story:** As a developer, I want all agent outputs to conform to strict Pydantic schemas, so that downstream components can process agent results reliably without parsing freeform text.

#### Acceptance Criteria

1. THE Planner_Agent SHALL return a Structured_Output conforming to a schema containing: a list of Task objects (each with `task_id`, `title`, `description`, `agent_type`, `dependencies`, `priority` as an integer between 1 and 10 where 1 is highest priority), a list of milestone strings, and an `architecture_notes` string.
2. THE Coding_Agent SHALL return a Structured_Output conforming to a schema containing: a list of file modification objects (each with `file_path`, `operation` (`create`/`modify`/`delete`), `content` where `content` SHALL be an empty string when `operation` is `delete`), a `summary` string, and a `confidence_score` float between 0.0 and 1.0.
3. THE Critic_Agent SHALL return a Structured_Output conforming to a schema containing: an `approved` boolean, a list of issue objects (each with `file_path`, `line_number` where 0 indicates a file-level issue not tied to a specific line, `severity` (`critical`/`major`/`minor`), `description`, `suggested_fix`), and an `overall_assessment` string.
4. THE Tester_Agent SHALL return a Structured_Output conforming to a schema containing: a `tests_passed` integer, a `tests_failed` integer, a list of failure objects (each with `test_name`, `error_message`, `suggested_fix`), and a `ready_for_review` boolean.
5. THE Synthesizer_Agent SHALL return a Structured_Output conforming to a schema containing: a `summary` string, a `changelog` list of strings, a `files_produced` list of strings, and a `next_steps` list of strings.
6. IF any Agent returns JSON that does not conform to its schema, THEN THE Agent_Executor SHALL log the raw response and the validation error before initiating a retry using the error-correction prompt mechanism defined in Requirement 3.

---

### Requirement 6: Multi-Agent Critique/Revision Workflow

**User Story:** As a user, I want the system to automatically review and revise generated code through a structured critique loop, so that output quality improves without my manual intervention.

#### Acceptance Criteria

1. THE Orchestrator SHALL execute the following agent sequence: Planner_Agent → Coding_Agent → Critic_Agent → Coding_Agent (revision, only when Critic_Agent returns `approved: false`) → Tester_Agent → Synthesizer_Agent.
2. WHEN the Critic_Agent returns an output with `approved: false`, THE Orchestrator SHALL create a new Coding_Agent revision Task, injecting the complete issue list (regardless of severity) from the Critic_Agent's output into the revision prompt.
3. WHEN the Critic_Agent returns an output with `approved: true`, THE Orchestrator SHALL proceed to the Tester_Agent Task without creating a revision Task.
4. THE Orchestrator SHALL enforce a maximum Critique_Revision_Loop count per coding task, where one loop iteration is defined as one Critic_Agent invocation plus one resulting Coding_Agent revision; the maximum is configurable per job with a valid range of 1–10 and a default of 3 iterations.
5. WHEN the maximum Critique_Revision_Loop count is reached without `approved: true`, THE Orchestrator SHALL write an escalation record to `metadata/escalations.json` and mark the task as `failed`.
6. WHEN the Critic_Agent prompt is assembled, THE Critic_Agent prompt SHALL include the exact file paths and content of all files modified by the preceding Coding_Agent invocation, provided the injected file contents fit within the Active_Context token limit defined in Requirement 9.

---

### Requirement 7: Tool Runtime

**User Story:** As a developer, I want agents to invoke a controlled set of tools for file I/O, command execution, and test running, so that agents can interact with the project workspace safely and reproducibly.

#### Acceptance Criteria

1. THE Tool_Runtime SHALL provide the following tools: `read_file`, `write_file`, `list_files`, `search_code`, `run_command`, `run_tests`, `git_diff`, `git_commit`.
2. WHEN `write_file` is invoked, THE Tool_Runtime SHALL validate that the target path is within the Job's Workspace directory before writing; IF the path traverses outside the Workspace, THEN THE Tool_Runtime SHALL return a structured error with `error_type: "path_traversal"` and SHALL NOT perform any write.
3. WHEN `run_command` is invoked, THE Tool_Runtime SHALL execute the command in a subprocess with a configurable timeout (default 60 seconds) and return stdout, stderr, and exit code as a structured result.
4. IF `run_command` exceeds its timeout, THEN THE Tool_Runtime SHALL terminate the subprocess and return a timeout error result containing fields: `error_type: "timeout"`, `timeout_seconds` (the configured value), `stdout` (partial), `stderr` (partial).
5. THE Tool_Runtime SHALL log every tool invocation with: tool name, arguments, result, duration, and the Task ID that triggered the invocation.
6. WHEN `run_tests` is invoked, THE Tool_Runtime SHALL execute the project's test suite and return a structured result containing pass count, fail count, and individual test failure details with fields: `test_name`, `error_message`, `suggested_fix`.
7. IF `run_command` is invoked with arguments containing shell metacharacters from the set `; & | > < $ \` ( ) { } [ ] * ? ! ~`, THEN THE Tool_Runtime SHALL return a validation error without executing the command.
8. WHERE git integration is enabled in the job configuration, THE Tool_Runtime SHALL make `git_diff` and `git_commit` tools available.
9. WHERE git integration is disabled in the job configuration, THE Tool_Runtime SHALL return a `tool_unavailable` error for `git_diff` and `git_commit` tools.
10. IF `read_file` is invoked on a path that does not exist or is not readable, THEN THE Tool_Runtime SHALL return a structured error with `error_type: "file_not_found"` or `error_type: "permission_denied"` without raising an unhandled exception.

---

### Requirement 8: Validation Layer

**User Story:** As a system architect, I want all code outputs to be automatically validated before being accepted into the workspace, so that the revision loop is driven by objective quality signals rather than agent self-assessment.

#### Acceptance Criteria

1. WHEN a Coding_Agent Task completes, THE Validation_Layer SHALL automatically run syntax validation on all modified files before the Critic_Agent is invoked; IF the linter binary is not found, THE Validation_Layer SHALL record a `linter_unavailable` warning and proceed without lint results.
2. THE Validation_Layer SHALL run language-appropriate linting (e.g., `ruff` for Python files) on all modified files and include lint results in the Critic_Agent's input context.
3. WHEN syntax validation fails for any modified file, THE Validation_Layer SHALL mark the Coding_Agent Task as `failed` and re-queue it without invoking the Critic_Agent; a syntax failure SHALL NOT consume a Task retry attempt and the retry counter SHALL remain unchanged.
4. WHEN `run_tests` completes, THE Validation_Layer SHALL parse test runner output (e.g., `pytest`) into a structured result containing pass count, fail count, and per-failure details.
5. WHEN validation results are produced, THE Validation_Layer SHALL record all validation results in the Job's Workspace under a `validation/` subdirectory with a filename in the format `validation_<ISO8601_UTC>.json` (e.g., `validation_2024-01-15T10-30-00Z.json`).

---

### Requirement 9: Memory System

**User Story:** As a system architect, I want the system to maintain structured memory across task executions, so that agents receive relevant context without overflowing their context windows.

#### Acceptance Criteria

1. THE Memory_System SHALL maintain an Active_Context object per running Task, containing only the task description, files referenced in the Task description or its dependency chain, and the most recent N Episodic_Memory summaries (default N=5, configurable).
2. WHEN a Task completes, THE Memory_System SHALL generate an Episodic_Memory entry with the following fields: `agent_type`, `task_id`, `outcome` (`completed`/`failed`), `files_modified` (list of paths), `summary_text` (string), `timestamp` (ISO 8601), and persist it to SQLite.
3. IF the assembled Active_Context exceeds the configured maximum token count (default 8192 tokens, estimated at 4 characters per token), THEN THE Memory_System SHALL truncate content in the following priority order: file contents are truncated before episodic summaries, with oldest file content truncated first.
4. THE Memory_System SHALL select relevant files for Active_Context inclusion based on files referenced in the Task description or its dependency chain.
5. THE Long_Term_Memory SHALL store all Episodic_Memory entries in a SQLite table indexed by Job ID and Task ID, queryable by case-insensitive substring match against the `summary_text` field.
6. THE Memory_System SHALL NOT inject the full repository file tree into any Agent prompt; THE Memory_System SHALL inject only file contents explicitly selected for the Active_Context.

---

### Requirement 10: Workspace Manager

**User Story:** As a developer, I want each project to have an organized, isolated on-disk workspace, so that artifacts, logs, and source files are easy to locate and do not interfere across jobs.

#### Acceptance Criteria

1. WHEN a Job is created, THE Workspace_Manager SHALL create a directory structure under a configurable base path with the following subdirectories: `src/`, `tests/`, `logs/`, `agent_outputs/`, `validation/`, `summaries/`, `metadata/`.
2. THE Workspace_Manager SHALL write a `metadata/job.json` file at job creation time containing the Job ID, goal string, creation timestamp, and configuration.
3. WHEN an Agent produces file modifications, THE Workspace_Manager SHALL apply those modifications atomically: writing to a temporary file first, then renaming to the target path.
4. THE Workspace_Manager SHALL maintain an append-only `logs/execution_history.jsonl` file recording the following events with ISO 8601 timestamps: task state transitions, tool invocations (name and arguments), agent invocation start/end, validation results, and job state transitions.
5. THE Workspace_Manager SHALL prevent path traversal by rejecting any file operation whose resolved path falls outside the Job's Workspace root directory; rejection SHALL return a structured error indicating path traversal and the file operation SHALL NOT be performed.
6. IF the Workspace_Manager cannot create the workspace directory (base path not writable or path already exists as a file), THEN THE Orchestrator SHALL transition the job to `failed` state before invoking the Planner_Agent.
7. IF an atomic file write fails (temp file write error or rename failure), THEN THE Workspace_Manager SHALL delete the temp file if it exists and propagate a structured error to the caller.

---

### Requirement 11: Terminal UI Dashboard

**User Story:** As a user, I want a real-time terminal dashboard that shows the status of all active agents, the task queue, and recent logs, so that I can monitor autonomous execution without polling logs manually.

#### Acceptance Criteria

1. THE Terminal_UI SHALL display a live-updating dashboard (refreshing at the rate configured in `[ui].refresh_rate`) using Rich or Textual showing: active job name, overall job progress percentage (calculated as (completed_tasks + failed_tasks) / total_tasks × 100), current agent activity, task queue with states, and a scrolling log panel.
2. WHEN a Task transitions state, THE Terminal_UI SHALL update the task queue display within 1 second of the state change.
3. THE Terminal_UI SHALL display per-agent status indicators showing: agent type, current task title, invocation duration in seconds with 1 decimal place, and model name.
4. WHEN the Validation_Layer produces results, THE Terminal_UI SHALL display those results (lint errors, test failures) in a dedicated panel.
5. THE Terminal_UI SHALL provide the keyboard shortcut 'p' to pause job execution, which causes the Orchestrator to stop dispatching new Tasks after the currently running Tasks complete.
6. WHILE a Task is in a Critique_Revision_Loop, THE Terminal_UI SHALL display the current Critique_Revision_Loop iteration count for that Task.
7. WHEN no job is active, THE Terminal_UI SHALL display an idle state panel showing the last completed job name and its terminal status.

---

### Requirement 12: Configuration System

**User Story:** As a developer, I want all system behavior to be configurable via a single configuration file, so that I can adapt the platform to different hardware setups and project requirements without modifying source code.

#### Acceptance Criteria

1. THE system SHALL load configuration from a TOML file at a path specified by the `OLLAMA_FLEET_CONFIG` environment variable, falling back to `./config.toml` if the variable is not set.
2. THE configuration file SHALL support the following top-level sections: `[ollama]` (server URL, model assignments, timeout), `[scheduler]` (retry_limit: 1–10, max_concurrent_tasks: 1–32), `[memory]` (max_context_tokens: 1024–131072), `[workspace]` (base path), `[ui]` (refresh_rate: 0.1–10.0 seconds), `[tools]` (command_timeout: 1–3600 seconds, git enabled).
3. WHEN the configuration file is missing or contains invalid values, THE system SHALL write an error to stderr including the config file path, the invalid field name, and the expected value type or range, then exit with a non-zero status code before attempting any job execution.
4. WHEN the system starts up, THE system SHALL validate the configuration against a Pydantic settings schema and, IF a required field is missing, SHALL write an error to stderr including the config file path, the invalid field name, and the expected value type or range, then exit with a non-zero status code.
5. IF the `OLLAMA_FLEET_CONFIG` environment variable is set but the specified file does not exist, THEN THE system SHALL write an error to stderr including the missing path and exit with a non-zero status code.

---

### Requirement 13: Failure Detection and Escalation

**User Story:** As a user, I want the system to detect and handle pathological failure modes such as infinite loops and repeated identical outputs, so that the system does not waste compute resources on unrecoverable tasks.

#### Acceptance Criteria

1. IF a Coding_Agent produces an output where the complete set of file modification objects is byte-for-byte identical to its immediately preceding invocation on the same Task, THEN THE Orchestrator SHALL mark the Task as `failed` rather than re-queuing it.
2. IF a single Task has been retried more than the configured Retry_Limit, THEN THE Orchestrator SHALL escalate the Task by writing an escalation record to the `metadata/escalations.json` append-only JSON array in the Workspace; the escalation record SHALL contain fields: `task_id`, `job_id`, `reason` (string), `retry_count` (integer), `timestamp` (ISO 8601).
3. WHEN a Task is escalated, THE Terminal_UI SHALL display a dedicated escalation panel listing `task_id`, `reason`, and `timestamp`; the panel SHALL persist until the user presses 'd' to dismiss.
4. IF a Job has had no Task state transitions for longer than a configurable stall timeout (default 600 seconds), THEN THE Orchestrator SHALL escalate the Job by writing an escalation record to `metadata/escalations.json` with fields matching criterion 2.
5. IF a Coding_Agent returns a `confidence_score` below a configurable threshold (default 0.4), THEN THE Orchestrator SHALL automatically route the Task through an additional Critic_Agent review before proceeding; THE Task SHALL NOT advance to Tester_Agent until the additional Critic_Agent review completes, subject to the loop rules in Requirement 6.

---

### Requirement 14: Phased Implementation Strategy

**User Story:** As a developer, I want the system to be buildable in incremental phases, so that a working single-agent pipeline is available before full multi-agent orchestration is complete.

#### Acceptance Criteria

1. THE system SHALL execute a complete Planner_Agent → Coding_Agent → Tester_Agent workflow for a submitted job as a Phase 1 deliverable, persisting all state to SQLite and displaying progress in Terminal_UI, without Critic_Agent or Synthesizer_Agent components being required.
2. THE system SHALL be buildable as a Phase 2 deliverable adding: Critic_Agent, Critique_Revision_Loop, Validation_Layer, and Synthesizer_Agent on top of the Phase 1 foundation.
3. THE system SHALL be buildable as a Phase 3 deliverable adding: full crash-resilient resumability, Task_Scheduler with dependency tracking, and Episodic_Memory persistence on top of the Phase 2 foundation.
4. THE system SHALL be buildable as a Phase 4 deliverable adding: advanced context compression (Active_Context token count reduced to ≤ 50% of the configured max_context_tokens limit through summarization when the raw context exceeds the limit), confidence-score-based routing, and Long_Term_Memory search on top of the Phase 3 foundation.
5. WHEN transitioning between phases, THE system SHALL maintain backward compatibility with existing SQLite database schemas through migration scripts; a migration script SHALL complete without error and all existing job/task records SHALL be readable by the new schema version.
6. WHEN a Phase N+1 component (e.g., Critic_Agent) is absent from a Phase N build, THE system SHALL not raise import errors or runtime exceptions due to the missing component.
