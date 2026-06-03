# Ollama Fleet

Ollama Fleet is a small orchestration framework for running specialized agents that plan, generate, test, and refine code. It provides a pipeline for breaking a goal into tasks, generating file specifications, producing code, validating outputs, and iterating with critics and rewriters. The system includes a FastAPI-based web UI for live monitoring and a workspace-based persistence layer for capturing agent prompts and raw model responses.

## Architecture Overview

- Agents pipeline:
  - Planner: determines what files are needed, the purpose of each file, and file-level dependencies. (No implementation details.)
  - File Specification Agent: creates per-file specifications (imports, exports, functions, classes, behavior, edge cases).
  - File Generator (Coder): writes or modifies files according to the specification.
  - Validator & Tester: runs checks and unit tests against workspace code.
  - Critic & Rewriter: reviews outputs and applies fixes or improvements.

- Core components:
  - `ollama_fleet/agents/` — Agent implementations and prompt builders.
  - `ollama_fleet/agents/executor.py` — Orchestrates model calls and parses outputs.
  - `ollama_fleet/orchestrator/` — High-level job/task orchestration and workspace persistence.
  - `web_gui.py` — Single-page app served by the FastAPI server for real-time updates.
  - `workspace/` and per-job `agent_outputs/` — Workspace files and JSON logs for agent outputs.

## Data flow

1. User submits a job (goal + optional architecture notes).
2. Planner produces a numbered list of files (filename, purpose, dependencies).
3. File Specification Agent expands each planned file into a structured spec.
4. Coder generates or updates files in the job workspace according to specs.
5. Validator/Tester runs checks; Critic reviews and can request revisions.
6. All agent prompts, parsed outputs, and raw model responses are saved to `agent_outputs/*.json` for persistence and inspection.

## Running locally

Prerequisites: Python 3.11, virtualenv, Ollama server (or configured LLM client).

Quick start:

1. Create and activate a virtual environment:

	python3 -m venv .venv
	source .venv/bin/activate

2. Install dependencies:

	pip install -r requirements.txt

3. Start the FastAPI server (development):

	python -m ollama_fleet.main

4. Open the web UI at http://localhost:1020 to monitor jobs and view agent outputs.

## Workspace persistence

Each job has a workspace directory where the system writes files and an `agent_outputs/` folder. Agent outputs are timestamped JSON files containing the parsed output and the raw model response. This enables replay, debugging, and auditing after reloads.

## Tests

Run unit tests with:

	pytest -q

## Development notes

- Keep the Planner focused on structure only (filename, purpose, dependencies).
- Put implementation details in the File Specification Agent and the Coder.
- Avoid storing secrets in workspace files.

## Troubleshooting

- If the UI shows missing outputs, confirm the job workspace exists and check `agent_outputs/` for JSON files.
- For model connectivity issues, inspect the configured Ollama/LLM client and logs.

## Contributing

PRs welcome. Follow the existing code style and add tests for new behavior.
