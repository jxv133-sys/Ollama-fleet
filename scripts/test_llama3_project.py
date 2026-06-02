#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from ollama_fleet.config import FleetSettings
from ollama_fleet.db.database import Database
from ollama_fleet.orchestrator.orchestrator import Orchestrator

async def main() -> None:
    settings = FleetSettings()
    settings.ollama.planner_model = "llama3.2:1b"
    settings.ollama.coder_model = "llama3.2:1b"
    settings.ollama.critic_model = "llama3.2:1b"
    settings.ollama.tester_model = "llama3.2:1b"
    settings.ollama.summarizer_model = "llama3.2:1b"
    settings.ollama.timeout = 1200.0
    settings.workspace.base_path = "./test_workspaces_llama3"

    db_path = Path("test_llama3.db")
    if db_path.exists():
        db_path.unlink()

    db = Database(db_path)
    await db.connect()

    orch = Orchestrator(db, settings)
    goal = (
        "Build a small Python package named issuebot that fetches the latest open GitHub issues "
        "for a public repository, stores them in data/issues.csv, exposes a CLI with commands "
        "to fetch, list, filter by label, and summarize top issues, includes a GitHub API module, "
        "adds pytest tests, and writes a README.md explaining usage."
    )

    print("Starting job with goal:")
    print(goal)
    print("Using Ollama base URL:", settings.ollama.base_url)
    print("Using model:", settings.ollama.coder_model)

    job_id_holder: dict[str, str | None] = {"job_id": None}

    async def submit_job() -> str:
        job_id = await orch.submit_job(goal=goal, config={"source": "llama3_test_api"})
        job_id_holder["job_id"] = job_id
        print("Job submission finished, job_id=", job_id)
        return job_id

    submit_task = asyncio.create_task(submit_job())
    start = time.time()
    first_reported = False

    while True:
        await asyncio.sleep(15)
        elapsed = int(time.time() - start)
        job_id = job_id_holder["job_id"]
        if job_id is None:
            print(f"[{elapsed}s] waiting for job_id...")
        else:
            job = await orch.job_manager.get_job(job_id)
            state = job.state if job else "unknown"
            print(f"[{elapsed}s] job={job_id} state={state}")

        if submit_task.done():
            break

        if elapsed >= 600 and not first_reported:
            first_reported = True
            print("\n10-minute check reached.")
            if job_id is None:
                print("Job ID not available after 10 minutes.")
            else:
                job = await orch.job_manager.get_job(job_id)
                if job:
                    print("Job state at 10 min:", job.state)
                    print("Workspace path:", job.workspace_path)
                else:
                    print("Job record unavailable at 10 min.")
            print("Continuing to wait for completion or until 30 more minutes have elapsed.\n")

        if elapsed >= 2400:
            print("Reached 40 minutes total runtime; exiting monitor.")
            break

    if submit_task.done() and not submit_task.cancelled():
        try:
            job_id = submit_task.result()
        except Exception as exc:
            print("Job task failed with exception:", type(exc).__name__, exc)
            await db.close()
            return

        job = await orch.job_manager.get_job(job_id)
        print("\nFinal job status:")
        print("job_id=", job_id)
        print("state=", job.state if job else "unknown")
        if job:
            ws = Path(job.workspace_path)
            print("workspace=", ws)
            print("files:")
            for path in sorted(ws.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(ws)
                    print(" -", rel, path.stat().st_size)
    else:
        print("Submit task did not complete cleanly.")

    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
