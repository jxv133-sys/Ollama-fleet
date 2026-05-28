-- Migration: 002_episodic_memory.sql
-- Adds episodic memory support for job history and task summaries.

CREATE TABLE IF NOT EXISTS episodic_memory (
    entry_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    agent_type      TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    files_modified  TEXT NOT NULL,
    summary_text    TEXT NOT NULL,
    timestamp       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodic_job_id ON episodic_memory(job_id);
CREATE INDEX IF NOT EXISTS idx_episodic_task_id ON episodic_memory(task_id);
