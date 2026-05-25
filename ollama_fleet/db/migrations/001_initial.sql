-- Migration: 001_initial.sql
-- Creates the core jobs, tasks, and escalations tables.
-- Idempotent: uses CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS.

-- ============================================================
-- jobs table
-- ============================================================
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    goal            TEXT NOT NULL,
    state           TEXT NOT NULL CHECK(state IN ('submitted','running','completed','failed','cancelled')),
    created_at      TEXT NOT NULL,   -- ISO 8601
    updated_at      TEXT NOT NULL,
    config_json     TEXT NOT NULL,   -- serialized JobConfig
    workspace_path  TEXT NOT NULL,
    version         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);

-- ============================================================
-- tasks table
-- ============================================================
CREATE TABLE IF NOT EXISTS tasks (
    task_id         TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL REFERENCES jobs(job_id),
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    agent_type      TEXT NOT NULL CHECK(agent_type IN ('planner','coder','critic','tester','synthesizer')),
    state           TEXT NOT NULL CHECK(state IN ('pending','running','completed','failed','blocked','cancelled')),
    priority        INTEGER NOT NULL DEFAULT 5,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    dependencies    TEXT NOT NULL DEFAULT '[]',  -- JSON array of task_ids
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    dispatched_at   TEXT,
    completed_at    TEXT,
    failure_reason  TEXT,
    agent_output    TEXT,            -- serialized AgentOutput JSON
    version         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tasks_job_id ON tasks(job_id);
CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
CREATE INDEX IF NOT EXISTS idx_tasks_job_state ON tasks(job_id, state);

-- ============================================================
-- escalations table
-- ============================================================
CREATE TABLE IF NOT EXISTS escalations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT NOT NULL,
    job_id          TEXT NOT NULL,
    reason          TEXT NOT NULL,
    retry_count     INTEGER NOT NULL,
    timestamp       TEXT NOT NULL,   -- ISO 8601
    dismissed       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_escalations_job_id ON escalations(job_id);
