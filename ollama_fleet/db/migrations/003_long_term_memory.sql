-- Migration: 003_long_term_memory.sql
-- Adds long-term memory persistence for searchable summaries.

CREATE TABLE IF NOT EXISTS long_term_memory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL,
    summary_text    TEXT NOT NULL,
    source          TEXT NOT NULL,
    metadata_json   TEXT NOT NULL,
    inserted_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ltm_job_id ON long_term_memory(job_id);
