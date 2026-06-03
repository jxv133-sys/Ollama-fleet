-- Migration: 004_project_memory.sql
-- Adds project memory system for tracking file structure, exports, imports, and dependencies.
-- This replaces the episodic memory approach with a structured project state index.

-- ============================================================
-- project_memory table
-- ============================================================
-- Stores file-level metadata extracted from generated code.
-- Updated after every successful file generation.
CREATE TABLE IF NOT EXISTS project_memory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    file_type       TEXT NOT NULL,  -- 'python', 'json', 'yaml', etc.
    exports         TEXT NOT NULL,  -- JSON array of exported names (functions, classes)
    imports         TEXT NOT NULL,  -- JSON array of imported file paths
    classes         TEXT NOT NULL,  -- JSON array of class definitions
    functions       TEXT NOT NULL,  -- JSON array of function definitions
    dependencies    TEXT NOT NULL,  -- JSON array of dependency file paths
    last_updated    TEXT NOT NULL,  -- ISO 8601
    source_hash     TEXT NOT NULL,  -- SHA256 of file content
    UNIQUE(job_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_project_memory_job_id ON project_memory(job_id);
CREATE INDEX IF NOT EXISTS idx_project_memory_file_path ON project_memory(job_id, file_path);

-- ============================================================
-- project_interfaces table
-- ============================================================
-- Extracted public interfaces for faster context lookups.
-- Denormalized for quick retrieval without parsing source.
CREATE TABLE IF NOT EXISTS project_interfaces (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL,
    source_file     TEXT NOT NULL,
    interface_type  TEXT NOT NULL,  -- 'class', 'function'
    interface_name  TEXT NOT NULL,
    signature       TEXT NOT NULL,  -- Function signature or class definition
    docstring       TEXT,
    exports_from    TEXT NOT NULL,  -- Path of the exporting file
    UNIQUE(job_id, source_file, interface_name)
);

CREATE INDEX IF NOT EXISTS idx_project_interfaces_job_id ON project_interfaces(job_id);
CREATE INDEX IF NOT EXISTS idx_project_interfaces_source ON project_interfaces(job_id, source_file);

-- ============================================================
-- project_state table
-- ============================================================
-- High-level project state snapshot.
-- Updated after each action to track overall progress.
CREATE TABLE IF NOT EXISTS project_state (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL UNIQUE,
    total_files     INTEGER NOT NULL DEFAULT 0,
    generated_files INTEGER NOT NULL DEFAULT 0,
    validated_files INTEGER NOT NULL DEFAULT 0,
    failed_files    INTEGER NOT NULL DEFAULT 0,
    last_action     TEXT NOT NULL,  -- Last action taken by orchestrator
    last_action_time TEXT NOT NULL,
    metadata_json   TEXT NOT NULL,  -- Additional state flags
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);

CREATE INDEX IF NOT EXISTS idx_project_state_job_id ON project_state(job_id);
