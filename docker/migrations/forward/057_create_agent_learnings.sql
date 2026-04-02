-- Migration: Create agent_learnings table for Cross-Agent Memory Fabric
-- Ticket: OMN-7240
-- Version: 1.0.0
--
-- PURPOSE: Store structured learning records extracted from successful agent sessions.
-- Each record captures what an agent learned (resolution summary), what errors it
-- encountered, which files it touched, and the task context — enabling future agents
-- to query for relevant prior solutions.
--
-- IDEMPOTENCY: Uses IF NOT EXISTS for table and indexes.

CREATE TABLE IF NOT EXISTS agent_learnings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL,
    repo            VARCHAR(128) NOT NULL,
    file_paths_touched TEXT[] NOT NULL DEFAULT '{}',
    error_signatures TEXT[] NOT NULL DEFAULT '{}',
    resolution_summary TEXT NOT NULL,
    ticket_id       VARCHAR(64),
    task_type       VARCHAR(64) NOT NULL DEFAULT 'unknown',
    confidence      DOUBLE PRECISION NOT NULL DEFAULT 0.8 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for repo-scoped queries (passive injection)
CREATE INDEX IF NOT EXISTS idx_agent_learnings_repo_created
    ON agent_learnings (repo, created_at DESC);

-- Index for freshness-based retrieval
CREATE INDEX IF NOT EXISTS idx_agent_learnings_created_at
    ON agent_learnings (created_at DESC);

-- Session provenance is covered by the UNIQUE index below

-- GIN index for error signature array containment queries
CREATE INDEX IF NOT EXISTS idx_agent_learnings_error_signatures
    ON agent_learnings USING GIN (error_signatures);

-- Idempotency: prevent duplicate learnings from Kafka redelivery
-- session_id is the natural dedup key (one learning per successful session)
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_learnings_session_id_unique
    ON agent_learnings (session_id);
