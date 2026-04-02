-- Migration: Create agent identity tables for persistent agent platform
-- Ticket: OMN-7294
-- Version: 1.0.0
--
-- PURPOSE: Store agent identity records and session snapshots for cross-machine
-- sync and dashboard queries. YAML is the local-first source of truth;
-- Postgres is the sync target for portability and queryability.
--
-- IDEMPOTENCY: Uses IF NOT EXISTS for all objects.

CREATE TABLE IF NOT EXISTS agent_identities (
    agent_id        VARCHAR(64) PRIMARY KEY,
    display_name    VARCHAR(256) NOT NULL,
    status          VARCHAR(32) NOT NULL DEFAULT 'idle',
    current_binding JSONB,
    active_tickets  TEXT[] NOT NULL DEFAULT '{}',
    current_branch  VARCHAR(256),
    working_directory VARCHAR(1024),
    trust_profile_id VARCHAR(128),
    persona_profile_id VARCHAR(128),
    skill_inventory_id VARCHAR(128),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at  TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_session_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id        VARCHAR(64) NOT NULL REFERENCES agent_identities(agent_id),
    session_id      VARCHAR(128) NOT NULL,
    terminal_id     VARCHAR(128),
    machine         VARCHAR(256),
    working_directory VARCHAR(1024),
    git_branch      VARCHAR(256),
    current_ticket  VARCHAR(64),
    files_touched   TEXT[] NOT NULL DEFAULT '{}',
    errors_hit      TEXT[] NOT NULL DEFAULT '{}',
    last_tool_name  VARCHAR(128),
    last_tool_success BOOLEAN,
    last_tool_summary VARCHAR(500),
    session_outcome VARCHAR(32),
    snapshot_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_started_at TIMESTAMPTZ,
    session_ended_at TIMESTAMPTZ
);

-- Index for agent lookups
CREATE INDEX IF NOT EXISTS idx_agent_snapshots_agent_id
    ON agent_session_snapshots (agent_id, snapshot_at DESC);

-- Index for temporal queries ("what was CAIA doing yesterday?")
CREATE INDEX IF NOT EXISTS idx_agent_snapshots_temporal
    ON agent_session_snapshots (agent_id, session_started_at DESC);

-- Index for active session lookups
CREATE INDEX IF NOT EXISTS idx_agent_snapshots_session
    ON agent_session_snapshots (session_id);
