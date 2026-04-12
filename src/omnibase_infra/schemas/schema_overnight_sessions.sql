-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- Overnight Sessions Projection Schema
-- Ticket: OMN-8455 (W2.8: overnight_sessions migration + node_projection_overnight)
-- Version: 1.0.0
--
-- Design Notes:
--   - overnight_sessions: one row per executor session (keyed on correlation_id)
--   - overnight_session_phases: normalized child rows (one per phase per session)
--   - Normalized phase table replaces JSONB array — enables efficient per-phase queries
--   - All UPSERT paths use ON CONFLICT DO NOTHING or DO UPDATE for idempotency
--   - Schema is idempotent (IF NOT EXISTS used throughout)
--
-- Related:
--   - OMN-8455: W2.8 overnight_sessions migration + node_projection_overnight
--   - executor-build-loop-architecture.md §7
--   - Topics: onex.evt.omnimarket.overnight.phase-start.v1
--             onex.evt.omnimarket.overnight.phase-completed.v1
--             onex.evt.omnimarket.overnight.session-completed.v1

CREATE TABLE IF NOT EXISTS overnight_sessions (
    session_id           TEXT PRIMARY KEY,             -- correlation_id from executor
    session_start_ts     TIMESTAMPTZ NOT NULL,
    contract_path        TEXT,
    dry_run              BOOLEAN NOT NULL DEFAULT FALSE,

    -- Aggregate metrics (updated on session-completed)
    phases_run           TEXT[] NOT NULL DEFAULT '{}',
    phases_failed        TEXT[] NOT NULL DEFAULT '{}',
    phases_skipped       TEXT[] NOT NULL DEFAULT '{}',
    dispatch_count       INT NOT NULL DEFAULT 0,

    -- Halt tracking
    halt_reason          TEXT,

    -- Terminal state
    session_status       TEXT NOT NULL DEFAULT 'in_progress',
    session_end_ts       TIMESTAMPTZ,

    -- Lifecycle
    accumulated_cost_usd NUMERIC(10,4) NOT NULL DEFAULT 0,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_session_status CHECK (
        session_status IN ('in_progress', 'completed', 'partial', 'failed')
    )
);

-- Normalized phase results table — avoids JSONB overhead, enables per-phase indexing
CREATE TABLE IF NOT EXISTS overnight_session_phases (
    id                   BIGSERIAL PRIMARY KEY,
    session_id           TEXT NOT NULL REFERENCES overnight_sessions(session_id) ON DELETE CASCADE,
    phase_name           TEXT NOT NULL,
    phase_status         TEXT NOT NULL,               -- success | failed | skipped
    duration_ms          INT NOT NULL DEFAULT 0,
    side_effect_summary  TEXT NOT NULL DEFAULT '',
    error_message        TEXT,
    sequence_number      INT NOT NULL DEFAULT 0,
    recorded_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_phase_status CHECK (
        phase_status IN ('success', 'failed', 'skipped')
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_session_phases_unique
    ON overnight_session_phases(session_id, phase_name, sequence_number);

CREATE INDEX IF NOT EXISTS ix_overnight_sessions_status
    ON overnight_sessions(session_status);

CREATE INDEX IF NOT EXISTS ix_overnight_sessions_start
    ON overnight_sessions(session_start_ts DESC);

CREATE INDEX IF NOT EXISTS ix_overnight_session_phases_session
    ON overnight_session_phases(session_id);
