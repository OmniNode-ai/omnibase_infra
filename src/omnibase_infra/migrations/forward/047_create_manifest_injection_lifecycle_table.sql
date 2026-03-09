-- Migration: 006_create_manifest_injection_lifecycle_table.sql
-- Purpose: Create audit table for manifest injection lifecycle events (OMN-1888 / OMN-2942)
-- Author: ONEX Infrastructure Team
-- Date: 2026-02-27
-- Ticket: OMN-2942
--
-- Design Decisions:
--
--   1. One table stores all three lifecycle stages (started / injected / failed).
--      The ``event_type`` column distinguishes the stage. This avoids three
--      separate narrow tables and allows efficient per-session queries.
--
--   2. Idempotency via (session_id, event_type) unique constraint.
--      Re-processing the same Kafka message is safe: ON CONFLICT DO NOTHING
--      ensures exactly-once semantics at the application level.
--      Note: a session can have both a ``manifest_injection_started`` row and
--      a ``manifest_injected`` OR ``manifest_injection_failed`` row — the
--      unique constraint is on (session_id, event_type), not session_id alone.
--
--   3. ``correlation_id`` is the distributed tracing join key to
--      ``injection_effectiveness`` and ``agent_routing_decisions``.
--
--   4. ``injection_success`` and ``injection_duration_ms`` are NULL for
--      ``manifest_injection_started`` events (outcome not yet known).
--
--   5. All timestamp fields are TIMESTAMPTZ (timezone-aware) for correct
--      cross-timezone comparisons in dashboard queries.

-- =============================================================================
-- TABLE: manifest_injection_lifecycle
-- =============================================================================

CREATE TABLE IF NOT EXISTS manifest_injection_lifecycle (
    -- Primary key
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Lifecycle stage
    event_type                  TEXT            NOT NULL,   -- 'manifest_injection_started' | 'manifest_injected' | 'manifest_injection_failed'

    -- Session identification
    entity_id                   UUID            NOT NULL,   -- Session UUID (partition key)
    session_id                  UUID            NOT NULL,   -- Session identifier as UUID

    -- Distributed tracing
    correlation_id              UUID            NOT NULL,   -- Join key to injection_effectiveness
    causation_id                UUID,                       -- Originating prompt event ID

    -- Timestamps
    emitted_at                  TIMESTAMPTZ     NOT NULL,   -- Hook emission timestamp
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Agent identification
    agent_label                 TEXT            NOT NULL,   -- Label/display name of the agent
    agent_domain                TEXT            NOT NULL DEFAULT '',

    -- Outcome (NULL for manifest_injection_started events)
    injection_success           BOOLEAN,
    injection_duration_ms       INTEGER,                    -- Time in milliseconds (nullable)

    -- Optional metadata
    routing_source              TEXT,                       -- 'explicit' | 'fuzzy_match' | 'fallback'
    agent_version               TEXT,
    yaml_path                   TEXT,

    -- Error tracking (populated for manifest_injection_failed events)
    error_message               TEXT,
    error_type                  TEXT,

    -- Idempotency: one row per (session, lifecycle stage)
    CONSTRAINT uk_manifest_injection_lifecycle_session_event
        UNIQUE (session_id, event_type),

    -- Enforce valid lifecycle stages
    CONSTRAINT chk_manifest_injection_lifecycle_event_type
        CHECK (event_type IN (
            'manifest_injection_started',
            'manifest_injected',
            'manifest_injection_failed'
        ))
);

-- =============================================================================
-- INDEXES: manifest_injection_lifecycle
-- =============================================================================

-- Time-range queries for dashboard
CREATE INDEX IF NOT EXISTS idx_manifest_injection_lifecycle_emitted_at
    ON manifest_injection_lifecycle (emitted_at DESC);

-- Join to injection_effectiveness via correlation_id
CREATE INDEX IF NOT EXISTS idx_manifest_injection_lifecycle_correlation_id
    ON manifest_injection_lifecycle (correlation_id);

-- Per-session lookup (all stages for a given session)
CREATE INDEX IF NOT EXISTS idx_manifest_injection_lifecycle_session_id
    ON manifest_injection_lifecycle (session_id);

-- Filter by lifecycle stage (e.g., count failures)
CREATE INDEX IF NOT EXISTS idx_manifest_injection_lifecycle_event_type
    ON manifest_injection_lifecycle (event_type, emitted_at DESC);

-- Per-agent aggregation (e.g., failure rate per agent)
CREATE INDEX IF NOT EXISTS idx_manifest_injection_lifecycle_agent_label
    ON manifest_injection_lifecycle (agent_label, event_type);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE manifest_injection_lifecycle IS
    'Audit trail for manifest injection lifecycle events emitted by omniclaude hooks. '
    'One row per (session_id, event_type) covering started / injected / failed stages. '
    'Provides the end-to-end injection effectiveness measurement loop required by OMN-1888. '
    'Populated by InjectionEffectivenessConsumer (OMN-2942).';

COMMENT ON COLUMN manifest_injection_lifecycle.event_type IS
    'Lifecycle stage: manifest_injection_started | manifest_injected | manifest_injection_failed';

COMMENT ON COLUMN manifest_injection_lifecycle.correlation_id IS
    'Join key to injection_effectiveness.correlation_id for end-to-end attribution';

COMMENT ON COLUMN manifest_injection_lifecycle.injection_success IS
    'NULL for manifest_injection_started events (outcome not yet known). '
    'TRUE for manifest_injected events. FALSE for manifest_injection_failed events.';

COMMENT ON COLUMN manifest_injection_lifecycle.injection_duration_ms IS
    'NULL for manifest_injection_started events. Non-null for completed/failed events.';
