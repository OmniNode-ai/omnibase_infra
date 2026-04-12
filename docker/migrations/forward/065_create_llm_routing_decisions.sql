-- Migration: 065_create_llm_routing_decisions
-- Description: Create llm_routing_decisions projection table (OMN-8540)
-- Created: 2026-04-12
--
-- Purpose: Projection table for onex.evt.omniclaude.llm-routing-decision.v1 events.
-- Consumed by omnidash's read-model consumer (omniclaude-projections.ts). Closes
-- the golden chain routing chain: the tail table was referenced but never migrated
-- into the omnibase_infra Postgres schema, causing 0/5 golden chain sweep results.
--
-- Schema derived from omnidash migrations 0006b_llm_routing_decisions.sql,
-- 0011a_llm_routing_decisions_correlation_id_uuid.sql, and
-- 0011b_llm_routing_decisions_nullable_fuzzy_agent.sql.
-- UPSERT key: correlation_id (UUID, idempotent re-delivery safe).
--
-- Idempotency: All statements use IF NOT EXISTS; safe to re-apply.
-- Rollback: See rollback/rollback_065_llm_routing_decisions.sql

-- ============================================================================
-- TABLE: llm_routing_decisions
-- ============================================================================
-- Projects LLM routing decision events from omniclaude Bifrost gateway.
-- UPSERT key: correlation_id (UUID).

CREATE TABLE IF NOT EXISTS llm_routing_decisions (
    id                      UUID        NOT NULL DEFAULT gen_random_uuid(),
    correlation_id          UUID        NOT NULL,
    session_id              TEXT,

    -- Routing decision
    llm_agent               TEXT        NOT NULL,
    fuzzy_agent             TEXT,
    agreement               BOOLEAN     NOT NULL DEFAULT FALSE,

    -- Confidence scores (0-1, NULL when not provided)
    llm_confidence          NUMERIC(5, 4) CHECK (llm_confidence IS NULL OR (llm_confidence >= 0 AND llm_confidence <= 1)),
    fuzzy_confidence        NUMERIC(5, 4) CHECK (fuzzy_confidence IS NULL OR (fuzzy_confidence >= 0 AND fuzzy_confidence <= 1)),

    -- Latency
    llm_latency_ms          INTEGER     NOT NULL DEFAULT 0,
    fuzzy_latency_ms        INTEGER     NOT NULL DEFAULT 0,

    -- Routing metadata
    used_fallback           BOOLEAN     NOT NULL DEFAULT FALSE,
    routing_prompt_version  TEXT        NOT NULL DEFAULT 'unknown',
    intent                  TEXT,
    model                   TEXT,
    cost_usd                NUMERIC(12, 8) CHECK (cost_usd IS NULL OR cost_usd >= 0),

    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    projected_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_llm_routing_decisions PRIMARY KEY (id),
    CONSTRAINT uq_llm_routing_decisions_correlation UNIQUE (correlation_id)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- idx_lrd_correlation_id omitted: UNIQUE (correlation_id) constraint already
-- creates an implicit unique index covering this column.

CREATE INDEX IF NOT EXISTS idx_lrd_created_at
    ON llm_routing_decisions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_lrd_agreement
    ON llm_routing_decisions (agreement, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_lrd_used_fallback
    ON llm_routing_decisions (used_fallback, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_lrd_prompt_version
    ON llm_routing_decisions (routing_prompt_version, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_lrd_agent_pair
    ON llm_routing_decisions (llm_agent, fuzzy_agent, created_at DESC)
    WHERE agreement = FALSE;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE llm_routing_decisions IS
    'LLM routing decision projection from onex.evt.omniclaude.llm-routing-decision.v1. '
    'UPSERT key: correlation_id (UUID). Golden chain routing tail table (OMN-8540).';

COMMENT ON COLUMN llm_routing_decisions.correlation_id IS
    'Distributed tracing correlation ID (UUID). UPSERT conflict target.';

COMMENT ON COLUMN llm_routing_decisions.llm_agent IS
    'Agent name selected by LLM routing.';

COMMENT ON COLUMN llm_routing_decisions.fuzzy_agent IS
    'Agent name selected by fuzzy-string routing (may differ from llm_agent). NULL on fallback path.';

COMMENT ON COLUMN llm_routing_decisions.agreement IS
    'True when LLM and fuzzy routing agreed on the same agent.';

COMMENT ON COLUMN llm_routing_decisions.used_fallback IS
    'True when the fuzzy matcher was used as fallback (LLM unavailable or timed out).';

COMMENT ON COLUMN llm_routing_decisions.routing_prompt_version IS
    'Version string of the routing prompt template used for this decision.';

COMMENT ON COLUMN llm_routing_decisions.cost_usd IS
    'Estimated LLM routing call cost in USD. NULL when not reported by upstream.';
