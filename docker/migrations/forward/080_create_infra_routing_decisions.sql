-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration: 080_create_infra_routing_decisions
-- Description: Create infra_routing_decisions projection table (OMN-8692)
-- Created: 2026-05-22
--
-- Purpose: Projection table for onex.evt.omnibase-infra.routing-decided.v1 events
-- emitted by AdapterModelRouter (PluginLlm). Consumers subscribe via
-- ServiceInfraRoutingDecisionsConsumer and project into this table.
--
-- UPSERT key: correlation_id (UUID, idempotent re-delivery safe).
-- Idempotency: All statements use IF NOT EXISTS; safe to re-apply.

-- ============================================================================
-- TABLE: infra_routing_decisions
-- ============================================================================

CREATE TABLE IF NOT EXISTS infra_routing_decisions (
    id                      UUID        NOT NULL DEFAULT gen_random_uuid(),
    correlation_id          UUID,

    -- Provider selection
    selected_provider       TEXT        NOT NULL,
    selected_tier           TEXT        NOT NULL,
    selected_model          TEXT        NOT NULL DEFAULT '',

    -- Selection mode and fallback
    selection_mode          TEXT        NOT NULL DEFAULT 'round_robin',
    fallback_indicator      BOOLEAN     NOT NULL DEFAULT FALSE,
    is_fallback             BOOLEAN     NOT NULL DEFAULT FALSE,
    reason                  TEXT        NOT NULL DEFAULT '',

    -- Evaluation metadata
    candidates_evaluated    INTEGER     NOT NULL DEFAULT 0,
    candidate_providers     JSONB,

    -- Optional context
    task_type               TEXT,
    session_id              TEXT,
    latency_ms              NUMERIC(12, 3),

    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    projected_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_infra_routing_decisions PRIMARY KEY (id)
);

-- Partial unique index: correlation_id is optional but when present must be unique.
CREATE UNIQUE INDEX IF NOT EXISTS uq_infra_routing_decisions_correlation
    ON infra_routing_decisions (correlation_id)
    WHERE correlation_id IS NOT NULL;

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_ird_created_at
    ON infra_routing_decisions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ird_tier_provider
    ON infra_routing_decisions (selected_tier, selected_provider, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ird_selection_mode
    ON infra_routing_decisions (selection_mode, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ird_fallback
    ON infra_routing_decisions (fallback_indicator, created_at DESC)
    WHERE fallback_indicator = TRUE;

CREATE INDEX IF NOT EXISTS idx_ird_session_id
    ON infra_routing_decisions (session_id)
    WHERE session_id IS NOT NULL;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE infra_routing_decisions IS
    'Infra-layer routing decision projection from onex.evt.omnibase-infra.routing-decided.v1. '
    'Emitted by AdapterModelRouter (PluginLlm). UPSERT key: correlation_id. OMN-8692.';

COMMENT ON COLUMN infra_routing_decisions.selected_tier IS
    'Tier of chosen provider: local, cheap_cloud, or claude.';

COMMENT ON COLUMN infra_routing_decisions.selection_mode IS
    'How provider was selected: round_robin, failover, priority, or cost_optimized.';

COMMENT ON COLUMN infra_routing_decisions.fallback_indicator IS
    'True when at least one provider was attempted and failed before this one.';

COMMENT ON COLUMN infra_routing_decisions.candidates_evaluated IS
    'Number of providers examined before selecting this one.';
