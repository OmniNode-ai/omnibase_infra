-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration 078: create delegation_events projection table (OMN-10526).
-- Consumed by HandlerProjectionDelegation from onex.evt.omniclaude.task-delegated.v1.

CREATE TABLE IF NOT EXISTS delegation_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id TEXT UNIQUE NOT NULL,
    session_id TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    task_type TEXT NOT NULL,
    delegated_to TEXT NOT NULL,
    model_name TEXT DEFAULT '',
    delegated_by TEXT,
    quality_gate_passed BOOLEAN DEFAULT false,
    quality_gates_checked JSONB,
    quality_gates_failed JSONB,
    cost_usd NUMERIC(18, 9) DEFAULT 0,
    cost_savings_usd NUMERIC(18, 9) DEFAULT 0,
    delegation_latency_ms INT,
    repo TEXT,
    is_shadow BOOLEAN DEFAULT false,
    llm_call_id TEXT,
    prompt_text TEXT,
    response_text TEXT,
    tokens_to_compliance INT NOT NULL DEFAULT 0,
    compliance_attempts INT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS delegation_shadow_comparisons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id TEXT UNIQUE NOT NULL,
    session_id TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    task_type TEXT NOT NULL,
    primary_agent TEXT NOT NULL,
    shadow_agent TEXT NOT NULL,
    divergence_detected BOOLEAN DEFAULT false,
    divergence_score NUMERIC(18, 9),
    primary_latency_ms INT,
    shadow_latency_ms INT,
    primary_cost_usd NUMERIC(18, 9),
    shadow_cost_usd NUMERIC(18, 9),
    divergence_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_delegation_events_session_id
    ON delegation_events (session_id);

CREATE INDEX IF NOT EXISTS idx_delegation_events_timestamp
    ON delegation_events (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_delegation_events_delegated_to
    ON delegation_events (delegated_to);

CREATE INDEX IF NOT EXISTS idx_delegation_shadow_comparisons_session_id
    ON delegation_shadow_comparisons (session_id);

CREATE INDEX IF NOT EXISTS idx_delegation_shadow_comparisons_timestamp
    ON delegation_shadow_comparisons (timestamp DESC);

COMMENT ON TABLE delegation_events IS
    'Projection of task-delegated events from onex.evt.omniclaude.task-delegated.v1. OMN-10526.';

UPDATE db_metadata
    SET schema_version = '078', updated_at = NOW()
    WHERE id = TRUE AND owner_service = 'omnibase_infra';
