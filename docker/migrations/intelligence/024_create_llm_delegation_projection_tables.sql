-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration: 024_create_llm_delegation_projection_tables
-- Description: Create projection tables for LLM delegation cost tracking.
--              Materialised from ModelLlmDelegationCompletedEvent Kafka events.
-- Ticket: OMN-11773 (epic OMN-11771)
--
-- Idempotency:
--   All statements use IF NOT EXISTS so re-applying this migration is safe.
--
-- Tables:
--   llm_delegation_call_log          — per-call log (idempotency key: call_id)
--   llm_delegation_daily_projection  — daily aggregate (unique: date, task_type, model_id, model_tier)

-- ---------------------------------------------------------------------------
-- Table 1: llm_delegation_call_log
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS llm_delegation_call_log (
    id                          UUID        NOT NULL DEFAULT gen_random_uuid(),

    -- Idempotency key from event request_id
    call_id                     TEXT        NOT NULL,
    correlation_id              TEXT        NOT NULL,
    task_id                     TEXT,
    attempt_number              INTEGER     NOT NULL DEFAULT 1,

    -- What was delegated
    task_type                   TEXT        NOT NULL,
    model_id                    TEXT        NOT NULL,
    model_tier                  TEXT        NOT NULL,
    endpoint_ref                TEXT        NOT NULL,
    input_hash                  TEXT        NOT NULL,
    output_hash                 TEXT        NOT NULL,

    -- Token accounting
    tokens_in                   INTEGER     NOT NULL,
    tokens_out                  INTEGER     NOT NULL,
    latency_ms                  INTEGER     NOT NULL,

    -- Cost accounting (NUMERIC for monetary precision)
    actual_cost_usd             NUMERIC(12, 6) NOT NULL,
    opus_equivalent_cost_usd    NUMERIC(12, 6) NOT NULL,
    savings_usd                 NUMERIC(12, 6) NOT NULL,

    -- Cost provenance
    usage_source                TEXT        NOT NULL,
    cost_basis                  TEXT        NOT NULL,
    pricing_manifest_version    TEXT        NOT NULL,
    pricing_observed_at         TIMESTAMPTZ,

    -- Quality tracking
    success                     BOOLEAN     NOT NULL,
    escalated_to                TEXT,
    quality_score               FLOAT8,
    escalation_reason           TEXT,
    failure_class               TEXT,

    -- Metadata
    session_id                  TEXT,
    repo_name                   TEXT,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_llm_delegation_call_log
        PRIMARY KEY (id),

    CONSTRAINT uq_llm_delegation_call_log_call_id
        UNIQUE (call_id),

    CONSTRAINT chk_llm_delegation_call_log_model_tier
        CHECK (model_tier IN ('free', 'local', 'frontier')),

    CONSTRAINT chk_llm_delegation_call_log_usage_source
        CHECK (usage_source IN ('measured', 'estimated', 'unknown')),

    CONSTRAINT chk_llm_delegation_call_log_cost_basis
        CHECK (cost_basis IN ('cloud_api_cost', 'zero_marginal_api_cost', 'unknown')),

    CONSTRAINT chk_llm_delegation_call_log_tokens_in
        CHECK (tokens_in >= 0),

    CONSTRAINT chk_llm_delegation_call_log_tokens_out
        CHECK (tokens_out >= 0),

    CONSTRAINT chk_llm_delegation_call_log_latency_ms
        CHECK (latency_ms >= 0)
);

CREATE INDEX IF NOT EXISTS idx_llm_delegation_call_log_date
    ON llm_delegation_call_log ((created_at::date));

CREATE INDEX IF NOT EXISTS idx_llm_delegation_call_log_task_type
    ON llm_delegation_call_log (task_type);

CREATE INDEX IF NOT EXISTS idx_llm_delegation_call_log_model_id
    ON llm_delegation_call_log (model_id);

CREATE INDEX IF NOT EXISTS idx_llm_delegation_call_log_created_at
    ON llm_delegation_call_log (created_at);

COMMENT ON TABLE llm_delegation_call_log IS
    'Per-call log for LLM delegation events. Idempotency key: call_id. OMN-11773.';

-- ---------------------------------------------------------------------------
-- Table 2: llm_delegation_daily_projection
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS llm_delegation_daily_projection (
    id                          UUID        NOT NULL DEFAULT gen_random_uuid(),
    date                        DATE        NOT NULL,
    task_type                   TEXT        NOT NULL,
    model_id                    TEXT        NOT NULL,
    model_tier                  TEXT        NOT NULL,

    -- Call aggregates
    call_count                  INTEGER     NOT NULL DEFAULT 0,
    total_tokens_in             BIGINT      NOT NULL DEFAULT 0,
    total_tokens_out            BIGINT      NOT NULL DEFAULT 0,

    -- Cost aggregates (NUMERIC for monetary precision)
    total_actual_cost_usd       NUMERIC(12, 6) NOT NULL DEFAULT 0,
    total_opus_equivalent_usd   NUMERIC(12, 6) NOT NULL DEFAULT 0,
    total_savings_usd           NUMERIC(12, 6) NOT NULL DEFAULT 0,

    -- Quality aggregates
    escalation_count            INTEGER     NOT NULL DEFAULT 0,
    avg_latency_ms              FLOAT8,
    avg_quality_score           FLOAT8,

    -- Reducer provenance (for replay-safety verification)
    projection_cursor           TEXT,
    source_event_id             TEXT,
    freshness_state             TEXT        CHECK (freshness_state IN ('fresh', 'stale', 'replaying')),
    observed_at                 TIMESTAMPTZ,
    reducer_version             TEXT,

    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_llm_delegation_daily_projection
        PRIMARY KEY (id),

    CONSTRAINT uq_llm_delegation_daily_projection_key
        UNIQUE (date, task_type, model_id, model_tier),

    CONSTRAINT chk_llm_delegation_daily_projection_model_tier
        CHECK (model_tier IN ('free', 'local', 'frontier'))
);

CREATE INDEX IF NOT EXISTS idx_llm_delegation_daily_projection_date
    ON llm_delegation_daily_projection (date);

CREATE INDEX IF NOT EXISTS idx_llm_delegation_daily_projection_task_type
    ON llm_delegation_daily_projection (task_type);

COMMENT ON TABLE llm_delegation_daily_projection IS
    'Daily aggregate projection for LLM delegation cost/savings. Unique key: (date, task_type, model_id, model_tier). OMN-11773.';
