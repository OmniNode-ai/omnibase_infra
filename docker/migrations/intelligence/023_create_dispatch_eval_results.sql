-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration: 023_create_dispatch_eval_results
-- Description: Create dispatch_eval_results table for dispatch evaluation outcomes.
-- Ticket: OMN-10390
--
-- Idempotency:
--   All statements use IF NOT EXISTS so re-applying this migration is safe.

CREATE TABLE IF NOT EXISTS dispatch_eval_results (
    task_id             TEXT        NOT NULL,
    dispatch_id         TEXT        NOT NULL,
    ticket_id           TEXT,
    verdict             TEXT        NOT NULL,
    quality_score       NUMERIC(4, 3),
    token_cost          INTEGER     NOT NULL DEFAULT 0,
    dollars_cost        NUMERIC(10, 4) NOT NULL DEFAULT 0,
    model_calls         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    evaluated_at        TIMESTAMPTZ NOT NULL,
    eval_latency_ms     INTEGER     NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    usage_source        TEXT        NOT NULL,
    estimation_method   TEXT,
    source_payload_hash TEXT,

    CONSTRAINT pk_dispatch_eval_results
        PRIMARY KEY (task_id, dispatch_id),
    CONSTRAINT uq_dispatch_eval_results_task_dispatch
        UNIQUE (task_id, dispatch_id),
    CONSTRAINT chk_dispatch_eval_results_verdict
        CHECK (verdict IN ('PASS', 'FAIL', 'ERROR', 'SKIPPED')),
    CONSTRAINT chk_dispatch_eval_results_quality_score
        CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)),
    CONSTRAINT chk_dispatch_eval_results_token_cost
        CHECK (token_cost >= 0),
    CONSTRAINT chk_dispatch_eval_results_dollars_cost
        CHECK (dollars_cost >= 0),
    CONSTRAINT chk_dispatch_eval_results_model_calls_array
        CHECK (jsonb_typeof(model_calls) = 'array'),
    CONSTRAINT chk_dispatch_eval_results_eval_latency_ms
        CHECK (eval_latency_ms >= 0),
    CONSTRAINT chk_dispatch_eval_results_usage_source
        CHECK (usage_source IN ('MEASURED', 'ESTIMATED', 'UNKNOWN')),
    CONSTRAINT chk_dispatch_eval_results_estimation_method
        CHECK (
            (usage_source = 'ESTIMATED' AND estimation_method IS NOT NULL)
            OR (usage_source <> 'ESTIMATED' AND estimation_method IS NULL)
        ),
    CONSTRAINT chk_dispatch_eval_results_source_payload_hash
        CHECK (
            (usage_source = 'MEASURED' AND source_payload_hash IS NOT NULL)
            OR (usage_source <> 'MEASURED' AND source_payload_hash IS NULL)
        )
);

CREATE INDEX IF NOT EXISTS idx_dispatch_eval_results_ticket_id
    ON dispatch_eval_results (ticket_id);

CREATE INDEX IF NOT EXISTS idx_dispatch_eval_results_evaluated_at_desc
    ON dispatch_eval_results (evaluated_at DESC);

COMMENT ON TABLE dispatch_eval_results IS
    'Dispatch evaluation outcomes keyed by task_id and dispatch_id. OMN-10390.';

COMMENT ON COLUMN dispatch_eval_results.evaluated_at IS
    'Authoritative ordering timestamp for dispatch evaluation projections.';

COMMENT ON COLUMN dispatch_eval_results.created_at IS
    'Insertion timestamp only; non-authoritative for projection ordering.';
