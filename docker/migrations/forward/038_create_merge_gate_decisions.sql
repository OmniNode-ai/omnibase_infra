-- =============================================================================
-- MIGRATION: Create merge_gate_decisions table
-- =============================================================================
-- Ticket: OMN-3140 (NodeMergeGateEffect + migration)
-- Version: 1.0.0
--
-- PURPOSE:
--   Persists merge gate decisions from the pr-queue-pipeline.
--   Gate decisions are idempotent: UNIQUE(pr_ref, head_sha) ensures
--   re-evaluations for the same PR + SHA upsert rather than duplicate.
--
-- ROLLBACK:
--   See rollback/rollback_038_create_merge_gate_decisions.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.merge_gate_decisions (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    gate_id           UUID        NOT NULL,
    pr_ref            TEXT        NOT NULL,
    head_sha          TEXT        NOT NULL,
    base_sha          TEXT        NOT NULL,
    decision          TEXT        NOT NULL CHECK (decision IN ('PASS', 'WARN', 'QUARANTINE')),
    tier              TEXT        NOT NULL CHECK (tier IN ('tier-a', 'tier-b')),
    violations        JSONB       NOT NULL DEFAULT '[]'
                      CHECK (jsonb_typeof(violations) = 'array'),
    run_id            UUID,
    correlation_id    UUID,
    run_fingerprint   TEXT,
    decided_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (pr_ref, head_sha)
);

-- Index for querying recent decisions by PR
CREATE INDEX IF NOT EXISTS idx_merge_gate_decisions_pr_ref
    ON merge_gate_decisions (pr_ref, decided_at DESC);

-- Index for correlation lookups
CREATE INDEX IF NOT EXISTS idx_merge_gate_decisions_correlation_id
    ON merge_gate_decisions (correlation_id)
    WHERE correlation_id IS NOT NULL;
