-- =============================================================================
-- MIGRATION: Create baselines snapshot table
-- =============================================================================
-- Ticket: OMN-8667
-- Epic: OMN-7264 (Intelligent Model Router MVP)
-- Version: 1.0.0
--
-- PURPOSE:
--   Creates the baselines table for storing system state snapshots captured
--   by /onex:baseline. This is SEPARATE from baselines_comparisons which
--   stores check-result comparisons. This table holds raw snapshot captures.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   docker/migrations/rollback/rollback_067_create_baselines_table.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.baselines (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    node_count INTEGER,
    topic_count INTEGER,
    test_count INTEGER,
    coverage_pct DOUBLE PRECISION,
    latency_p50_ms INTEGER,
    latency_p99_ms INTEGER,
    snapshot_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_baselines_created_at ON public.baselines (created_at DESC);

COMMENT ON TABLE public.baselines IS
    'System state snapshots captured by /onex:baseline skill. '
    'Stores raw metric captures (node_count, topic_count, test coverage, latency). '
    'Separate from baselines_comparisons which stores delta check results. (OMN-8667)';

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '067',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
