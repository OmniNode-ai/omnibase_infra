-- =============================================================================
-- MIGRATION: Create delta_metrics_by_model table
-- =============================================================================
-- Ticket: OMN-3142 (NodeDeltaBundleEffect + NodeDeltaMetricsEffect)
-- Version: 1.0.0
--
-- PURPOSE:
--   Aggregated per-model, per-subsystem performance rollups computed from
--   delta_bundles. Each row represents a time-period summary of PR outcomes
--   for a specific coding model and subsystem combination.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   See rollback/rollback_040_create_delta_metrics_by_model.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS delta_metrics_by_model (
    -- Primary key
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Rollup dimensions
    coding_model        TEXT            NOT NULL,
    subsystem           TEXT            NOT NULL,

    -- Counters
    total_prs           INTEGER         NOT NULL DEFAULT 0,
    merged_prs          INTEGER         NOT NULL DEFAULT 0,
    reverted_prs        INTEGER         NOT NULL DEFAULT 0,
    quarantine_prs      INTEGER         NOT NULL DEFAULT 0,
    fix_prs             INTEGER         NOT NULL DEFAULT 0,

    -- Aggregated metrics
    avg_gate_violations NUMERIC(6,2),

    -- Time period for this rollup
    period_start        DATE            NOT NULL,
    period_end          DATE            NOT NULL,

    -- Audit timestamp
    computed_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Natural key: one rollup per (model, subsystem, period)
    UNIQUE (coding_model, subsystem, period_start, period_end)
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Support queries by coding model (dashboard model comparison)
CREATE INDEX IF NOT EXISTS idx_delta_metrics_coding_model
    ON delta_metrics_by_model (coding_model);

-- Support time-range queries (period analysis)
CREATE INDEX IF NOT EXISTS idx_delta_metrics_period
    ON delta_metrics_by_model (period_start, period_end);

-- Support subsystem filtering
CREATE INDEX IF NOT EXISTS idx_delta_metrics_subsystem
    ON delta_metrics_by_model (subsystem);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE delta_metrics_by_model IS
    'Per-model, per-subsystem PR performance rollups (OMN-3142). '
    'Aggregated from delta_bundles by NodeDeltaMetricsEffect. '
    'Used for dashboard stabilization tax and model comparison views.';

COMMENT ON COLUMN delta_metrics_by_model.coding_model IS
    'LLM model identifier (e.g. "claude-opus-4-20250514").';
COMMENT ON COLUMN delta_metrics_by_model.subsystem IS
    'Subsystem classification (e.g. "omnibase_infra").';
COMMENT ON COLUMN delta_metrics_by_model.total_prs IS
    'Total PRs processed in this period for this model+subsystem.';
COMMENT ON COLUMN delta_metrics_by_model.merged_prs IS
    'PRs with outcome=merged.';
COMMENT ON COLUMN delta_metrics_by_model.reverted_prs IS
    'PRs with outcome=reverted.';
COMMENT ON COLUMN delta_metrics_by_model.quarantine_prs IS
    'PRs with gate_decision=QUARANTINE.';
COMMENT ON COLUMN delta_metrics_by_model.fix_prs IS
    'PRs with is_fix_pr=TRUE (stabilization tax).';
COMMENT ON COLUMN delta_metrics_by_model.avg_gate_violations IS
    'Average number of gate violations per PR in this rollup.';
COMMENT ON COLUMN delta_metrics_by_model.period_start IS
    'Start date (inclusive) of the rollup period.';
COMMENT ON COLUMN delta_metrics_by_model.period_end IS
    'End date (inclusive) of the rollup period.';

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '040',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
