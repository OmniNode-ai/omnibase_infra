-- =============================================================================
-- MIGRATION: Create delta_bundles table
-- =============================================================================
-- Ticket: OMN-3142 (NodeDeltaBundleEffect + NodeDeltaMetricsEffect)
-- Version: 1.0.0
--
-- PURPOSE:
--   Tracks the full lifecycle of a PR as a "delta bundle" — from merge-gate
--   decision through to final outcome (merged, reverted, closed). Each row
--   captures the gate decision, coding model, fix-PR status, and eventual
--   outcome for a single (pr_ref, head_sha) pair.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   See rollback/rollback_039_create_delta_bundles.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS delta_bundles (
    -- Primary key
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Bundle identity (unique per event)
    bundle_id           UUID            NOT NULL UNIQUE,

    -- PR identity
    pr_ref              TEXT            NOT NULL,
    head_sha            TEXT            NOT NULL,
    base_sha            TEXT            NOT NULL,

    -- Model and subsystem context
    coding_model        TEXT,
    subsystem           TEXT,

    -- Gate decision from merge-gate
    gate_decision       TEXT            NOT NULL CHECK (gate_decision IN ('PASS', 'WARN', 'QUARANTINE')),
    gate_violations     JSONB           NOT NULL DEFAULT '[]',

    -- Fix-PR stabilization tracking
    is_fix_pr           BOOLEAN         NOT NULL DEFAULT FALSE,
    stabilizes_pr_ref   TEXT,

    -- PR outcome (populated on pr-outcome event)
    outcome             TEXT            CHECK (outcome IN ('merged', 'reverted', 'closed')),
    merged_at           TIMESTAMPTZ,
    bundle_completed_at TIMESTAMPTZ,

    -- Audit timestamps
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Natural key: one bundle per (pr_ref, head_sha) pair
    UNIQUE (pr_ref, head_sha)
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Support queries by PR reference (dashboard lookups)
CREATE INDEX IF NOT EXISTS idx_delta_bundles_pr_ref
    ON delta_bundles (pr_ref);

-- Support queries by coding model (per-model analytics)
CREATE INDEX IF NOT EXISTS idx_delta_bundles_coding_model
    ON delta_bundles (coding_model);

-- Support time-range queries
CREATE INDEX IF NOT EXISTS idx_delta_bundles_created_at
    ON delta_bundles (created_at DESC);

-- Support fix-PR analysis
CREATE INDEX IF NOT EXISTS idx_delta_bundles_is_fix_pr
    ON delta_bundles (is_fix_pr) WHERE is_fix_pr = TRUE;

-- Support outcome filtering
CREATE INDEX IF NOT EXISTS idx_delta_bundles_outcome
    ON delta_bundles (outcome) WHERE outcome IS NOT NULL;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE delta_bundles IS
    'PR lifecycle tracking as delta bundles (OMN-3142). '
    'Each row represents a single (pr_ref, head_sha) pair from merge-gate '
    'decision through to final outcome. Used by NodeDeltaBundleEffect.';

COMMENT ON COLUMN delta_bundles.bundle_id IS
    'Unique event identifier from the merge-gate-decision event.';
COMMENT ON COLUMN delta_bundles.pr_ref IS
    'PR reference string (e.g. "owner/repo#123").';
COMMENT ON COLUMN delta_bundles.head_sha IS
    'Git HEAD SHA at time of gate decision.';
COMMENT ON COLUMN delta_bundles.base_sha IS
    'Git base SHA for the PR diff.';
COMMENT ON COLUMN delta_bundles.coding_model IS
    'LLM model that authored the code (e.g. "claude-opus-4-20250514").';
COMMENT ON COLUMN delta_bundles.subsystem IS
    'Subsystem classification (e.g. "omnibase_infra", "omniclaude").';
COMMENT ON COLUMN delta_bundles.gate_decision IS
    'Merge-gate verdict: PASS, WARN, or QUARANTINE.';
COMMENT ON COLUMN delta_bundles.gate_violations IS
    'JSON array of gate violation details.';
COMMENT ON COLUMN delta_bundles.is_fix_pr IS
    'True if this PR carries a stabilizes:<pr_ref> label.';
COMMENT ON COLUMN delta_bundles.stabilizes_pr_ref IS
    'The original PR ref this fix-PR stabilizes (from label).';
COMMENT ON COLUMN delta_bundles.outcome IS
    'Final PR outcome: merged, reverted, or closed.';
COMMENT ON COLUMN delta_bundles.merged_at IS
    'Timestamp when PR was merged (NULL if not merged).';
COMMENT ON COLUMN delta_bundles.bundle_completed_at IS
    'Timestamp when outcome was recorded (bundle lifecycle complete).';

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '039',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
