-- Migration: 064_create_pattern_learning_artifacts
-- Description: Create pattern_learning_artifacts projection table (OMN-8540)
-- Created: 2026-04-12
--
-- Purpose: Projection table for onex.evt.omniintelligence.pattern-stored.v1 events.
-- Consumed by omniintelligence projection consumers. Closes the golden chain
-- pattern_learning chain: the tail table was referenced but never migrated into
-- the omnibase_infra Postgres schema, causing 0/5 golden chain sweep results.
--
-- Schema derived from omnidash migrations 0001_omnidash_analytics_read_model.sql,
-- 0038_pattern_learning_unique_pattern_id.sql, and 0040_add_evidence_tier_to_patlearn.sql.
--
-- Idempotency: All statements use IF NOT EXISTS; safe to re-apply.
-- Rollback: See rollback/rollback_064_pattern_learning_artifacts.sql

-- ============================================================================
-- TABLE: pattern_learning_artifacts
-- ============================================================================
-- Projects pattern-stored events from omniintelligence.
-- Lookup key for golden chain: pattern_name (not correlation_id).

CREATE TABLE IF NOT EXISTS pattern_learning_artifacts (
    id                  UUID        NOT NULL DEFAULT gen_random_uuid(),
    pattern_id          UUID        NOT NULL,
    pattern_name        VARCHAR(255) NOT NULL,
    pattern_type        VARCHAR(100) NOT NULL,
    language            VARCHAR(50),
    lifecycle_state     TEXT        NOT NULL DEFAULT 'candidate',
    state_changed_at    TIMESTAMPTZ,
    composite_score     NUMERIC(10, 6) NOT NULL,
    scoring_evidence    JSONB       NOT NULL DEFAULT '{}',
    signature           JSONB       NOT NULL DEFAULT '{}',
    metrics             JSONB       DEFAULT '{}',
    metadata            JSONB       DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    projected_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_pattern_learning_artifacts PRIMARY KEY (id),
    CONSTRAINT uq_pattern_learning_pattern_id UNIQUE (pattern_id)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_patlearn_lifecycle_state
    ON pattern_learning_artifacts (lifecycle_state);

CREATE INDEX IF NOT EXISTS idx_patlearn_composite_score
    ON pattern_learning_artifacts (composite_score DESC);

CREATE INDEX IF NOT EXISTS idx_patlearn_state_changed_at
    ON pattern_learning_artifacts (state_changed_at DESC);

CREATE INDEX IF NOT EXISTS idx_patlearn_created_at
    ON pattern_learning_artifacts (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_patlearn_updated_at
    ON pattern_learning_artifacts (updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_patlearn_lifecycle_score
    ON pattern_learning_artifacts (lifecycle_state, composite_score DESC);

CREATE INDEX IF NOT EXISTS idx_patlearn_pattern_name
    ON pattern_learning_artifacts (pattern_name);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE pattern_learning_artifacts IS
    'Pattern learning projection from onex.evt.omniintelligence.pattern-stored.v1. '
    'UPSERT key: pattern_id. Golden chain pattern_learning tail table (OMN-8540).';

COMMENT ON COLUMN pattern_learning_artifacts.pattern_id IS
    'Unique pattern identifier. Used as UPSERT conflict target.';

COMMENT ON COLUMN pattern_learning_artifacts.pattern_name IS
    'Human-readable pattern name. Used as alternate golden chain lookup key.';

COMMENT ON COLUMN pattern_learning_artifacts.lifecycle_state IS
    'Pattern lifecycle state: candidate, promoted, deprecated, archived.';

COMMENT ON COLUMN pattern_learning_artifacts.composite_score IS
    'Composite quality score (0-1) used for pattern promotion decisions.';

COMMENT ON COLUMN pattern_learning_artifacts.scoring_evidence IS
    'JSON evidence used to compute composite_score (signal breakdown).';

COMMENT ON COLUMN pattern_learning_artifacts.signature IS
    'Structural signature of the pattern for similarity matching.';
