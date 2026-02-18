-- Migration: 003_create_baselines_tables.sql
-- Purpose: Create baselines comparison tables for A/B treatment/control ROI measurement
-- Author: ONEX Infrastructure Team
-- Date: 2026-02-18
-- Ticket: OMN-2305
--
-- Design Decisions:
--
--   1. Three tables cover the baselines API surface:
--      - baselines_comparisons: One row per A/B comparison period (treatment vs control)
--      - baselines_trend: Time-series metrics per cohort for the /trend endpoint
--      - baselines_breakdown: Aggregated per-pattern performance for the /breakdown endpoint
--
--   2. Treatment vs control definition:
--      - Treatment: agent_routing_decisions rows WHERE confidence_score >= 0.8
--        (high-confidence selections with active pattern injection)
--      - Control: agent_routing_decisions rows WHERE confidence_score < 0.8 OR NULL
--        (low-confidence or missing injection context)
--      This maps directly to the `cohort` field in injection_effectiveness.
--
--   3. ROI formula: (treatment_success_rate - control_success_rate) / control_success_rate
--      Expressed as a percentage. NULL when control_success_rate is zero.
--
--   4. Idempotency via comparison_date unique constraint: Re-running batch
--      computation upserts rows safely. ON CONFLICT DO UPDATE ensures dashboard
--      shows latest computed values.
--
--   5. REAL vs NUMERIC for metrics: Using REAL for dashboard display values
--      (percentages, rates). No sub-penny precision needed. Computation accuracy
--      is maintained in SQL before writing.
--
--   6. All count fields are BIGINT to match COUNT(*) SQL aggregate return type
--      and prevent overflow for high-volume deployments.

-- =============================================================================
-- TABLE: baselines_comparisons
-- =============================================================================
-- One row per comparison period (daily aggregation).
-- Stores treatment vs control group outcomes used by /api/baselines/comparisons
-- and /api/baselines/summary.
-- =============================================================================

CREATE TABLE IF NOT EXISTS baselines_comparisons (
    -- Primary key
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Comparison period
    comparison_date             DATE            NOT NULL,   -- Day of the comparison (truncated to date)
    period_label                TEXT,                       -- Human label, e.g. "2026-02-18"

    -- Treatment group (sessions with high-confidence pattern injection)
    treatment_sessions          BIGINT          NOT NULL DEFAULT 0,
    treatment_success_rate      REAL,                       -- Ratio of completed actions to total (0.0-1.0)
    treatment_avg_latency_ms    REAL,                       -- Average user-visible latency
    treatment_avg_cost_tokens   REAL,                       -- Average token usage per session
    treatment_total_tokens      BIGINT          NOT NULL DEFAULT 0,

    -- Control group (sessions without pattern injection or low confidence)
    control_sessions            BIGINT          NOT NULL DEFAULT 0,
    control_success_rate        REAL,                       -- Ratio of completed actions to total (0.0-1.0)
    control_avg_latency_ms      REAL,                       -- Average user-visible latency
    control_avg_cost_tokens     REAL,                       -- Average token usage per session
    control_total_tokens        BIGINT          NOT NULL DEFAULT 0,

    -- Derived ROI metrics
    roi_pct                     REAL,                       -- (treatment_success_rate - control_success_rate) / control_success_rate * 100
    latency_improvement_pct     REAL,                       -- (control_latency - treatment_latency) / control_latency * 100
    cost_improvement_pct        REAL,                       -- (control_tokens - treatment_tokens) / control_tokens * 100
    sample_size                 BIGINT          NOT NULL DEFAULT 0,  -- treatment_sessions + control_sessions

    -- Metadata
    computed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Idempotency: one row per day
    CONSTRAINT uk_baselines_comparisons_date UNIQUE (comparison_date)
);

-- =============================================================================
-- INDEXES: baselines_comparisons
-- =============================================================================

-- Time-range queries for the comparisons endpoint
CREATE INDEX IF NOT EXISTS idx_baselines_comparisons_date_desc
    ON baselines_comparisons (comparison_date DESC);

-- Recent comparisons for summary display
CREATE INDEX IF NOT EXISTS idx_baselines_comparisons_computed_at
    ON baselines_comparisons (computed_at DESC);

-- =============================================================================
-- TABLE: baselines_trend
-- =============================================================================
-- Time-series rows for the /api/baselines/trend endpoint.
-- One row per cohort per day, enabling treatment vs control trend lines.
-- =============================================================================

CREATE TABLE IF NOT EXISTS baselines_trend (
    -- Primary key
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Time bucket
    trend_date                  DATE            NOT NULL,
    cohort                      TEXT            NOT NULL,   -- 'treatment' or 'control'

    -- Trend metrics
    session_count               BIGINT          NOT NULL DEFAULT 0,
    success_rate                REAL,                       -- success rate for this cohort/day
    avg_latency_ms              REAL,                       -- average latency this day
    avg_cost_tokens             REAL,                       -- average token cost this day
    roi_pct                     REAL,                       -- ROI relative to control baseline for this day

    -- Metadata
    computed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Idempotency: one row per cohort per day
    CONSTRAINT uk_baselines_trend_date_cohort UNIQUE (trend_date, cohort)
);

-- =============================================================================
-- INDEXES: baselines_trend
-- =============================================================================

-- Time-range queries ordered by date
CREATE INDEX IF NOT EXISTS idx_baselines_trend_date_desc
    ON baselines_trend (trend_date DESC);

-- Cohort filter queries
CREATE INDEX IF NOT EXISTS idx_baselines_trend_cohort
    ON baselines_trend (cohort, trend_date DESC);

-- =============================================================================
-- TABLE: baselines_breakdown
-- =============================================================================
-- Per-pattern performance breakdown for the /api/baselines/breakdown endpoint.
-- Aggregated from agent_routing_decisions grouped by selected_agent.
-- =============================================================================

CREATE TABLE IF NOT EXISTS baselines_breakdown (
    -- Primary key
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Pattern identifier (derived from md5(selected_agent)::uuid for consistency)
    pattern_id                  UUID            NOT NULL,
    pattern_label               TEXT,                       -- Human-readable label (selected_agent name)

    -- Effectiveness metrics
    treatment_success_rate      REAL,                       -- Success rate in treatment cohort
    control_success_rate        REAL,                       -- Success rate in control cohort
    roi_pct                     REAL,                       -- Pattern-specific ROI
    sample_count                BIGINT          NOT NULL DEFAULT 0,
    treatment_count             BIGINT          NOT NULL DEFAULT 0,
    control_count               BIGINT          NOT NULL DEFAULT 0,

    -- Confidence signal
    confidence                  REAL,                       -- NULL until sample_count >= 20

    -- Metadata
    computed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Idempotency: one row per pattern
    CONSTRAINT uk_baselines_breakdown_pattern UNIQUE (pattern_id)
);

-- =============================================================================
-- INDEXES: baselines_breakdown
-- =============================================================================

-- ROI-sorted display for dashboard
CREATE INDEX IF NOT EXISTS idx_baselines_breakdown_roi_desc
    ON baselines_breakdown (roi_pct DESC NULLS LAST);

-- Pattern label search
CREATE INDEX IF NOT EXISTS idx_baselines_breakdown_label
    ON baselines_breakdown (pattern_label);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE baselines_comparisons IS
    'Daily A/B baseline comparisons between treatment (with pattern injection) and '
    'control (without) groups. Populated by ServiceBatchComputeBaselines. '
    'Powers /api/baselines/comparisons and /api/baselines/summary endpoints (OMN-2305).';

COMMENT ON COLUMN baselines_comparisons.treatment_sessions IS
    'Sessions with confidence_score >= 0.8 in agent_routing_decisions (active injection)';

COMMENT ON COLUMN baselines_comparisons.control_sessions IS
    'Sessions with confidence_score < 0.8 or NULL (no or low-confidence injection)';

COMMENT ON COLUMN baselines_comparisons.roi_pct IS
    'ROI = (treatment_success_rate - control_success_rate) / control_success_rate * 100. '
    'NULL when control_success_rate is zero or NULL.';

COMMENT ON TABLE baselines_trend IS
    'Time-series metrics per cohort per day for the /api/baselines/trend endpoint. '
    'One row per (cohort, date) pair. Populated by ServiceBatchComputeBaselines (OMN-2305).';

COMMENT ON COLUMN baselines_trend.cohort IS
    'A/B cohort: treatment (high-confidence injection) or control (no/low injection)';

COMMENT ON TABLE baselines_breakdown IS
    'Per-pattern performance breakdown for /api/baselines/breakdown endpoint. '
    'One row per selected_agent (treated as a pattern proxy). '
    'Populated by ServiceBatchComputeBaselines (OMN-2305).';

COMMENT ON COLUMN baselines_breakdown.pattern_id IS
    'Deterministic UUID derived from md5(selected_agent)::uuid for stable cross-run identity';

COMMENT ON COLUMN baselines_breakdown.confidence IS
    'Confidence score set when sample_count >= 20. NULL for insufficient data.';
