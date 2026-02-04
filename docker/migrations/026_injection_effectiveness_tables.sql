-- Migration: 026_injection_effectiveness_tables
-- Description: Create tables for injection effectiveness observability (OMN-1890)
-- Created: 2026-02-04
--
-- Purpose: Stores injection effectiveness metrics for A/B testing analysis.
-- Three tables track session-level metrics, per-prompt latency breakdowns,
-- and per-pattern utilization rates.
--
-- Idempotency:
--   - injection_effectiveness: ON CONFLICT (session_id) DO UPDATE
--   - latency_breakdowns: ON CONFLICT (session_id, prompt_index) DO NOTHING
--   - pattern_hit_rates: ON CONFLICT (session_id, pattern_id) DO NOTHING
--
-- TTL Column: updated_at for injection_effectiveness, created_at for others
--
-- Rollback: See 026_injection_effectiveness_tables_rollback.sql

-- ============================================================================
-- INJECTION_EFFECTIVENESS TABLE (session-level summary)
-- ============================================================================
-- Records session-level injection metrics with upsert support for incremental
-- updates as context-utilization, agent-match, and latency events arrive.

CREATE TABLE IF NOT EXISTS injection_effectiveness (
    -- Identity (session_id is the unique key for upserts)
    session_id UUID PRIMARY KEY,
    correlation_id UUID,

    -- Environment Context (future-proofing for multi-runtime)
    realm TEXT,                            -- Trust domain / environment
    runtime_id TEXT,                       -- Runtime instance identifier
    routing_path VARCHAR(20),              -- 'event' | 'local' | 'hybrid'

    -- A/B Testing
    cohort VARCHAR(32),                    -- 'control' | 'treatment'
    cohort_identity_type VARCHAR(32),      -- 'user_id' | 'repo_path' | 'session_id'

    -- Injection metrics
    total_injected_tokens INTEGER,
    patterns_injected INTEGER,

    -- Utilization (heuristic-based, NOT token attribution)
    utilization_score FLOAT,               -- 0.0-1.0
    utilization_method VARCHAR(64),        -- 'identifier_match' | 'semantic' | 'timeout'
    injected_identifiers_count INTEGER,
    reused_identifiers_count INTEGER,

    -- Agent matching (graded 0.0-1.0, replaces boolean)
    agent_match_score FLOAT,               -- 0.0-1.0
    expected_agent VARCHAR(255),
    actual_agent VARCHAR(255),

    -- Latency (user-visible = MAX of prompt latencies)
    -- Note: latency_delta is computed via JOIN with latency_baseline, not stored
    user_visible_latency_ms INTEGER,

    -- Audit (TTL keys off updated_at for upsert tables)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Metadata (extensible JSON for future fields)
    metadata JSONB,

    -- Constraints
    CONSTRAINT valid_cohort CHECK (cohort IS NULL OR cohort IN ('control', 'treatment')),
    CONSTRAINT valid_cohort_identity CHECK (
        cohort_identity_type IS NULL
        OR cohort_identity_type IN ('user_id', 'repo_path', 'session_id')
    ),
    CONSTRAINT valid_utilization_score CHECK (
        utilization_score IS NULL
        OR (utilization_score >= 0.0 AND utilization_score <= 1.0)
    ),
    CONSTRAINT valid_agent_match_score CHECK (
        agent_match_score IS NULL
        OR (agent_match_score >= 0.0 AND agent_match_score <= 1.0)
    ),
    CONSTRAINT valid_routing_path CHECK (
        routing_path IS NULL
        OR routing_path IN ('event', 'local', 'hybrid')
    )
);

-- ============================================================================
-- LATENCY_BREAKDOWNS TABLE (per-prompt performance)
-- ============================================================================
-- Records per-prompt latency breakdowns for detailed performance analysis.
-- Each prompt in a session gets one row with internal and user-perceived timings.

CREATE TABLE IF NOT EXISTS latency_breakdowns (
    -- Identity
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    prompt_id UUID NOT NULL,               -- Unique prompt identifier from emitter

    -- A/B Testing (denormalized for fast queries)
    cohort VARCHAR(32),
    cache_hit BOOLEAN DEFAULT FALSE,

    -- Internal timings (ms) - nullable as not all events have all timings
    routing_latency_ms INTEGER,
    retrieval_latency_ms INTEGER,
    injection_latency_ms INTEGER,

    -- User-perceived latency (required)
    user_latency_ms INTEGER NOT NULL,

    -- Timestamps
    emitted_at TIMESTAMPTZ,                -- Event time from producer (for drift analysis)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- Ingest time

    -- Foreign key to session (ensures referential integrity)
    CONSTRAINT fk_latency_breakdowns_session
        FOREIGN KEY (session_id)
        REFERENCES injection_effectiveness(session_id)
        ON DELETE CASCADE,

    -- Constraints
    CONSTRAINT unique_session_prompt UNIQUE (session_id, prompt_id),
    CONSTRAINT valid_cohort_breakdown CHECK (cohort IS NULL OR cohort IN ('control', 'treatment'))
);

-- ============================================================================
-- PATTERN_HIT_RATES TABLE (pattern effectiveness)
-- ============================================================================
-- Records per-pattern utilization metrics for pattern-level effectiveness analysis.
-- Supports minimum support gating (N=20) in consumer logic, not schema constraint.

CREATE TABLE IF NOT EXISTS pattern_hit_rates (
    -- Identity
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id UUID NOT NULL,              -- Pattern identifier (matches learned_patterns.id)
    domain_id UUID,                        -- Domain scope (if pattern is domain-specific)

    -- Utilization Stats (aggregated across all sessions)
    utilization_method VARCHAR(64) NOT NULL,  -- Method used to measure
    utilization_score REAL NOT NULL,       -- Rolling average score (0.0-1.0)
    hit_count INTEGER NOT NULL DEFAULT 0,  -- Times pattern was used in response
    miss_count INTEGER NOT NULL DEFAULT 0, -- Times pattern was injected but not used

    -- Confidence (minimum support gating: N>=20)
    sample_count INTEGER NOT NULL DEFAULT 0,  -- Total observations
    confidence REAL,                       -- Confidence level (NULL if N<20)

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One row per pattern per method (aggregate across all sessions)
    CONSTRAINT unique_pattern_method UNIQUE (pattern_id, utilization_method),

    -- Constraints
    CONSTRAINT valid_pattern_utilization_score CHECK (
        utilization_score >= 0.0 AND utilization_score <= 1.0
    ),
    CONSTRAINT valid_pattern_confidence CHECK (
        confidence IS NULL
        OR (confidence >= 0.0 AND confidence <= 1.0)
    )
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- injection_effectiveness indexes
CREATE INDEX IF NOT EXISTS idx_injection_effectiveness_updated_at
    ON injection_effectiveness (updated_at);

CREATE INDEX IF NOT EXISTS idx_injection_effectiveness_cohort
    ON injection_effectiveness (cohort)
    WHERE cohort IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_injection_effectiveness_cohort_created
    ON injection_effectiveness (cohort, created_at)
    WHERE cohort IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_injection_effectiveness_utilization_method
    ON injection_effectiveness (utilization_method)
    WHERE utilization_method IS NOT NULL;

-- latency_breakdowns indexes
CREATE INDEX IF NOT EXISTS idx_latency_breakdowns_session_id
    ON latency_breakdowns (session_id);

CREATE INDEX IF NOT EXISTS idx_latency_breakdowns_cohort_created
    ON latency_breakdowns (cohort, created_at)
    WHERE cohort IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_latency_breakdowns_cache_hit
    ON latency_breakdowns (cache_hit, created_at)
    WHERE cache_hit = TRUE;

-- pattern_hit_rates indexes
CREATE INDEX IF NOT EXISTS idx_pattern_hit_rates_pattern_id
    ON pattern_hit_rates (pattern_id);

CREATE INDEX IF NOT EXISTS idx_pattern_hit_rates_domain_id
    ON pattern_hit_rates (domain_id)
    WHERE domain_id IS NOT NULL;

-- Partial index for confident patterns (N>=20)
CREATE INDEX IF NOT EXISTS idx_pattern_hit_rates_confident
    ON pattern_hit_rates (sample_count)
    WHERE sample_count >= 20;

CREATE INDEX IF NOT EXISTS idx_pattern_hit_rates_updated_at
    ON pattern_hit_rates (updated_at);

-- injection_effectiveness environment indexes
CREATE INDEX IF NOT EXISTS idx_injection_effectiveness_realm
    ON injection_effectiveness (realm)
    WHERE realm IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_injection_effectiveness_routing_path
    ON injection_effectiveness (routing_path)
    WHERE routing_path IS NOT NULL;

-- ============================================================================
-- TRIGGERS
-- ============================================================================
-- Auto-update updated_at on any row modification for injection_effectiveness

CREATE OR REPLACE FUNCTION update_injection_effectiveness_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_injection_effectiveness_updated_at ON injection_effectiveness;
CREATE TRIGGER trigger_injection_effectiveness_updated_at
    BEFORE UPDATE ON injection_effectiveness
    FOR EACH ROW
    EXECUTE FUNCTION update_injection_effectiveness_updated_at();

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE injection_effectiveness IS
    'Session-level injection effectiveness metrics for A/B testing (OMN-1890). '
    'TTL keys off updated_at. Supports upsert for incremental updates.';

COMMENT ON COLUMN injection_effectiveness.session_id IS
    'Unique session identifier - PRIMARY KEY for upsert operations';
COMMENT ON COLUMN injection_effectiveness.correlation_id IS
    'Request correlation ID for tracing across services';
COMMENT ON COLUMN injection_effectiveness.cohort IS
    'A/B test cohort: control (no injection) or treatment (with injection)';
COMMENT ON COLUMN injection_effectiveness.cohort_identity_type IS
    'Identity type used for cohort assignment: user_id > repo_path > session_id';
COMMENT ON COLUMN injection_effectiveness.utilization_score IS
    'Heuristic-based utilization score 0.0-1.0 (NOT token attribution)';
COMMENT ON COLUMN injection_effectiveness.utilization_method IS
    'Method used to calculate utilization: identifier_match, semantic, or timeout';
COMMENT ON COLUMN injection_effectiveness.agent_match_score IS
    'Graded agent matching score 0.0-1.0 (replaces boolean match)';
COMMENT ON COLUMN injection_effectiveness.user_visible_latency_ms IS
    'MAX of all prompt latencies in session - user-perceived total latency';

COMMENT ON COLUMN injection_effectiveness.realm IS
    'Trust domain / environment identifier for multi-runtime deployments';
COMMENT ON COLUMN injection_effectiveness.runtime_id IS
    'Runtime instance identifier for multi-runtime deployments';
COMMENT ON COLUMN injection_effectiveness.routing_path IS
    'Event routing path: event (Kafka), local (direct), hybrid (both)';

COMMENT ON TABLE latency_breakdowns IS
    'Per-prompt latency breakdowns for detailed performance analysis (OMN-1890). '
    'One row per (session_id, prompt_id) pair. FK to injection_effectiveness.';

COMMENT ON COLUMN latency_breakdowns.prompt_id IS
    'Unique prompt identifier from event emitter (UUID)';
COMMENT ON COLUMN latency_breakdowns.cache_hit IS
    'Whether this prompt benefited from cached context';
COMMENT ON COLUMN latency_breakdowns.user_latency_ms IS
    'User-perceived latency for this specific prompt';
COMMENT ON COLUMN latency_breakdowns.emitted_at IS
    'Event timestamp from producer (for drift analysis vs created_at ingest time)';

COMMENT ON TABLE pattern_hit_rates IS
    'Pattern-level utilization aggregates across all sessions (OMN-1890). '
    'One row per (pattern_id, utilization_method). Confidence is NULL until N>=20.';

COMMENT ON COLUMN pattern_hit_rates.pattern_id IS
    'Pattern identifier (UUID) - matches learned_patterns.id';
COMMENT ON COLUMN pattern_hit_rates.domain_id IS
    'Domain scope (if pattern is domain-specific)';
COMMENT ON COLUMN pattern_hit_rates.utilization_score IS
    'Rolling average utilization score (0.0-1.0) for this pattern';
COMMENT ON COLUMN pattern_hit_rates.hit_count IS
    'Times pattern was used in response (reused)';
COMMENT ON COLUMN pattern_hit_rates.miss_count IS
    'Times pattern was injected but not used';
COMMENT ON COLUMN pattern_hit_rates.sample_count IS
    'Total observations (hit_count + miss_count)';
COMMENT ON COLUMN pattern_hit_rates.confidence IS
    'Confidence level (NULL if sample_count < 20 per minimum support gating)';

-- ============================================================================
-- TRIGGER: Auto-update updated_at for pattern_hit_rates
-- ============================================================================

CREATE OR REPLACE FUNCTION update_pattern_hit_rates_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_pattern_hit_rates_updated_at ON pattern_hit_rates;
CREATE TRIGGER trigger_pattern_hit_rates_updated_at
    BEFORE UPDATE ON pattern_hit_rates
    FOR EACH ROW
    EXECUTE FUNCTION update_pattern_hit_rates_updated_at();
