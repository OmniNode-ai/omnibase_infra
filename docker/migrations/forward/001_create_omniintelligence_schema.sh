#!/bin/bash
# 001_create_omniintelligence_schema.sh
# Provisions the omniintelligence database schema (tables, indexes, triggers,
# functions, materialized views, and seed data).
#
# This script runs during PostgreSQL container initialization
# (docker-entrypoint-initdb.d) and MUST execute AFTER
# 000_create_multiple_databases.sh, which creates the omniintelligence
# database itself. Alphabetical ordering guarantees this (001 > 000).
#
# Schema contents correspond to omniintelligence migrations 000-013:
#   - fsm_state, fsm_state_history
#   - workflow_executions
#   - domain_taxonomy (with seed data)
#   - learned_patterns (with signature_hash, evidence_tier)
#   - pattern_disable_events
#   - pattern_injections (with run_id)
#   - disabled_patterns_current (materialized view)
#   - pattern_lifecycle_transitions
#   - pattern_measured_attributions
#
# Idempotent: uses IF NOT EXISTS / IF NOT EXISTS throughout.

set -e
set -u

echo "============================================="
echo "001: Provisioning omniintelligence schema"
echo "============================================="

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "omniintelligence" <<'EOSQL'

-- Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- FSM State Table
CREATE TABLE IF NOT EXISTS fsm_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fsm_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    current_state VARCHAR(50) NOT NULL,
    previous_state VARCHAR(50),
    transition_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB,
    lease_id VARCHAR(255),
    lease_epoch INTEGER,
    lease_expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_fsm_entity UNIQUE (fsm_type, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_fsm_state_fsm_entity ON fsm_state(fsm_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_fsm_state_current_state ON fsm_state(current_state);
CREATE INDEX IF NOT EXISTS idx_fsm_state_type_state ON fsm_state(fsm_type, current_state);
CREATE INDEX IF NOT EXISTS idx_fsm_state_lease ON fsm_state(lease_id, lease_epoch) WHERE lease_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_fsm_state_lease_expiry ON fsm_state(lease_expires_at) WHERE lease_expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_fsm_state_transition_time ON fsm_state(transition_timestamp);
CREATE INDEX IF NOT EXISTS idx_fsm_state_metadata ON fsm_state USING GIN (metadata);

CREATE OR REPLACE FUNCTION update_fsm_state_updated_at() RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$ LANGUAGE plpgsql;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_fsm_state_updated_at') THEN
        CREATE TRIGGER trigger_fsm_state_updated_at BEFORE UPDATE ON fsm_state FOR EACH ROW EXECUTE FUNCTION update_fsm_state_updated_at();
    END IF;
END $$;

-- FSM State History
CREATE TABLE IF NOT EXISTS fsm_state_history (
    id BIGSERIAL PRIMARY KEY,
    fsm_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    from_state VARCHAR(50),
    to_state VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,
    transitioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_ms FLOAT,
    correlation_id VARCHAR(255),
    metadata JSONB,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    partition_key DATE NOT NULL DEFAULT CURRENT_DATE
);
CREATE INDEX IF NOT EXISTS idx_fsm_history_entity ON fsm_state_history(fsm_type, entity_id, transitioned_at DESC);
CREATE INDEX IF NOT EXISTS idx_fsm_history_correlation ON fsm_state_history(correlation_id);
CREATE INDEX IF NOT EXISTS idx_fsm_history_time ON fsm_state_history(transitioned_at DESC);
CREATE INDEX IF NOT EXISTS idx_fsm_history_failures ON fsm_state_history(fsm_type, success) WHERE success = FALSE;
CREATE INDEX IF NOT EXISTS idx_fsm_history_partition ON fsm_state_history(partition_key);
CREATE INDEX IF NOT EXISTS idx_fsm_history_metadata ON fsm_state_history USING GIN (metadata);

CREATE OR REPLACE FUNCTION record_fsm_state_history() RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'UPDATE' AND OLD.current_state != NEW.current_state) OR TG_OP = 'INSERT' THEN
        INSERT INTO fsm_state_history (fsm_type, entity_id, from_state, to_state, action, transitioned_at, duration_ms, correlation_id, metadata, success)
        VALUES (NEW.fsm_type, NEW.entity_id, OLD.current_state, NEW.current_state, COALESCE(NEW.metadata->>'last_action', 'UNKNOWN'), NEW.transition_timestamp,
            CASE WHEN OLD.transition_timestamp IS NOT NULL THEN EXTRACT(EPOCH FROM (NEW.transition_timestamp - OLD.transition_timestamp)) * 1000 ELSE NULL END,
            NEW.metadata->>'correlation_id', NEW.metadata, TRUE);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_record_fsm_history') THEN
        CREATE TRIGGER trigger_record_fsm_history AFTER INSERT OR UPDATE ON fsm_state FOR EACH ROW EXECUTE FUNCTION record_fsm_state_history();
    END IF;
END $$;

CREATE OR REPLACE FUNCTION cleanup_old_fsm_history(retention_days INTEGER DEFAULT 90) RETURNS INTEGER AS $$
DECLARE deleted_count INTEGER;
BEGIN DELETE FROM fsm_state_history WHERE transitioned_at < NOW() - (retention_days || ' days')::INTERVAL; GET DIAGNOSTICS deleted_count = ROW_COUNT; RETURN deleted_count;
END; $$ LANGUAGE plpgsql;

-- Workflow Executions
CREATE TABLE IF NOT EXISTS workflow_executions (
    workflow_id VARCHAR(255) PRIMARY KEY,
    operation_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'RUNNING',
    current_step VARCHAR(100),
    completed_steps JSONB DEFAULT '[]'::JSONB,
    failed_step VARCHAR(100),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_ms FLOAT,
    correlation_id VARCHAR(255) NOT NULL,
    input_payload JSONB,
    output_results JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_workflow_exec_entity ON workflow_executions(entity_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_exec_status ON workflow_executions(status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_exec_operation ON workflow_executions(operation_type, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_exec_correlation ON workflow_executions(correlation_id);
CREATE INDEX IF NOT EXISTS idx_workflow_exec_time ON workflow_executions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_exec_stuck ON workflow_executions(status, updated_at) WHERE status = 'RUNNING';
CREATE INDEX IF NOT EXISTS idx_workflow_exec_metadata ON workflow_executions USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_workflow_exec_completed_steps ON workflow_executions USING GIN (completed_steps);
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_workflow_exec_updated_at') THEN
        CREATE TRIGGER trigger_workflow_exec_updated_at BEFORE UPDATE ON workflow_executions FOR EACH ROW EXECUTE FUNCTION update_fsm_state_updated_at();
    END IF;
END $$;

CREATE OR REPLACE FUNCTION calculate_workflow_duration() RETURNS TRIGGER AS $$
BEGIN IF NEW.completed_at IS NOT NULL AND OLD.completed_at IS NULL THEN NEW.duration_ms = EXTRACT(EPOCH FROM (NEW.completed_at - NEW.started_at)) * 1000; END IF; RETURN NEW;
END; $$ LANGUAGE plpgsql;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_calculate_duration') THEN
        CREATE TRIGGER trigger_calculate_duration BEFORE UPDATE ON workflow_executions FOR EACH ROW EXECUTE FUNCTION calculate_workflow_duration();
    END IF;
END $$;

CREATE OR REPLACE FUNCTION cleanup_completed_workflows(retention_days INTEGER DEFAULT 30) RETURNS INTEGER AS $$
DECLARE deleted_count INTEGER;
BEGIN DELETE FROM workflow_executions WHERE status IN ('COMPLETED', 'FAILED') AND completed_at < NOW() - (retention_days || ' days')::INTERVAL; GET DIAGNOSTICS deleted_count = ROW_COUNT; RETURN deleted_count;
END; $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION find_stuck_workflows(timeout_hours INTEGER DEFAULT 24) RETURNS TABLE (workflow_id VARCHAR(255), operation_type VARCHAR(50), current_step VARCHAR(100), stuck_duration_hours FLOAT) AS $$
BEGIN RETURN QUERY SELECT we.workflow_id, we.operation_type, we.current_step, EXTRACT(EPOCH FROM (NOW() - we.updated_at)) / 3600 AS stuck_duration_hours FROM workflow_executions we WHERE we.status = 'RUNNING' AND we.updated_at < NOW() - (timeout_hours || ' hours')::INTERVAL ORDER BY we.updated_at ASC;
END; $$ LANGUAGE plpgsql;

-- Domain Taxonomy
CREATE TABLE IF NOT EXISTS domain_taxonomy (
    id SERIAL PRIMARY KEY,
    domain_id VARCHAR(50) NOT NULL UNIQUE,
    domain_version VARCHAR(20) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_domain_taxonomy_version ON domain_taxonomy(domain_version);
INSERT INTO domain_taxonomy (domain_id, domain_version, description) VALUES
    ('code_generation', '1.0', 'Creating new code'), ('code_review', '1.0', 'Reviewing existing code'),
    ('debugging', '1.0', 'Finding and fixing bugs'), ('testing', '1.0', 'Writing or running tests'),
    ('documentation', '1.0', 'Writing docs or comments'), ('refactoring', '1.0', 'Restructuring existing code'),
    ('architecture', '1.0', 'System design decisions'), ('devops', '1.0', 'CI/CD, deployment, infra'),
    ('data_analysis', '1.0', 'Data processing and analysis'), ('general', '1.0', 'General purpose tasks')
ON CONFLICT (domain_id) DO NOTHING;

CREATE OR REPLACE FUNCTION update_domain_taxonomy_updated_at() RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$ LANGUAGE plpgsql;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_domain_taxonomy_updated_at') THEN
        CREATE TRIGGER trigger_domain_taxonomy_updated_at BEFORE UPDATE ON domain_taxonomy FOR EACH ROW EXECUTE FUNCTION update_domain_taxonomy_updated_at();
    END IF;
END $$;

-- Learned Patterns
CREATE TABLE IF NOT EXISTS learned_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_signature TEXT NOT NULL,
    domain_id VARCHAR(50) NOT NULL REFERENCES domain_taxonomy(domain_id) ON DELETE RESTRICT ON UPDATE CASCADE,
    domain_version VARCHAR(20) NOT NULL,
    domain_candidates JSONB NOT NULL DEFAULT '[]',
    keywords TEXT[],
    confidence FLOAT NOT NULL CHECK (confidence >= 0.5),
    status VARCHAR(20) NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'provisional', 'validated', 'deprecated')),
    promoted_at TIMESTAMPTZ,
    deprecated_at TIMESTAMPTZ,
    deprecation_reason TEXT,
    source_session_ids UUID[] NOT NULL,
    recurrence_count INT NOT NULL DEFAULT 1 CONSTRAINT check_recurrence_count_min CHECK (recurrence_count >= 1),
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    distinct_days_seen INT NOT NULL DEFAULT 1 CONSTRAINT check_distinct_days_seen_min CHECK (distinct_days_seen >= 1),
    quality_score FLOAT DEFAULT 0.5 CONSTRAINT check_quality_score_bounds CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    injection_count_rolling_20 INT DEFAULT 0 CONSTRAINT check_injection_count_rolling_bounds CHECK (injection_count_rolling_20 >= 0 AND injection_count_rolling_20 <= 20),
    success_count_rolling_20 INT DEFAULT 0 CONSTRAINT check_success_count_rolling_bounds CHECK (success_count_rolling_20 >= 0 AND success_count_rolling_20 <= 20),
    failure_count_rolling_20 INT DEFAULT 0 CONSTRAINT check_failure_count_rolling_bounds CHECK (failure_count_rolling_20 >= 0 AND failure_count_rolling_20 <= 20),
    failure_streak INT DEFAULT 0 CONSTRAINT check_failure_streak_non_negative CHECK (failure_streak >= 0),
    CONSTRAINT check_rolling_metrics_sum CHECK (success_count_rolling_20 + failure_count_rolling_20 <= injection_count_rolling_20),
    CONSTRAINT check_promoted_at_status_consistency CHECK (
        (status = 'candidate' AND promoted_at IS NULL) OR (status IN ('provisional', 'validated') AND promoted_at IS NOT NULL) OR (status = 'deprecated')
    ),
    version INT NOT NULL DEFAULT 1,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    supersedes UUID REFERENCES learned_patterns(id),
    superseded_by UUID REFERENCES learned_patterns(id),
    compiled_snippet TEXT,
    compiled_token_count INT,
    compiled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_signature_domain_version UNIQUE (pattern_signature, domain_id, version),
    CONSTRAINT check_no_self_supersede CHECK (supersedes IS NULL OR supersedes != id),
    CONSTRAINT check_no_self_superseded_by CHECK (superseded_by IS NULL OR superseded_by != id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_current_pattern ON learned_patterns (pattern_signature, domain_id) WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS idx_learned_patterns_domain ON learned_patterns(domain_id);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_status ON learned_patterns(status);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_domain_status ON learned_patterns(domain_id, status);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_first_seen ON learned_patterns(first_seen_at);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_last_seen ON learned_patterns(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_promotion_candidates ON learned_patterns(status, distinct_days_seen, quality_score) WHERE status IN ('candidate', 'provisional');
CREATE INDEX IF NOT EXISTS idx_learned_patterns_quality ON learned_patterns(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_failure_streak ON learned_patterns(failure_streak) WHERE failure_streak > 0;
CREATE INDEX IF NOT EXISTS idx_learned_patterns_keywords ON learned_patterns USING GIN (keywords);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_domain_candidates ON learned_patterns USING GIN (domain_candidates);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_source_sessions ON learned_patterns USING GIN (source_session_ids);

CREATE OR REPLACE FUNCTION update_learned_patterns_updated_at() RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$ LANGUAGE plpgsql;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_learned_patterns_updated_at') THEN
        CREATE TRIGGER trigger_learned_patterns_updated_at BEFORE UPDATE ON learned_patterns FOR EACH ROW EXECUTE FUNCTION update_learned_patterns_updated_at();
    END IF;
END $$;

-- Pattern Disable Events
CREATE TABLE IF NOT EXISTS pattern_disable_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL UNIQUE,
    event_type VARCHAR(20) NOT NULL CHECK (event_type IN ('disabled', 're_enabled')),
    pattern_id UUID REFERENCES learned_patterns(id) ON DELETE RESTRICT ON UPDATE CASCADE,
    pattern_class VARCHAR(100),
    reason TEXT NOT NULL,
    event_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT check_target_specified CHECK (pattern_id IS NOT NULL OR pattern_class IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_pattern_disable_events_pattern_id_latest ON pattern_disable_events(pattern_id, event_at DESC) WHERE pattern_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_pattern_disable_events_pattern_class_latest ON pattern_disable_events(pattern_class, event_at DESC) WHERE pattern_class IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_pattern_disable_events_actor_audit ON pattern_disable_events(actor, event_at DESC);
CREATE INDEX IF NOT EXISTS idx_pattern_disable_events_type ON pattern_disable_events(event_type);
CREATE INDEX IF NOT EXISTS idx_pattern_disable_events_event_at ON pattern_disable_events(event_at DESC);

-- Pattern Injections
CREATE TABLE IF NOT EXISTS pattern_injections (
    injection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    correlation_id UUID,
    pattern_ids UUID[] NOT NULL DEFAULT '{}',
    injected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    injection_context VARCHAR(30) NOT NULL CHECK (injection_context IN ('SessionStart', 'UserPromptSubmit', 'PreToolUse', 'SubagentStart')),
    cohort VARCHAR(20) NOT NULL DEFAULT 'treatment' CHECK (cohort IN ('control', 'treatment')),
    assignment_seed BIGINT NOT NULL,
    compiled_content TEXT,
    compiled_token_count INT CHECK (compiled_token_count >= 0),
    outcome_recorded BOOLEAN NOT NULL DEFAULT FALSE,
    outcome_success BOOLEAN,
    outcome_recorded_at TIMESTAMPTZ,
    outcome_failure_reason TEXT,
    contribution_heuristic JSONB,
    heuristic_method VARCHAR(50),
    heuristic_confidence FLOAT CHECK (heuristic_confidence >= 0.0 AND heuristic_confidence <= 1.0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pattern_injections_session_id ON pattern_injections(session_id);
CREATE INDEX IF NOT EXISTS idx_pattern_injections_pending_outcome ON pattern_injections(session_id, outcome_recorded) WHERE outcome_recorded = FALSE;
CREATE INDEX IF NOT EXISTS idx_pattern_injections_cohort ON pattern_injections(cohort);
CREATE INDEX IF NOT EXISTS idx_pattern_injections_injected_at ON pattern_injections(injected_at);
CREATE INDEX IF NOT EXISTS idx_pattern_injections_pattern_ids ON pattern_injections USING GIN (pattern_ids);
CREATE INDEX IF NOT EXISTS idx_pattern_injections_correlation_id ON pattern_injections(correlation_id) WHERE correlation_id IS NOT NULL;

CREATE OR REPLACE FUNCTION update_pattern_injections_updated_at() RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$ LANGUAGE plpgsql;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_pattern_injections_updated_at') THEN
        CREATE TRIGGER trigger_pattern_injections_updated_at BEFORE UPDATE ON pattern_injections FOR EACH ROW EXECUTE FUNCTION update_pattern_injections_updated_at();
    END IF;
END $$;

-- Disabled Patterns Materialized View
CREATE MATERIALIZED VIEW IF NOT EXISTS disabled_patterns_current AS
WITH ranked_events AS (
    SELECT id, event_id, event_type, pattern_id, pattern_class, reason, event_at, actor, created_at,
        ROW_NUMBER() OVER (PARTITION BY COALESCE('id:' || pattern_id::text, 'class:' || pattern_class) ORDER BY event_at DESC, created_at DESC, id DESC) AS rn
    FROM pattern_disable_events
)
SELECT id, event_id, event_type, pattern_id, pattern_class, reason, event_at, actor, created_at
FROM ranked_events WHERE rn = 1 AND event_type = 'disabled';

CREATE UNIQUE INDEX IF NOT EXISTS idx_disabled_patterns_current_pattern_id_unique ON disabled_patterns_current(pattern_id) WHERE pattern_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_disabled_patterns_current_pattern_class_unique ON disabled_patterns_current(pattern_class) WHERE pattern_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_disabled_patterns_current_pattern_class ON disabled_patterns_current(pattern_class) WHERE pattern_class IS NOT NULL;

-- Add signature_hash to learned_patterns
ALTER TABLE learned_patterns ADD COLUMN IF NOT EXISTS signature_hash TEXT;
UPDATE learned_patterns SET signature_hash = encode(digest(pattern_signature, 'sha256'), 'hex') WHERE signature_hash IS NULL;
ALTER TABLE learned_patterns ALTER COLUMN signature_hash SET NOT NULL;
ALTER TABLE learned_patterns ADD CONSTRAINT unique_signature_hash_domain_version UNIQUE (domain_id, signature_hash, version);
CREATE UNIQUE INDEX IF NOT EXISTS idx_current_pattern_hash ON learned_patterns (signature_hash, domain_id) WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS idx_learned_patterns_domain_hash ON learned_patterns(domain_id, signature_hash);

-- Pattern Lifecycle Transitions
CREATE TABLE IF NOT EXISTS pattern_lifecycle_transitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL,
    pattern_id UUID NOT NULL REFERENCES learned_patterns(id) ON DELETE RESTRICT,
    from_status VARCHAR(20) NOT NULL CHECK (from_status IN ('candidate', 'provisional', 'validated', 'deprecated')),
    to_status VARCHAR(20) NOT NULL CHECK (to_status IN ('candidate', 'provisional', 'validated', 'deprecated')),
    transition_trigger VARCHAR(50) NOT NULL,
    correlation_id UUID,
    actor VARCHAR(100),
    reason TEXT,
    gate_snapshot JSONB,
    transition_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_request_pattern_transition UNIQUE (request_id, pattern_id)
);
CREATE INDEX IF NOT EXISTS idx_pattern_lifecycle_pattern_id ON pattern_lifecycle_transitions(pattern_id);
CREATE INDEX IF NOT EXISTS idx_pattern_lifecycle_transition_at ON pattern_lifecycle_transitions(transition_at);
CREATE INDEX IF NOT EXISTS idx_pattern_lifecycle_correlation_id ON pattern_lifecycle_transitions(correlation_id) WHERE correlation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_pattern_lifecycle_trigger ON pattern_lifecycle_transitions(transition_trigger);
CREATE INDEX IF NOT EXISTS idx_pattern_lifecycle_from_to_status ON pattern_lifecycle_transitions(from_status, to_status);

-- Add evidence_tier to learned_patterns
ALTER TABLE learned_patterns ADD COLUMN IF NOT EXISTS evidence_tier TEXT NOT NULL DEFAULT 'unmeasured' CONSTRAINT check_evidence_tier_valid CHECK (evidence_tier IN ('unmeasured', 'observed', 'measured', 'verified'));
CREATE INDEX IF NOT EXISTS idx_learned_patterns_evidence_tier ON learned_patterns(evidence_tier);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_status_evidence_tier ON learned_patterns(status, evidence_tier) WHERE status IN ('candidate', 'provisional') AND is_current = TRUE;

-- Pattern Measured Attributions
CREATE TABLE IF NOT EXISTS pattern_measured_attributions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id UUID NOT NULL REFERENCES learned_patterns(id) ON DELETE RESTRICT,
    session_id UUID NOT NULL,
    run_id UUID,
    evidence_tier TEXT NOT NULL CONSTRAINT check_attribution_evidence_tier_valid CHECK (evidence_tier IN ('observed', 'measured', 'verified')),
    measured_attribution_json JSONB,
    CONSTRAINT check_attribution_json_run_consistency CHECK ((run_id IS NULL AND measured_attribution_json IS NULL) OR (run_id IS NOT NULL AND measured_attribution_json IS NOT NULL)),
    CONSTRAINT check_attribution_evidence_run_consistency CHECK ((run_id IS NULL AND evidence_tier = 'observed') OR (run_id IS NOT NULL AND evidence_tier IN ('measured', 'verified'))),
    correlation_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pattern_attributions_pattern_created ON pattern_measured_attributions(pattern_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pattern_attributions_session ON pattern_measured_attributions(session_id);
CREATE INDEX IF NOT EXISTS idx_pattern_attributions_run_id ON pattern_measured_attributions(run_id) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_pattern_attributions_correlation ON pattern_measured_attributions(correlation_id) WHERE correlation_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_pattern_measured_attributions_idempotent ON pattern_measured_attributions (pattern_id, session_id, run_id) WHERE run_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_pattern_measured_attributions_idempotent_observed ON pattern_measured_attributions (pattern_id, session_id) WHERE run_id IS NULL;

-- Add run_id to pattern_injections
ALTER TABLE pattern_injections ADD COLUMN IF NOT EXISTS run_id UUID;
CREATE INDEX IF NOT EXISTS idx_pattern_injections_run_id ON pattern_injections(run_id) WHERE run_id IS NOT NULL;

EOSQL

echo ""
echo "============================================="
echo "001: omniintelligence schema provisioned."
echo "============================================="
