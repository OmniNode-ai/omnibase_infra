-- Rollback: 026_injection_effectiveness_tables
-- Description: Remove injection effectiveness tables (OMN-1890)
-- Created: 2026-02-04
--
-- Usage: Execute this script to completely remove all objects created by
--        026_injection_effectiveness_tables.sql
--
-- Order: Objects are dropped in reverse dependency order to avoid errors.

-- ============================================================================
-- DROP TRIGGERS
-- ============================================================================

DROP TRIGGER IF EXISTS trigger_injection_effectiveness_updated_at ON injection_effectiveness;
DROP FUNCTION IF EXISTS update_injection_effectiveness_updated_at();

DROP TRIGGER IF EXISTS trigger_pattern_hit_rates_updated_at ON pattern_hit_rates;
DROP FUNCTION IF EXISTS update_pattern_hit_rates_updated_at();

-- ============================================================================
-- DROP INDEXES (explicitly, though CASCADE on table drop would handle these)
-- ============================================================================

-- pattern_hit_rates indexes
DROP INDEX IF EXISTS idx_pattern_hit_rates_updated_at;
DROP INDEX IF EXISTS idx_pattern_hit_rates_confident;
DROP INDEX IF EXISTS idx_pattern_hit_rates_domain_id;
DROP INDEX IF EXISTS idx_pattern_hit_rates_pattern_id;

-- latency_breakdowns indexes
DROP INDEX IF EXISTS idx_latency_breakdowns_cache_hit;
DROP INDEX IF EXISTS idx_latency_breakdowns_cohort_created;
DROP INDEX IF EXISTS idx_latency_breakdowns_session_id;

-- injection_effectiveness indexes
DROP INDEX IF EXISTS idx_injection_effectiveness_routing_path;
DROP INDEX IF EXISTS idx_injection_effectiveness_realm;
DROP INDEX IF EXISTS idx_injection_effectiveness_utilization_method;
DROP INDEX IF EXISTS idx_injection_effectiveness_cohort_created;
DROP INDEX IF EXISTS idx_injection_effectiveness_cohort;
DROP INDEX IF EXISTS idx_injection_effectiveness_correlation_id;
DROP INDEX IF EXISTS idx_injection_effectiveness_updated_at;
DROP INDEX IF EXISTS idx_injection_effectiveness_created_at;

-- ============================================================================
-- DROP TABLES
-- ============================================================================

DROP TABLE IF EXISTS pattern_hit_rates;
DROP TABLE IF EXISTS latency_breakdowns;
DROP TABLE IF EXISTS injection_effectiveness;
