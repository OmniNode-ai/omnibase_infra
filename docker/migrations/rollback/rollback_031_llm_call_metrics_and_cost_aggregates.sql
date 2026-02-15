-- Rollback: forward/031_create_llm_call_metrics_and_cost_aggregates.sql
-- Description: Remove llm_call_metrics and llm_cost_aggregates tables (OMN-2236)
-- Created: 2026-02-15
--
-- WARNING: THIS ROLLBACK WILL PERMANENTLY DELETE ALL DATA IN BOTH TABLES.
-- This operation is IRREVERSIBLE. Ensure you have backups before proceeding.
-- All LLM call metrics and cost aggregation data will be lost.
--
-- Order: Objects are dropped in reverse dependency order to avoid errors.

-- ============================================================================
-- DROP TRIGGERS
-- ============================================================================

DROP TRIGGER IF EXISTS trigger_llm_cost_aggregates_updated_at ON llm_cost_aggregates;
DROP FUNCTION IF EXISTS update_llm_cost_aggregates_updated_at();

-- ============================================================================
-- DROP INDEXES: llm_cost_aggregates
-- ============================================================================

DROP INDEX IF EXISTS idx_llm_cost_aggregates_updated_at;
DROP INDEX IF EXISTS idx_llm_cost_aggregates_window;
DROP INDEX IF EXISTS idx_llm_cost_aggregates_aggregation_key;

-- ============================================================================
-- DROP INDEXES: llm_call_metrics
-- ============================================================================

DROP INDEX IF EXISTS idx_llm_call_metrics_input_hash;
DROP INDEX IF EXISTS idx_llm_call_metrics_usage_source;
DROP INDEX IF EXISTS idx_llm_call_metrics_model_created;
DROP INDEX IF EXISTS idx_llm_call_metrics_created_at;
DROP INDEX IF EXISTS idx_llm_call_metrics_correlation_id;
DROP INDEX IF EXISTS idx_llm_call_metrics_model_id;
DROP INDEX IF EXISTS idx_llm_call_metrics_run_id;
DROP INDEX IF EXISTS idx_llm_call_metrics_session_id;

-- ============================================================================
-- DROP TABLES
-- ============================================================================

DROP TABLE IF EXISTS llm_cost_aggregates;
DROP TABLE IF EXISTS llm_call_metrics;

-- ============================================================================
-- DROP ENUMS
-- ============================================================================

DROP TYPE IF EXISTS cost_aggregation_window;
DROP TYPE IF EXISTS usage_source_type;

-- ============================================================================
-- REVERT DB_METADATA
-- ============================================================================

UPDATE public.db_metadata
SET schema_version = '030', updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
