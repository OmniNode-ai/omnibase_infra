-- Rollback Migration: 013_rollback_llm_metrics_tables
-- Description: Rollback LLM metrics, history, and patterns tables
-- Rollback for: 013_create_llm_metrics_tables
-- Created: 2025-10-31

-- Drop trigger first
DROP TRIGGER IF EXISTS trigger_update_llm_patterns_updated_at ON llm_patterns;
DROP FUNCTION IF EXISTS update_llm_patterns_updated_at();

-- Drop tables in reverse order (respecting foreign key dependencies)
DROP TABLE IF EXISTS llm_patterns CASCADE;
DROP TABLE IF EXISTS llm_generation_history CASCADE;
DROP TABLE IF EXISTS llm_generation_metrics CASCADE;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Rollback 013_rollback_llm_metrics_tables completed successfully';
    RAISE NOTICE 'Dropped tables: llm_patterns, llm_generation_history, llm_generation_metrics';
    RAISE NOTICE 'Dropped trigger and function for updated_at';
END $$;
