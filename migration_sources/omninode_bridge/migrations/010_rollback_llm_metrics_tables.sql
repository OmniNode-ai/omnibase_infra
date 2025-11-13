-- Rollback Migration: 010_rollback_llm_metrics_tables
-- Description: Rollback LLM metrics tables creation
-- Rollback for: 010_create_llm_metrics_tables
-- Created: 2025-10-31

-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS llm_patterns;
DROP TABLE IF EXISTS llm_generation_history;
DROP TABLE IF EXISTS llm_context_windows;
DROP TABLE IF EXISTS llm_generation_metrics;
