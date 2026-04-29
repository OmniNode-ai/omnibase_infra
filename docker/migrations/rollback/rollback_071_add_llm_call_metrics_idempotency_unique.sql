-- Rollback: forward/071_add_llm_call_metrics_idempotency_unique.sql
-- Description: Remove LLM call metric idempotency uniqueness (OMN-10332)
-- Created: 2026-04-29

DROP INDEX IF EXISTS idx_llm_call_metrics_idempotency_unique;
