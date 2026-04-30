-- Rollback: forward/073_add_llm_call_metrics_gpu_fields.sql
-- Description: Remove LLM call metric GPU evidence and aggregate compute cost (OMN-10338)
-- Created: 2026-04-29

DROP INDEX IF EXISTS idx_llm_call_metrics_gpu_type;

ALTER TABLE llm_cost_aggregates
    DROP COLUMN IF EXISTS compute_cost_usd;

ALTER TABLE llm_call_metrics
    DROP COLUMN IF EXISTS compute_usage_source,
    DROP COLUMN IF EXISTS gpu_count,
    DROP COLUMN IF EXISTS gpu_type,
    DROP COLUMN IF EXISTS gpu_seconds;

UPDATE public.db_metadata
SET schema_version = '072', updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
