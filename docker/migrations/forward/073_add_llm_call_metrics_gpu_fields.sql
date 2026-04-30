-- Migration: 073_add_llm_call_metrics_gpu_fields
-- Predecessor: 072_add_llm_call_metrics_attribution
-- Description: Add GPU usage evidence and aggregate compute cost columns (OMN-10338)
-- Created: 2026-04-29
--
-- Purpose:
--   Local-model calls have zero token/API cost but consume GPU time. These
--   nullable evidence columns preserve historical compatibility while allowing
--   projections to compute compute_cost_usd from the pricing manifest.
--
-- Rollback: See rollback/rollback_073_add_llm_call_metrics_gpu_fields.sql

ALTER TABLE llm_call_metrics
    ADD COLUMN IF NOT EXISTS gpu_seconds NUMERIC(10, 3),
    ADD COLUMN IF NOT EXISTS gpu_type VARCHAR(64),
    ADD COLUMN IF NOT EXISTS gpu_count SMALLINT,
    ADD COLUMN IF NOT EXISTS compute_usage_source usage_source_type;

CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_gpu_type
    ON llm_call_metrics (gpu_type)
    WHERE gpu_type IS NOT NULL;

ALTER TABLE llm_cost_aggregates
    ADD COLUMN IF NOT EXISTS compute_cost_usd NUMERIC(14, 6) NOT NULL DEFAULT 0;

COMMENT ON COLUMN llm_call_metrics.gpu_seconds IS
    'Measured or estimated GPU wall-clock seconds for local-model inference (OMN-10338).';

COMMENT ON COLUMN llm_call_metrics.gpu_type IS
    'Configured GPU type used for local-model inference, e.g. rtx_5090 (OMN-10338).';

COMMENT ON COLUMN llm_call_metrics.gpu_count IS
    'Configured GPU count used for local-model inference (OMN-10338).';

COMMENT ON COLUMN llm_call_metrics.compute_usage_source IS
    'Provenance for GPU usage evidence using usage_source_type enum (OMN-10338).';

COMMENT ON COLUMN llm_cost_aggregates.compute_cost_usd IS
    'Projected GPU compute cost in USD, computed from gpu_seconds and pricing_manifest compute_cost rates (OMN-10338).';

UPDATE public.db_metadata
SET schema_version = '073', updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
