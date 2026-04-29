-- Rollback: forward/072_add_llm_call_metrics_attribution.sql
-- Description: Remove LLM call metric repository and machine attribution (OMN-10333)
-- Created: 2026-04-29

DROP INDEX IF EXISTS idx_llm_call_metrics_machine_id;
DROP INDEX IF EXISTS idx_llm_call_metrics_repo_name;

ALTER TABLE llm_call_metrics
    DROP COLUMN IF EXISTS machine_id,
    DROP COLUMN IF EXISTS repo_name;
