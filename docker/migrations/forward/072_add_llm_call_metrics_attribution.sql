-- Migration: 072_add_llm_call_metrics_attribution
-- Predecessor: 071_add_llm_call_metrics_idempotency_unique
-- Description: Add repository and machine attribution to LLM call metrics (OMN-10333)
-- Created: 2026-04-29
--
-- Purpose:
--   Cost and usage projections need durable repo and machine attribution on
--   raw LLM call metrics. These columns are nullable to preserve compatibility
--   with historical events and producers that have not yet been upgraded.
--
-- Rollback: See rollback/rollback_072_add_llm_call_metrics_attribution.sql

ALTER TABLE llm_call_metrics
    ADD COLUMN IF NOT EXISTS repo_name VARCHAR(255),
    ADD COLUMN IF NOT EXISTS machine_id VARCHAR(255);

CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_repo_name
    ON llm_call_metrics (repo_name)
    WHERE repo_name IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_machine_id
    ON llm_call_metrics (machine_id)
    WHERE machine_id IS NOT NULL;

COMMENT ON COLUMN llm_call_metrics.repo_name IS
    'Repository name attributed by the LLM call producer for cost and usage analysis (OMN-10333).';

COMMENT ON COLUMN llm_call_metrics.machine_id IS
    'Machine identifier attributed by the LLM call producer for cost and usage analysis (OMN-10333).';
