-- Migration: 071_add_llm_call_metrics_idempotency_unique
-- Predecessor: 070_build_loop_runs
-- Description: Enforce idempotency for LLM call metric ingestion (OMN-10332)
-- Created: 2026-04-29
--
-- Purpose:
--   Session-end emitters can replay the same logical LLM usage event. This
--   unique index makes the existing input_hash column authoritative for
--   deduplicating those replays without comparing full payloads.
--
-- Rollback: See rollback/rollback_071_add_llm_call_metrics_idempotency_unique.sql

CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_call_metrics_idempotency_unique
    ON llm_call_metrics (
        model_id,
        session_id,
        COALESCE(run_id, ''),
        input_hash
    )
    WHERE input_hash IS NOT NULL;

COMMENT ON INDEX idx_llm_call_metrics_idempotency_unique IS
    'Deduplicates replayed LLM call metric events by model, session, run, and input hash (OMN-10332).';
