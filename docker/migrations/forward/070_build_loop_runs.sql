-- =============================================================================
-- MIGRATION: Create build_loop_runs table
-- =============================================================================
-- Ticket: OMN-9774
-- Epic: OMN-8943 (Prove Full Four-Node ONEX Pattern End-to-End)
-- Plan: docs/plans/2026-04-25-overnight-p0-integration-plan.md (Wave 2.5a)
-- Version: 1.0.0
--
-- PURPOSE:
--   Append-only audit table receiving one row per terminal
--   `onex.evt.omnimarket.build-loop-orchestrator-completed.v1` event.
--   Consumed by node_build_loop_projection_compute and persisted by
--   node_build_loop_write_effect.
--
-- DESIGN — APPEND-ONLY (intentional divergence from event_ledger):
--   No unique constraint on run_id. If a workflow emits the terminal event
--   more than once during retries, both rows persist so duplicates are
--   observable rather than silently swallowed. Contrast with event_ledger,
--   which uses ON CONFLICT DO NOTHING on (topic, partition, kafka_offset)
--   for kafka-position idempotency.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   docker/migrations/rollback/rollback_070_build_loop_runs.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.build_loop_runs (
    id UUID PRIMARY KEY,
    run_id TEXT NOT NULL,
    workflow_name TEXT NOT NULL,
    event_type TEXT NOT NULL,
    terminal_event_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS build_loop_runs_workflow_created_idx
    ON public.build_loop_runs (workflow_name, created_at DESC);
