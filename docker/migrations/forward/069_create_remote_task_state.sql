-- =============================================================================
-- MIGRATION: Create remote_task_state table
-- =============================================================================
-- Ticket: OMN-9631
-- Epic: OMN-9620 (A2A Delegation Bridge)
-- Version: 1.0.0
--
-- PURPOSE:
--   Creates durable state for remote agent task invocations. This table lets
--   node_remote_agent_invoke_effect resume unfinished remote tasks after restart.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   docker/migrations/rollback/rollback_069_drop_remote_task_state.sql
--
-- NOTE:
--   Planned as migration 068, but origin/main already uses 068 for
--   landing_content_projection. This migration uses the next available number.
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.remote_task_state (
    task_id UUID PRIMARY KEY,
    invocation_kind TEXT NOT NULL,
    protocol TEXT,
    target_ref TEXT NOT NULL,
    remote_task_handle TEXT,
    correlation_id UUID NOT NULL,
    status TEXT NOT NULL,
    last_remote_status TEXT,
    last_emitted_event_type TEXT,
    submitted_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    error TEXT,

    CONSTRAINT remote_task_state_invocation_kind_check CHECK (
        invocation_kind IN ('agent', 'model')
    ),
    CONSTRAINT remote_task_state_protocol_check CHECK (
        protocol IS NULL OR protocol IN ('A2A')
    ),
    CONSTRAINT remote_task_state_status_check CHECK (
        status IN (
            'SUBMITTED',
            'ACCEPTED',
            'PROGRESS',
            'ARTIFACT',
            'COMPLETED',
            'FAILED',
            'TIMED_OUT',
            'CANCELED'
        )
    ),
    CONSTRAINT remote_task_state_last_emitted_event_type_check CHECK (
        last_emitted_event_type IS NULL OR last_emitted_event_type IN (
            'SUBMITTED',
            'ACCEPTED',
            'PROGRESS',
            'ARTIFACT',
            'COMPLETED',
            'FAILED',
            'TIMED_OUT',
            'CANCELED'
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_remote_task_state_status
    ON public.remote_task_state (status);

CREATE INDEX IF NOT EXISTS idx_remote_task_state_correlation_id
    ON public.remote_task_state (correlation_id);

COMMENT ON TABLE public.remote_task_state IS
    'Durable remote agent task state for A2A delegation bridge restart recovery. '
    'Rows are created and updated by node_remote_agent_invoke_effect. (OMN-9631)';

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '069',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
