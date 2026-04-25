-- Migration: 069_create_remote_task_state
-- Description: Persisted remote task lifecycle state for A2A delegation bridge resumption.
-- Idempotency: CREATE TABLE/INDEX use IF NOT EXISTS; safe to re-apply.

CREATE TABLE IF NOT EXISTS remote_task_state (
    task_id                  UUID         NOT NULL,
    invocation_kind          TEXT         NOT NULL,
    protocol                 TEXT,
    target_ref               TEXT         NOT NULL,
    remote_task_handle       TEXT,
    correlation_id           UUID         NOT NULL,
    status                   TEXT         NOT NULL,
    last_remote_status       TEXT,
    last_emitted_event_type  TEXT,
    submitted_at             TIMESTAMPTZ  NOT NULL,
    updated_at               TIMESTAMPTZ  NOT NULL,
    completed_at             TIMESTAMPTZ,
    error                    TEXT,

    CONSTRAINT pk_remote_task_state PRIMARY KEY (task_id),
    CONSTRAINT chk_remote_task_state_invocation_kind
        CHECK (invocation_kind IN ('agent', 'model')),
    CONSTRAINT chk_remote_task_state_protocol
        CHECK (protocol IS NULL OR protocol IN ('A2A')),
    CONSTRAINT chk_remote_task_state_status
        CHECK (
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
    CONSTRAINT chk_remote_task_state_last_emitted
        CHECK (
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
    ON remote_task_state (status);

CREATE INDEX IF NOT EXISTS idx_remote_task_state_correlation_id
    ON remote_task_state (correlation_id);

UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '069',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
