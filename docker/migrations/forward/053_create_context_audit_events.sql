-- =============================================================================
-- MIGRATION: Create context_audit_events table
-- =============================================================================
-- Ticket: OMN-5239 (Task 9: Add context_audit_events DB migration)
-- Version: 1.0.0
--
-- PURPOSE:
--   Creates the operational audit store for enforcement decisions, CLI
--   queries, and replay. Each row captures a single enforcement event
--   including token budget usage and the action taken.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   See rollback/rollback_053_create_context_audit_events.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.context_audit_events (
    id                      BIGSERIAL       PRIMARY KEY,
    task_id                 TEXT            NOT NULL,
    parent_task_id          TEXT,
    correlation_id          TEXT            NOT NULL,
    contract_id             TEXT,
    event_type              TEXT            NOT NULL,
    enforcement_level       TEXT            NOT NULL,
    enforcement_action      TEXT,
    violation_details       JSONB,
    context_tokens_used     INT,
    context_budget_tokens   INT,
    return_tokens           INT,
    return_max_tokens       INT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_context_audit_correlation
    ON context_audit_events (correlation_id);

CREATE INDEX IF NOT EXISTS idx_context_audit_task
    ON context_audit_events (task_id);

CREATE INDEX IF NOT EXISTS idx_context_audit_type
    ON context_audit_events (event_type);

CREATE INDEX IF NOT EXISTS idx_context_audit_created
    ON context_audit_events (created_at);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE context_audit_events IS
    'Operational audit store for enforcement decisions, CLI queries, and replay (OMN-5239). '
    'Each row represents a single enforcement event with token budget usage and action taken.';

COMMENT ON COLUMN context_audit_events.task_id IS
    'Unique identifier for the task that triggered this audit event.';

COMMENT ON COLUMN context_audit_events.parent_task_id IS
    'Parent task identifier for nested task hierarchies (NULL if top-level).';

COMMENT ON COLUMN context_audit_events.correlation_id IS
    'Correlation ID for tracing events across services.';

COMMENT ON COLUMN context_audit_events.contract_id IS
    'ONEX contract identifier that declared the enforcement rule (NULL if not contract-bound).';

COMMENT ON COLUMN context_audit_events.event_type IS
    'Classification of the audit event (e.g. enforcement_check, budget_exceeded).';

COMMENT ON COLUMN context_audit_events.enforcement_level IS
    'Enforcement strictness level at the time of the event.';

COMMENT ON COLUMN context_audit_events.enforcement_action IS
    'Action taken by the enforcer (NULL if no action was required).';

COMMENT ON COLUMN context_audit_events.violation_details IS
    'JSON object with structured detail about any violation. NULL if no violation.';

COMMENT ON COLUMN context_audit_events.context_tokens_used IS
    'Number of context tokens consumed at the time of the event.';

COMMENT ON COLUMN context_audit_events.context_budget_tokens IS
    'Context token budget declared for this task.';

COMMENT ON COLUMN context_audit_events.return_tokens IS
    'Return tokens consumed at the time of the event.';

COMMENT ON COLUMN context_audit_events.return_max_tokens IS
    'Maximum return tokens allowed for this task.';

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '053',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
