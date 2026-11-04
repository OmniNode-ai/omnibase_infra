-- Adds the deterministic row key node_projection_context_audit upserts on.
--
-- public.context_audit_events is created by omnibase_infra migration 053 with a
-- BIGSERIAL id and no natural key, so the retired consumer module could only
-- INSERT and a replay wrote every event again. This node derives event_key from
-- the event's Kafka coordinates (topic, partition, offset); the writer's
-- INSERT ... ON CONFLICT (event_key) DO UPDATE then keeps one row per event.
--
-- Rows written by the retired module keep event_key NULL. A unique index treats
-- NULLs as distinct, so they stay as they are.
--
-- The application database does not carry omnibase_infra's flat migration 053
-- (that set belongs to the separate omnibase_infra service database), so the
-- table is created here with the same shape when absent. Where it already
-- exists the CREATE is a no-op. The file is safe to re-run.

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

-- ---- BEGIN OMN-15376 shape reconciliation: context_audit_events ----
-- A pre-existing drifted table is converged onto the shape declared above with
-- guarded adds, then NOT NULL and the primary key are restored. On the
-- fresh-create path every statement is a no-op. No DROP, no recreate: rows are
-- preserved, and a column that cannot be made NOT NULL fails loud.

ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS id BIGSERIAL;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS task_id TEXT;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS parent_task_id TEXT;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS correlation_id TEXT;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS contract_id TEXT;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS event_type TEXT;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS enforcement_level TEXT;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS enforcement_action TEXT;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS violation_details JSONB;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS context_tokens_used INT;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS context_budget_tokens INT;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS return_tokens INT;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS return_max_tokens INT;
ALTER TABLE public.context_audit_events ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

-- SET NOT NULL fails loud (PostgreSQL names the column) when a pre-existing row
-- holds NULL; that needs a data ruling, so the migration refuses to guess.
ALTER TABLE public.context_audit_events ALTER COLUMN id SET NOT NULL;
ALTER TABLE public.context_audit_events ALTER COLUMN task_id SET NOT NULL;
ALTER TABLE public.context_audit_events ALTER COLUMN correlation_id SET NOT NULL;
ALTER TABLE public.context_audit_events ALTER COLUMN event_type SET NOT NULL;
ALTER TABLE public.context_audit_events ALTER COLUMN enforcement_level SET NOT NULL;
ALTER TABLE public.context_audit_events ALTER COLUMN created_at SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'public.context_audit_events'::regclass AND contype = 'p'
    ) THEN
        ALTER TABLE public.context_audit_events
            ADD CONSTRAINT context_audit_events_pkey PRIMARY KEY (id);
    END IF;
END$$;

-- ---- END OMN-15376 shape reconciliation: context_audit_events ----

CREATE INDEX IF NOT EXISTS idx_context_audit_correlation
    ON public.context_audit_events (correlation_id);
CREATE INDEX IF NOT EXISTS idx_context_audit_task
    ON public.context_audit_events (task_id);
CREATE INDEX IF NOT EXISTS idx_context_audit_type
    ON public.context_audit_events (event_type);
CREATE INDEX IF NOT EXISTS idx_context_audit_created
    ON public.context_audit_events (created_at);

ALTER TABLE public.context_audit_events
    ADD COLUMN IF NOT EXISTS event_key TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_context_audit_events_event_key
    ON public.context_audit_events (event_key);

COMMENT ON COLUMN public.context_audit_events.event_key IS
    'Deterministic key of the source event: sha256 of topic:partition:offset in UUID shape. NULL on rows written before node_projection_context_audit.';
