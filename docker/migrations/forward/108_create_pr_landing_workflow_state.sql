-- OMN-19829: durable per-PR landing workflow state, keyed on landing_key.
--
-- node_pr_landing_orchestrator (omnimarket) declares
--   state_io: {database: omnibase_infra, table: pr_landing_workflow_state,
--              key: landing_key, codec: ...state_codec.StateIoCodec}
-- and the runtime's state_io dispatch seam
-- (omnibase_infra/runtime/auto_wiring/handler_wiring.py,
-- omnibase_infra/runtime/state_io/state_store_adapter.py) loads this row before
-- each leg and CAS-writes it after, publishing the leg's emissions from the
-- in-row outbox. Without the table every leg's load fails and the orchestrator
-- consumes, raises and DLQs every message.
--
-- Why a SEPARATE table: the row is keyed on the PR, not on a correlation_id or
-- a session. landing_key is `owner/repo#<number>`; every ingress model the
-- orchestrator consumes derives it from its repository and pull request number,
-- and the declared key names BOTH that payload field and this primary-key
-- column (OMN-16924).
--
-- Column shape is identical to migration 102 (and to 090 + 093) because the
-- SAME StateStoreAdapter reads and writes it: `payload` is opaque JSONB (infra
-- never decodes its business shape; the omnimarket codec owns that),
-- `tenant_id` / `state` / `in_flight` are the infra-owned denormalized columns
-- the wiring seam extracts from well-known top-level payload keys, and
-- `pending_emissions` / `publish_attempts` are the in-row outbox columns the
-- adapter's SQL references unconditionally.
--
-- Targets the omnibase_infra database via the flat forward-migration set
-- (POSTGRES_DB=omnibase_infra in docker-compose.infra.yml's forward-migration
-- service), matching migrations 090 and 102, NOT node-vendored under
-- docker/migrations/forward/nodes/, which applies to NODE_PGDB. The runtime's
-- TABLE grant is derived from the STATE_IO_TABLE_DECLARATIONS entry in
-- src/omnibase_infra/topology/table_grant_derivation.py.
--
-- Idempotent CREATE so warm dev/stability volumes reconcile cleanly.

CREATE TABLE IF NOT EXISTS public.pr_landing_workflow_state (
    landing_key       TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL,
    state             TEXT NOT NULL,
    in_flight         BOOLEAN NOT NULL DEFAULT FALSE,
    payload           JSONB NOT NULL,
    version           INTEGER NOT NULL DEFAULT 0,
    pending_emissions JSONB,
    publish_attempts  INTEGER NOT NULL DEFAULT 0,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Give-up staleness sweep predicate (StateStoreAdapter.recover_stale_rows),
-- the same predicate the adapter's shared sweep issues against every state_io
-- table.
CREATE INDEX IF NOT EXISTS ix_pr_landing_workflow_state_stale_sweep
    ON public.pr_landing_workflow_state (updated_at)
    WHERE state NOT IN ('COMPLETED', 'FAILED') AND in_flight;

-- Recovery-select predicate (StateStoreAdapter.select_recoverable_batches).
CREATE INDEX IF NOT EXISTS ix_pr_landing_workflow_state_recoverable_batches
    ON public.pr_landing_workflow_state (updated_at)
    WHERE in_flight AND pending_emissions IS NOT NULL;

-- updated_at refresh trigger, the same shape as migrations 090 and 102:
-- StateStoreAdapter's seed/cas_update SQL never writes updated_at itself.
CREATE OR REPLACE FUNCTION public.refresh_pr_landing_workflow_state_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_pr_landing_workflow_state_updated_at
    ON public.pr_landing_workflow_state;
CREATE TRIGGER trg_pr_landing_workflow_state_updated_at
    BEFORE UPDATE ON public.pr_landing_workflow_state
    FOR EACH ROW
    EXECUTE FUNCTION public.refresh_pr_landing_workflow_state_updated_at();
