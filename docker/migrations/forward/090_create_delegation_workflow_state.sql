-- OMN-14208: durable, tenant-scoped delegation FSM working state.
--
-- Slice-1 of the multitenant runtime work: graduates delegation FSM state off
-- the process-local ClassVar dict (`_shared_workflows`, lost on every restart
-- / never true across processes) onto a durable row loaded fresh per leg and
-- CAS-persisted before the leg's events publish. See the runtime seam in
-- omnibase_infra/runtime/auto_wiring/handler_wiring.py (opt-in `state_io`
-- contract binding) and omnibase_infra/runtime/state_io/state_store_adapter.py.
--
-- correlation_id is the ONLY key every leg's wire payload carries (legs 2-5
-- have no tenant_id on the wire) — PRIMARY KEY on correlation_id alone, NOT a
-- composite (tenant_id, correlation_id) key, since correlation_id is already
-- a globally-unique UUID and a composite key would permit an ambiguous read.
--
-- `payload` is opaque JSONB from this table's perspective: omnibase_infra
-- never decodes its business shape. `tenant_id` / `state` / `in_flight` are
-- denormalized top-level columns (extracted from well-known keys on the
-- opaque payload by the wiring seam) so staleness sweeps and dashboards can
-- filter/index without decoding the payload.
--
-- Targets the omnibase_infra database via the flat forward-migration set
-- (POSTGRES_DB=omnibase_infra in docker-compose.infra.yml's forward-migration
-- service) — NOT node-vendored under docker/migrations/forward/nodes/, which
-- applies to NODE_PGDB=omnidash_analytics (the forbidden target for this
-- table per OMN-13829 / the corrected slice-1 design).
--
-- Idempotent CREATE so warm dev/stability volumes reconcile cleanly.

CREATE TABLE IF NOT EXISTS delegation_workflow_state (
    correlation_id TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL,
    state          TEXT NOT NULL,
    in_flight      BOOLEAN NOT NULL DEFAULT FALSE,
    payload        JSONB NOT NULL,
    version        INTEGER NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_delegation_workflow_state_stale_sweep
    ON delegation_workflow_state (updated_at)
    WHERE state NOT IN ('COMPLETED', 'FAILED') AND in_flight;

CREATE OR REPLACE FUNCTION refresh_delegation_workflow_state_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_delegation_workflow_state_updated_at
    ON delegation_workflow_state;
CREATE TRIGGER trg_delegation_workflow_state_updated_at
    BEFORE UPDATE ON delegation_workflow_state
    FOR EACH ROW
    EXECUTE FUNCTION refresh_delegation_workflow_state_updated_at();
