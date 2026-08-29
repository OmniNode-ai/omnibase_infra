-- OMN-16924: durable session phase reducer state, keyed on session_id.
--
-- Replaces `node_session_phase_reducer`'s previous state of record: a
-- cwd-relative `.onex_state/session/phase_state.yaml` file. The runtime
-- container's cwd is /app (root:root 0755) while the process runs as
-- omniinfra, so every bus dispatch of that reducer raised
-- `PermissionError: [Errno 13] Permission denied: '.onex_state'` and DLQ'd —
-- a 100% failure rate on all three subscribed topics, on every lane.
--
-- Operator ruling 2026-08-29: "onex_state should be configurable via contract
-- overlay right? for our purposes, state should only be kept in the database."
-- So the reducer's state of record is this table, the handler performs no I/O
-- at all, and the runtime's state_io dispatch seam
-- (omnibase_infra/runtime/auto_wiring/handler_wiring.py,
-- omnibase_infra/runtime/state_io/state_store_adapter.py) does the read before
-- handle() and the CAS-write after — the same seam migration 090 introduced
-- for delegation workflow state.
--
-- Why a SEPARATE table rather than reusing delegation_workflow_state: the two
-- rows are keyed on different things and have different lifetimes. A
-- multi-leg orchestrator keys on correlation_id (the only id every leg
-- carries); a REDUCER folds per DOMAIN entity, and this reducer's entity is
-- the SESSION. `HandlerSessionPhaseReducer.delta` explicitly rejects an event
-- whose session_id does not match the folded state's, so session_id — not
-- correlation_id — is its identity. The contract declares `state_io.key:
-- session_id`, which names BOTH the wire payload field and this primary-key
-- column.
--
-- Column shape is otherwise identical to migration 090 + 093 because the SAME
-- StateStoreAdapter reads and writes it: `payload` is opaque JSONB (infra
-- never decodes its business shape — the omnimarket-side codec owns that),
-- `tenant_id` / `state` / `in_flight` are the infra-owned denormalized columns
-- the wiring seam extracts from well-known top-level payload keys, and
-- `pending_emissions` / `publish_attempts` are the in-row outbox columns the
-- adapter's SQL references unconditionally.
--
-- Targets the omnibase_infra database via the flat forward-migration set
-- (POSTGRES_DB=omnibase_infra in docker-compose.infra.yml's forward-migration
-- service), matching migration 090 — NOT node-vendored under
-- docker/migrations/forward/nodes/, which applies to NODE_PGDB.
--
-- Idempotent CREATE so warm dev/stability volumes reconcile cleanly.

CREATE TABLE IF NOT EXISTS session_phase_state (
    session_id        TEXT PRIMARY KEY,
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

-- Give-up staleness sweep predicate (StateStoreAdapter.recover_stale_rows).
-- A reducer fold never sets in_flight, so this index stays empty in steady
-- state; it exists so the adapter's shared sweep is cheap here too.
CREATE INDEX IF NOT EXISTS ix_session_phase_state_stale_sweep
    ON session_phase_state (updated_at)
    WHERE state NOT IN ('COMPLETED', 'FAILED') AND in_flight;

-- Recovery-select predicate (StateStoreAdapter.select_recoverable_batches).
CREATE INDEX IF NOT EXISTS ix_session_phase_state_recoverable_batches
    ON session_phase_state (updated_at)
    WHERE in_flight AND pending_emissions IS NOT NULL;

CREATE OR REPLACE FUNCTION refresh_session_phase_state_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_session_phase_state_updated_at
    ON session_phase_state;
CREATE TRIGGER trg_session_phase_state_updated_at
    BEFORE UPDATE ON session_phase_state
    FOR EACH ROW
    EXECUTE FUNCTION refresh_session_phase_state_updated_at();
