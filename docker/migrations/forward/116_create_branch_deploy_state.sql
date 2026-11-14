-- OMN-24118: durable branch deploy state, keyed on the deploy's correlation id.
--
-- node_branch_deploy_orchestrator (omnimarket) declares
--   state_io: {database: omnibase_infra, table: branch_deploy_state,
--              key: correlation_id}
-- and the row is the one stored deploy record of a branch deploy keyed on the
-- deploy's correlation id.
--
-- Idempotent CREATE so warm dev/stability volumes reconcile cleanly.

CREATE TABLE IF NOT EXISTS public.branch_deploy_state (
    correlation_id    TEXT PRIMARY KEY,
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

CREATE INDEX IF NOT EXISTS ix_branch_deploy_state_stale_sweep
    ON public.branch_deploy_state (updated_at)
    WHERE state NOT IN ('COMPLETED', 'FAILED') AND in_flight;

CREATE INDEX IF NOT EXISTS ix_branch_deploy_state_recoverable_batches
    ON public.branch_deploy_state (updated_at)
    WHERE in_flight AND pending_emissions IS NOT NULL;

CREATE OR REPLACE FUNCTION public.refresh_branch_deploy_state_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_branch_deploy_state_updated_at
    ON public.branch_deploy_state;
CREATE TRIGGER trg_branch_deploy_state_updated_at
    BEFORE UPDATE ON public.branch_deploy_state
    FOR EACH ROW
    EXECUTE FUNCTION public.refresh_branch_deploy_state_updated_at();

DO $grant$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'role_omnibase_infra') THEN
        GRANT SELECT, INSERT, UPDATE ON TABLE public.branch_deploy_state TO role_omnibase_infra;

        IF NOT has_table_privilege('role_omnibase_infra', 'public.branch_deploy_state', 'SELECT')
            OR NOT has_table_privilege('role_omnibase_infra', 'public.branch_deploy_state', 'INSERT')
            OR NOT has_table_privilege('role_omnibase_infra', 'public.branch_deploy_state', 'UPDATE') THEN
            RAISE EXCEPTION 'role_omnibase_infra lacks SELECT, INSERT, UPDATE on public.branch_deploy_state';
        END IF;
    ELSE
        RAISE NOTICE 'role_omnibase_infra is absent on this lane so nothing is granted';
    END IF;
END
$grant$;
