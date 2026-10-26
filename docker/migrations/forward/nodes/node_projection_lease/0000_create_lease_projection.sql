-- OMN-21427: node projections run in the application database, whereas flat
-- migration 112 runs in omnibase_infra. Materialize the lease projection in
-- the database selected by the tenant_projection binding as well.

CREATE TABLE IF NOT EXISTS public.lease_projection (
    kind        TEXT NOT NULL,
    resource    TEXT NOT NULL,
    lease_id    UUID NOT NULL,
    fence       BIGINT NOT NULL CHECK (fence >= 1),
    holder      TEXT NOT NULL,
    state       TEXT NOT NULL CHECK (state IN ('held', 'released', 'expired')),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (kind, resource)
);

-- Flat migration 103 provisions this cluster-wide identity before node SQL.
SELECT 'tenant_projection_writer'::regrole;

GRANT SELECT, INSERT, UPDATE ON public.lease_projection TO tenant_projection_writer;

DO $require_lease_projection_privileges$
BEGIN
    IF NOT has_table_privilege('tenant_projection_writer', 'public.lease_projection', 'SELECT')
       OR NOT has_table_privilege('tenant_projection_writer', 'public.lease_projection', 'INSERT')
       OR NOT has_table_privilege('tenant_projection_writer', 'public.lease_projection', 'UPDATE') THEN
        RAISE EXCEPTION 'missing required tenant_projection_writer privilege on public.lease_projection';
    END IF;
END
$require_lease_projection_privileges$;
