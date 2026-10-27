-- OMN-21424: generic lease projection, one row per (kind, resource).
-- The writer upserts on (kind, resource) and never lowers fence: fence is the
-- etcd revision of the grant and rises with every grant of a resource.
-- The topology declares the relation in the application database, so it is
-- created by this node lineage, not by the flat service-database sequence.

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

-- ---- BEGIN OMN-15376 shape reconciliation: lease_projection ----
-- OMN-17292: CREATE IF NOT EXISTS alone accepts an incomplete legacy table.
-- Keep existing rows; missing required values fail closed rather than inventing
-- a lease identity or fence.
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS resource TEXT NOT NULL;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS lease_id UUID NOT NULL;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS fence BIGINT NOT NULL;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS holder TEXT NOT NULL;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS state TEXT NOT NULL;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

DO $reconcile_lease_constraints$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'public.lease_projection'::regclass AND contype = 'p'
    ) THEN
        ALTER TABLE public.lease_projection
            ADD CONSTRAINT lease_projection_pkey PRIMARY KEY (kind, resource);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'public.lease_projection'::regclass
          AND conname = 'lease_projection_fence_check'
    ) THEN
        ALTER TABLE public.lease_projection
            ADD CONSTRAINT lease_projection_fence_check CHECK (fence >= 1);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'public.lease_projection'::regclass
          AND conname = 'lease_projection_state_check'
    ) THEN
        ALTER TABLE public.lease_projection
            ADD CONSTRAINT lease_projection_state_check
                CHECK (state IN ('held', 'released', 'expired'));
    END IF;
END
$reconcile_lease_constraints$;
-- ---- END OMN-15376 shape reconciliation: lease_projection ----

-- OMN-21279: the topology declares a TABLE grant on this relation for the
-- tenant projection writer; the grant is issued here, next to the CREATE.
-- Flat migration 103 provisions the cluster role before node migrations run.
DO $require_tenant_projection_writer$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'tenant_projection_writer'
    ) THEN
        RAISE EXCEPTION
            'tenant_projection_writer role missing; apply flat migration 103 first';
    END IF;
END
$require_tenant_projection_writer$;

GRANT SELECT, INSERT, UPDATE
    ON public.lease_projection
    TO tenant_projection_writer;

SELECT 1 / count(*) AS lease_projection_writer_privilege_assertion
WHERE has_table_privilege(
    'tenant_projection_writer',
    'public.lease_projection',
    'SELECT, INSERT, UPDATE'
);
