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

-- ---- BEGIN OMN-15376 shape reconciliation: lease_projection ----
-- CREATE TABLE IF NOT EXISTS no-ops on a pre-existing table of another shape;
-- the statements below converge it onto the declared shape and are
-- no-ops on the fresh-create path. Rows are kept; a NULL that blocks NOT NULL
-- fails loud.
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS kind TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS resource TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS lease_id UUID;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS fence BIGINT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS holder TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS state TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();

-- Explicit targets let the application database gate prove schema ownership.
-- PostgreSQL checks existing rows before SET NOT NULL and refuses NULLs;
-- it never fills missing values or discards rows to converge the shape.
ALTER TABLE public.lease_projection ALTER COLUMN kind SET NOT NULL;
ALTER TABLE public.lease_projection ALTER COLUMN resource SET NOT NULL;
ALTER TABLE public.lease_projection ALTER COLUMN lease_id SET NOT NULL;
ALTER TABLE public.lease_projection ALTER COLUMN fence SET NOT NULL;
ALTER TABLE public.lease_projection ALTER COLUMN holder SET NOT NULL;
ALTER TABLE public.lease_projection ALTER COLUMN state SET NOT NULL;
ALTER TABLE public.lease_projection ALTER COLUMN updated_at SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'public.lease_projection'::regclass AND contype = 'p'
    ) THEN
        ALTER TABLE public.lease_projection
            ADD CONSTRAINT lease_projection_pkey PRIMARY KEY (kind, resource);
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'public.lease_projection'::regclass
          AND contype = 'c' AND conname = 'lease_projection_fence_check'
    ) THEN
        ALTER TABLE public.lease_projection
            ADD CONSTRAINT lease_projection_fence_check CHECK (fence >= 1);
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'public.lease_projection'::regclass
          AND contype = 'c' AND conname = 'lease_projection_state_check'
    ) THEN
        ALTER TABLE public.lease_projection
            ADD CONSTRAINT lease_projection_state_check CHECK (state IN ('held', 'released', 'expired'));
    END IF;
END$$;
-- ---- END OMN-15376 shape reconciliation: lease_projection ----

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
