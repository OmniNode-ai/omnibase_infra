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
-- the guarded statements below converge it onto the declared shape and are
-- no-ops on the fresh-create path. Rows are kept; a NULL that blocks NOT NULL
-- fails loud.
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS kind TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS resource TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS lease_id UUID;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS fence BIGINT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS holder TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS state TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();

DO $$
DECLARE
    v_col  TEXT;
    v_nulls BIGINT;
BEGIN
    FOREACH v_col IN ARRAY ARRAY['kind', 'resource', 'lease_id', 'fence', 'holder', 'state', 'updated_at']
    LOOP
        EXECUTE format(
            'SELECT count(*) FROM %s WHERE %I IS NULL', 'public.lease_projection'::regclass, v_col
        ) INTO v_nulls;
        IF v_nulls = 0 THEN
            EXECUTE format(
                'ALTER TABLE %s ALTER COLUMN %I SET NOT NULL', 'public.lease_projection'::regclass, v_col
            );
        ELSE
            RAISE EXCEPTION
                'OMN-15376: cannot converge lease_projection.% to NOT NULL -- % pre-existing row(s) hold NULL. This needs a data ruling; the migration refuses to guess.',
                v_col, v_nulls;
        END IF;
    END LOOP;
END$$;

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
