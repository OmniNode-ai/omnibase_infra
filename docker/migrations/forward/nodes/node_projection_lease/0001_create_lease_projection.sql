-- OMN-21424: generic lease projection, one row per (kind, resource).
-- The writer upserts on (kind, resource) and never lowers fence: fence is the
-- etcd revision of the grant and rises with every grant of a resource.

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
-- CREATE TABLE IF NOT EXISTS no-ops against a pre-existing lease_projection of
-- a different shape. The guarded adds below converge such a table onto the
-- shape declared above; on the fresh path every statement is a no-op. No DROP,
-- no recreate: pre-existing rows are preserved.

ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS kind TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS resource TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS lease_id UUID;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS fence BIGINT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS holder TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS state TEXT;
ALTER TABLE public.lease_projection ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();

-- SET NOT NULL fails loud, naming the column, when a pre-existing row holds
-- NULL; that needs a data ruling, and the migration refuses to guess.
ALTER TABLE public.lease_projection
    ALTER COLUMN kind SET NOT NULL,
    ALTER COLUMN resource SET NOT NULL,
    ALTER COLUMN lease_id SET NOT NULL,
    ALTER COLUMN fence SET NOT NULL,
    ALTER COLUMN holder SET NOT NULL,
    ALTER COLUMN state SET NOT NULL,
    ALTER COLUMN updated_at SET NOT NULL;

DO $$
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
END$$;

-- ---- END OMN-15376 shape reconciliation: lease_projection ----

-- OMN-21279: the topology declares a TABLE grant on this relation for the
-- tenant projection writer; the grant is issued here, next to the CREATE.
-- Flat migration 103 provisions the cluster role before this file runs.
DO $require_tenant_projection_writer$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'tenant_projection_writer'
    ) THEN
        RAISE EXCEPTION
            'tenant_projection_writer role missing; apply flat migration 103 before this migration';
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
