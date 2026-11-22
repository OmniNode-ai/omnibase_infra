-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- tenant_api_keys.expires_at on the lanes that serve the gateway (OMN-26472).
--
-- Applied by scripts/run-forward-migrations.sh from the case on
-- ONEX_MIGRATION_GATEWAY_LANE, as the lane's superuser connected to the runner's
-- own database (psql -X -q -v ON_ERROR_STOP=1 -f this file). It lives OUTSIDE
-- docker/migrations/forward/ on purpose: that directory is the ordered,
-- checksummed flat corpus, and its slot-confinement and application-database
-- gates read every .sql file under it. This is a re-asserted schema seam, not a
-- migration.
--
-- WHAT IT DOES, and nothing else, on database omninode_cloud, schema public:
--   1  ADD COLUMN IF NOT EXISTS expires_at timestamptz NULL on tenant_api_keys
--   2  a partial index on (expires_at) WHERE is_active
--   3  a readback that raises unless the column exists, is nullable and is
--      timestamp with time zone, and the index has that one key and that predicate
-- No default and no backfill: the existing key rows are unchanged, and a NULL
-- expiry means the key does not expire. Expiry is set at insert only, so no
-- grant changes here (the writer's INSERT already covers the column).
--
-- GATES (a named skip, re-asserted on the next run): the database omninode_cloud
-- is absent on this lane, or the table public.tenant_api_keys does not exist in
-- it yet. An absent table is a skip and never fails a run started with
-- ON_ERROR_STOP=1. A column, type or index that differs after the DDL ran FAILS
-- the run.
--
-- WHO RUNS IT: only a lane whose forward-migration service carries
-- ONEX_MIGRATION_GATEWAY_LANE. The runner is shared by every lane, and the
-- collaborator lane's cloud one-shot runs after the runner finishes, so no seam
-- here may run ahead of that lane's corpus.

SELECT
    EXISTS (SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'omninode_cloud') AS db_present
\gset

\if :db_present
  \connect omninode_cloud

  SELECT to_regclass('public.tenant_api_keys') IS NOT NULL AS table_present
  \gset

  \if :table_present
    ALTER TABLE public.tenant_api_keys ADD COLUMN IF NOT EXISTS expires_at timestamptz NULL;

    CREATE INDEX IF NOT EXISTS ix_tenant_api_keys_expires_at_active
      ON public.tenant_api_keys (expires_at) WHERE is_active;

    -- Read the column and the index back. ADD COLUMN IF NOT EXISTS and CREATE INDEX
    -- IF NOT EXISTS are silent no-ops over a column or index that already exists with
    -- a different shape, so the readback is the only thing separating a converged
    -- table from a drifted one.
    DO $expiry_readback$
    DECLARE
      col_nullable text;
      col_type text;
      idx_keys text;
      idx_predicate text;
    BEGIN
      SELECT c.is_nullable, c.data_type
        INTO col_nullable, col_type
        FROM information_schema.columns c
       WHERE c.table_schema = 'public' AND c.table_name = 'tenant_api_keys'
         AND c.column_name = 'expires_at';
      IF col_type IS NULL THEN
        RAISE EXCEPTION 'tenant_api_keys.expires_at is missing after the ALTER ran';
      END IF;
      IF col_nullable IS DISTINCT FROM 'YES' THEN
        RAISE EXCEPTION 'tenant_api_keys.expires_at is not nullable (is_nullable=%); existing key rows carry no expiry', col_nullable;
      END IF;
      IF col_type IS DISTINCT FROM 'timestamp with time zone' THEN
        RAISE EXCEPTION 'tenant_api_keys.expires_at has type % and must be timestamp with time zone', col_type;
      END IF;

      SELECT string_agg(a.attname, ',' ORDER BY k.ord), pg_catalog.pg_get_expr(i.indpred, i.indrelid)
        INTO idx_keys, idx_predicate
        FROM pg_catalog.pg_class ic
        JOIN pg_catalog.pg_index i ON i.indexrelid = ic.oid
        CROSS JOIN LATERAL unnest(i.indkey::int2[]) WITH ORDINALITY AS k(attnum, ord)
        JOIN pg_catalog.pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
       WHERE ic.relname = 'ix_tenant_api_keys_expires_at_active'
         AND ic.relnamespace = 'public'::regnamespace
         AND i.indrelid = 'public.tenant_api_keys'::regclass
       GROUP BY i.indpred, i.indrelid;
      IF idx_keys IS DISTINCT FROM 'expires_at' OR idx_predicate IS DISTINCT FROM 'is_active' THEN
        RAISE EXCEPTION 'index ix_tenant_api_keys_expires_at_active on tenant_api_keys must key (expires_at) WHERE is_active (found keys=%, predicate=%)', idx_keys, idx_predicate;
      END IF;
    END
    $expiry_readback$;
    \echo [forward-migration]   ok tenant_api_keys.expires_at present
  \else
    \echo [forward-migration]   skip  tenant_api_keys expiry (table public.tenant_api_keys absent in omninode_cloud on this lane)
  \endif
\else
  \echo [forward-migration]   skip  tenant_api_keys expiry (database omninode_cloud absent on this lane)
\endif
