-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- tenant_api_keys.expires_at on the runtime plane (OMN-26121; T6's minimum, carried by T14).
--
-- Applied by scripts/run-forward-migrations.sh after tenant_identity_schema.sql and
-- before tenant_identity_schema_readback.sql, only where the rendered forward-migration
-- service carries ONEX_MIGRATION_GATEWAY_LANE (dev and the stability lane). This is the only
-- place that adds the column: the create step builds tenant_api_keys without it, so a
-- fresh volume and a warm one end identical.
--
-- GATES (a named skip, never a failure, so a run started with ON_ERROR_STOP=1 survives):
-- database omninode_cloud absent, or table public.tenant_api_keys absent in it.

SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'omninode_cloud') AS db_present
\gset

\if :db_present
  \connect omninode_cloud

  SELECT to_regclass('public.tenant_api_keys') IS NOT NULL AS table_present
  \gset

  \if :table_present
    ALTER TABLE public.tenant_api_keys ADD COLUMN IF NOT EXISTS expires_at timestamptz NULL;
    CREATE INDEX IF NOT EXISTS idx_tenant_api_keys_expires_at
      ON public.tenant_api_keys (expires_at) WHERE is_active;

    -- Read it back: the column must be nullable timestamptz and the index must be the
    -- partial index on (expires_at) WHERE is_active, or the run fails.
    DO $expiry_readback$
    DECLARE
      col_type text;
      col_nullable text;
      idx_def text;
    BEGIN
      SELECT data_type, is_nullable INTO col_type, col_nullable
        FROM information_schema.columns
       WHERE table_schema = 'public' AND table_name = 'tenant_api_keys' AND column_name = 'expires_at';
      IF col_type IS NULL THEN
        RAISE EXCEPTION 'tenant_api_keys.expires_at is missing after the expiry seam ran';
      END IF;
      IF col_nullable <> 'YES' OR col_type <> 'timestamp with time zone' THEN
        RAISE EXCEPTION 'tenant_api_keys.expires_at must be nullable timestamp with time zone (found % nullable=%)', col_type, col_nullable;
      END IF;
      SELECT indexdef INTO idx_def
        FROM pg_catalog.pg_indexes
       WHERE schemaname = 'public' AND tablename = 'tenant_api_keys'
         AND indexname = 'idx_tenant_api_keys_expires_at';
      IF idx_def IS DISTINCT FROM 'CREATE INDEX idx_tenant_api_keys_expires_at ON public.tenant_api_keys USING btree (expires_at) WHERE is_active' THEN
        RAISE EXCEPTION 'idx_tenant_api_keys_expires_at is missing or differs: %', coalesce(idx_def, '<absent>');
      END IF;
    END
    $expiry_readback$;
    \echo [forward-migration]   ok    tenant_api_keys.expires_at present
  \else
    \echo [forward-migration]   skip  tenant_api_keys expiry (table public.tenant_api_keys absent in omninode_cloud on this lane)
  \endif
\else
  \echo [forward-migration]   skip  tenant_api_keys expiry (database omninode_cloud absent on this lane)
\endif
