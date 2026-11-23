-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- gateway_runtime grants on omninode_cloud (OMN-26151).
--
-- Applied by scripts/run-forward-migrations.sh on every compose up, from the
-- login-only role grant seam, as the lane's superuser connected to the runner's
-- own database (psql -X -q -v ON_ERROR_STOP=1 -f this file). It lives OUTSIDE
-- docker/migrations/forward/ on purpose: that directory is the ordered, checksummed
-- flat corpus, and its slot-confinement and application-database gates read every
-- .sql file under it. This is a re-asserted grant script, not a migration.
--
-- The role (gateway_runtime) is minted LOGIN-only by 000_create_multiple_databases.sh
-- on a fresh volume and by the runner's credential seam on a warm one; this file
-- issues no credential and no role attribute. The lab workflow gateway (omnimarket
-- node_workflow_gateway_effect, secret gateway.database.dsn) connects as it.
--
-- THE NINE ITEMS, AND NOTHING ELSE, on database omninode_cloud, schema public:
--   1  CONNECT on the database
--   2  USAGE on the schema
--   3  SELECT on tenants, tenant_plans, plan_tiers, plan_entitlements (items 3 to 6)
--   7  SELECT and UPDATE (last_used_at) on tenant_api_keys
--   8  SELECT, INSERT, UPDATE on gateway_workflows
--   9  INSERT on gateway_rate_limit_events
-- No CREATE, no TRUNCATE, no DELETE, no TRIGGER, no REFERENCES, no grant option, and
-- no sequence: none of the seven tables draws from one.
--
-- GATES (a named skip, re-asserted on the next run): the role is not provisioned,
-- the database is absent, or one of the seven tables does not exist yet (the
-- cloud-migration one-shot owns them; on a stability lane it runs first). A role
-- that holds SUPERUSER or BYPASSRLS, or an effective privilege set that differs
-- from the nine items after the grants ran, FAILS the run.

-- OMN-26121: the tenant identity schema (create where absent, expires_at, readback) runs first, on
-- the lanes that carry ONEX_MIGRATION_GATEWAY_LANE; the entry file names a skip everywhere else.
\ir tenant_identity_gateway_lane.sql

SELECT
    EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'gateway_runtime') AS role_present,
    EXISTS (SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'omninode_cloud') AS db_present
\gset

\if :role_present
  \if :db_present
    DO $gateway_escalation$
    BEGIN
      IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles
                  WHERE rolname = 'gateway_runtime' AND (rolsuper OR rolbypassrls)) THEN
        RAISE EXCEPTION 'gateway_runtime holds SUPERUSER or BYPASSRLS; no grant list constrains it. A role administrator must remove those attributes.';
      END IF;
    END
    $gateway_escalation$;

    GRANT CONNECT ON DATABASE omninode_cloud TO "gateway_runtime";

    \connect omninode_cloud

    SELECT count(*) = 7 AS tables_present
    FROM unnest(ARRAY['tenants', 'tenant_plans', 'plan_tiers', 'plan_entitlements',
                      'tenant_api_keys', 'gateway_workflows', 'gateway_rate_limit_events']) AS t(name)
    WHERE to_regclass('public.' || t.name) IS NOT NULL
    \gset

    \if :tables_present
      BEGIN;
      REVOKE ALL ON ALL TABLES IN SCHEMA public FROM "gateway_runtime";
      GRANT USAGE ON SCHEMA public TO "gateway_runtime";
      GRANT SELECT ON public.tenants, public.tenant_plans, public.plan_tiers, public.plan_entitlements TO "gateway_runtime";
      GRANT SELECT ON public.tenant_api_keys TO "gateway_runtime";
      GRANT UPDATE (last_used_at) ON public.tenant_api_keys TO "gateway_runtime";
      GRANT SELECT, INSERT, UPDATE ON public.gateway_workflows TO "gateway_runtime";
      GRANT INSERT ON public.gateway_rate_limit_events TO "gateway_runtime";
      COMMIT;

      -- Read the effective privileges back and compare them with the nine items.
      -- A GRANT without grant option on the object warns and returns success, so the
      -- readback is the only thing separating a real grant from a silent no-op. The
      -- table-level set is exact (an extra privilege, including one inherited from
      -- PUBLIC, makes it differ), the column-level UPDATE set must be last_used_at
      -- alone, and the schema must offer USAGE and not CREATE.
      DO $gateway_readback$
      DECLARE
        table_set text;
        update_columns text;
      BEGIN
        SELECT coalesce(string_agg(c.relname || ':' || p.priv, ',' ORDER BY c.relname, p.priv), '')
          INTO table_set
          FROM pg_catalog.pg_class c
         CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER']) AS p(priv)
         WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
           AND has_table_privilege('gateway_runtime', c.oid, p.priv);
        SELECT coalesce(string_agg(a.attname, ','), '')
          INTO update_columns
          FROM pg_catalog.pg_attribute a
         WHERE a.attrelid = 'public.tenant_api_keys'::regclass AND a.attnum > 0 AND NOT a.attisdropped
           AND has_column_privilege('gateway_runtime', a.attrelid, a.attnum, 'UPDATE');
        IF NOT has_database_privilege('gateway_runtime', 'omninode_cloud', 'CONNECT')
           OR NOT has_schema_privilege('gateway_runtime', 'public', 'USAGE')
           OR has_schema_privilege('gateway_runtime', 'public', 'CREATE')
           OR table_set IS DISTINCT FROM 'gateway_rate_limit_events:INSERT,gateway_workflows:INSERT,gateway_workflows:SELECT,gateway_workflows:UPDATE,plan_entitlements:SELECT,plan_tiers:SELECT,tenant_api_keys:SELECT,tenant_plans:SELECT,tenants:SELECT'
           OR update_columns IS DISTINCT FROM 'last_used_at' THEN
          RAISE EXCEPTION 'gateway_runtime on omninode_cloud does not hold exactly the nine declared grants after they ran (tables=%, update_columns=%). A GRANT without grant option on the object warns and returns success; re-run as a role that owns the relations.', table_set, update_columns;
        END IF;
      END
      $gateway_readback$;
      \echo [forward-migration]   ok    gateway_runtime holds exactly the nine gateway grants on omninode_cloud
    \else
      \echo [forward-migration]   skip  gateway_runtime grants (the seven gateway tables do not all exist in omninode_cloud yet -- the cloud-migration one-shot owns them; re-asserted on the next run)
    \endif
  \else
    \echo [forward-migration]   skip  gateway_runtime grants (database omninode_cloud absent on this lane)
  \endif
\else
  \echo [forward-migration]   skip  gateway_runtime grants (role not provisioned on this lane)
\endif
