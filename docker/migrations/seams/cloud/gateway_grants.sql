-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- The gateway role's grants on a cloud database (OMN-26712). Role and database come from the
-- psql variables gateway_role and cloud_database (apply_cloud_database.sql sets them from
-- settings); the grant list is the one the lab gateway grants seam holds, nine items:
--   CONNECT on the database; USAGE on schema public
--   SELECT on tenants, tenant_plans, plan_tiers, plan_entitlements
--   SELECT and UPDATE (last_used_at) on tenant_api_keys
--   SELECT, INSERT, UPDATE on gateway_workflows; INSERT on gateway_rate_limit_events
-- No CREATE, DELETE, TRUNCATE, TRIGGER, REFERENCES, grant option or sequence. Unlike the lab
-- seam this one never skips: an absent role, or a role holding SUPERUSER or BYPASSRLS, FAILS.

SELECT
    NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :'gateway_role') AS role_absent,
    EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :'gateway_role' AND (rolsuper OR rolbypassrls)) AS role_escalated
\gset
\if :role_absent
  DO $gw$ BEGIN RAISE EXCEPTION 'the gateway role named by ONEX_CLOUD_GATEWAY_ROLE does not exist; the cloud owner creates it'; END $gw$;
\endif
\if :role_escalated
  DO $gw$ BEGIN RAISE EXCEPTION 'the gateway role holds SUPERUSER or BYPASSRLS; no grant list constrains it'; END $gw$;
\endif

\connect :"cloud_database"

BEGIN;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM :"gateway_role";
GRANT USAGE ON SCHEMA public TO :"gateway_role";
GRANT SELECT ON public.tenants, public.tenant_plans, public.plan_tiers, public.plan_entitlements TO :"gateway_role";
GRANT SELECT ON public.tenant_api_keys TO :"gateway_role";
GRANT UPDATE (last_used_at) ON public.tenant_api_keys TO :"gateway_role";
GRANT SELECT, INSERT, UPDATE ON public.gateway_workflows TO :"gateway_role";
GRANT INSERT ON public.gateway_rate_limit_events TO :"gateway_role";
COMMIT;
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'gateway_role')
\gexec

-- Readback: the effective set must equal the declared one exactly (a GRANT by a role that does
-- not own the relations warns and returns success).
SELECT
    coalesce((SELECT string_agg(c.relname || ':' || p.priv, ',' ORDER BY c.relname, p.priv)
                FROM pg_catalog.pg_class c
               CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER']) AS p(priv)
               WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                 AND has_table_privilege(:'gateway_role', c.oid, p.priv)), '') AS table_set,
    coalesce((SELECT string_agg(a.attname, ',')
                FROM pg_catalog.pg_attribute a
               WHERE a.attrelid = 'public.tenant_api_keys'::regclass AND a.attnum > 0 AND NOT a.attisdropped
                 AND has_column_privilege(:'gateway_role', a.attrelid, a.attnum, 'UPDATE')), '') AS update_columns,
    has_database_privilege(:'gateway_role', current_database(), 'CONNECT')
      AND has_schema_privilege(:'gateway_role', 'public', 'USAGE')
      AND NOT has_schema_privilege(:'gateway_role', 'public', 'CREATE') AS connect_ok
\gset
SELECT
    :'table_set' = 'gateway_rate_limit_events:INSERT,gateway_workflows:INSERT,gateway_workflows:SELECT,gateway_workflows:UPDATE,plan_entitlements:SELECT,plan_tiers:SELECT,tenant_api_keys:SELECT,tenant_plans:SELECT,tenants:SELECT'
    AND :'update_columns' = 'last_used_at'
    AND :'connect_ok' = 't' AS gateway_exact
\gset
\if :gateway_exact
  \echo [cloud-migration] ok    the gateway role holds exactly the nine gateway grants
\else
  \echo [cloud-migration] FAIL the gateway role does not hold exactly the declared grants: :table_set / :update_columns
  DO $gw$ BEGIN RAISE EXCEPTION 'the gateway role does not hold exactly the nine declared grants after they ran'; END $gw$;
\endif
