-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- The tenant identity writer role's grants on a cloud database (OMN-26712). Role and database
-- come from the psql variables writer_role and cloud_database; the list is the one
-- the lab writer grants seam holds:
--   CONNECT on the database; USAGE on schema public, not CREATE
--   tenants: SELECT, INSERT; UPDATE (provisioning_state, kc_patch_attempt_count,
--            kc_patch_last_error, principal_id, name)
--   plan_tiers: SELECT; tenant_plans: SELECT, INSERT
--   tenant_api_keys: SELECT, INSERT; UPDATE (is_active, revoked_at)
--   tenant_event_outbox: SELECT, INSERT; UPDATE (published_at, attempts, last_error)
--   USAGE on the sequence behind tenant_event_outbox.id
-- No DELETE, TRUNCATE, REFERENCES, TRIGGER, grant option or CREATE. No credential is minted
-- here: the cloud owner creates the role and stores its secret. An absent role FAILS the run.

SELECT
    NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :'writer_role') AS role_absent,
    EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :'writer_role' AND (rolsuper OR rolbypassrls)) AS role_escalated
\gset
\if :role_absent
  DO $tw$ BEGIN RAISE EXCEPTION 'the writer role named by ONEX_CLOUD_TENANT_IDENTITY_WRITER_ROLE does not exist; the cloud owner creates it'; END $tw$;
\endif
\if :role_escalated
  DO $tw$ BEGIN RAISE EXCEPTION 'the writer role holds SUPERUSER or BYPASSRLS; no grant list constrains it'; END $tw$;
\endif

\connect :"cloud_database"

BEGIN;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM :"writer_role";
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM :"writer_role";
GRANT USAGE ON SCHEMA public TO :"writer_role";
GRANT SELECT, INSERT ON public.tenants TO :"writer_role";
GRANT UPDATE (provisioning_state, kc_patch_attempt_count, kc_patch_last_error, principal_id, name) ON public.tenants TO :"writer_role";
GRANT SELECT ON public.plan_tiers TO :"writer_role";
GRANT SELECT, INSERT ON public.tenant_plans TO :"writer_role";
GRANT SELECT, INSERT ON public.tenant_api_keys TO :"writer_role";
GRANT UPDATE (is_active, revoked_at) ON public.tenant_api_keys TO :"writer_role";
GRANT SELECT, INSERT ON public.tenant_event_outbox TO :"writer_role";
GRANT UPDATE (published_at, attempts, last_error) ON public.tenant_event_outbox TO :"writer_role";
SELECT format('GRANT USAGE ON SEQUENCE %s TO %I', pg_catalog.pg_get_serial_sequence('public.tenant_event_outbox', 'id'), :'writer_role')
\gexec
COMMIT;
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'writer_role')
\gexec

-- Readback, every set exact: table-level privileges, the same at any column, the UPDATE columns,
-- the sequences, and any WITH GRANT OPTION form (delegation raises).
SELECT
    coalesce((SELECT string_agg(c.relname || ':' || p.priv, ',' ORDER BY c.relname, p.priv)
                FROM pg_catalog.pg_class c
               CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER']) AS p(priv)
               WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                 AND has_table_privilege(:'writer_role', c.oid, p.priv)), '') AS table_set,
    coalesce((SELECT string_agg(c.relname || ':' || p.priv, ',' ORDER BY c.relname, p.priv)
                FROM pg_catalog.pg_class c
               CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'REFERENCES']) AS p(priv)
               WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                 AND has_any_column_privilege(:'writer_role', c.oid, p.priv)), '') AS any_column_set,
    coalesce((SELECT string_agg(c.relname || '.' || a.attname, ',' ORDER BY c.relname, a.attname)
                FROM pg_catalog.pg_attribute a
                JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
               WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                 AND a.attnum > 0 AND NOT a.attisdropped
                 AND has_column_privilege(:'writer_role', a.attrelid, a.attnum, 'UPDATE')
                 AND NOT has_table_privilege(:'writer_role', c.oid, 'UPDATE')), '') AS update_columns,
    coalesce((SELECT string_agg(c.relname || ':' || p.priv, ',' ORDER BY c.relname, p.priv)
                FROM pg_catalog.pg_class c
               CROSS JOIN unnest(ARRAY['USAGE', 'SELECT', 'UPDATE']) AS p(priv)
               WHERE c.relnamespace = 'public'::regnamespace AND c.relkind = 'S'
                 AND has_sequence_privilege(:'writer_role', c.oid, p.priv)), '') AS sequence_set,
    (SELECT count(*) FROM pg_catalog.pg_class c
      CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER']) AS p(priv)
     WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
       AND (has_table_privilege(:'writer_role', c.oid, p.priv || ' WITH GRANT OPTION')
            OR CASE WHEN p.priv IN ('SELECT', 'INSERT', 'UPDATE', 'REFERENCES')
                    THEN has_any_column_privilege(:'writer_role', c.oid, p.priv || ' WITH GRANT OPTION')
                    ELSE false END)) AS delegated_tables,
    has_database_privilege(:'writer_role', current_database(), 'CONNECT')
      AND NOT has_database_privilege(:'writer_role', current_database(), 'CREATE')
      AND has_schema_privilege(:'writer_role', 'public', 'USAGE')
      AND NOT has_schema_privilege(:'writer_role', 'public', 'CREATE')
      AND NOT has_schema_privilege(:'writer_role', 'public', 'USAGE WITH GRANT OPTION')
      AND NOT has_database_privilege(:'writer_role', current_database(), 'CONNECT WITH GRANT OPTION') AS connect_ok
\gset
SELECT
    :'table_set' = 'plan_tiers:SELECT,tenant_api_keys:INSERT,tenant_api_keys:SELECT,tenant_event_outbox:INSERT,tenant_event_outbox:SELECT,tenant_plans:INSERT,tenant_plans:SELECT,tenants:INSERT,tenants:SELECT'
    AND :'any_column_set' = 'plan_tiers:SELECT,tenant_api_keys:INSERT,tenant_api_keys:SELECT,tenant_api_keys:UPDATE,tenant_event_outbox:INSERT,tenant_event_outbox:SELECT,tenant_event_outbox:UPDATE,tenant_plans:INSERT,tenant_plans:SELECT,tenants:INSERT,tenants:SELECT,tenants:UPDATE'
    AND :'update_columns' = 'tenant_api_keys.is_active,tenant_api_keys.revoked_at,tenant_event_outbox.attempts,tenant_event_outbox.last_error,tenant_event_outbox.published_at,tenants.kc_patch_attempt_count,tenants.kc_patch_last_error,tenants.name,tenants.principal_id,tenants.provisioning_state'
    AND :'sequence_set' = 'tenant_event_outbox_id_seq:USAGE'
    AND :'delegated_tables' = '0'
    AND :'connect_ok' = 't' AS writer_exact
\gset
\if :writer_exact
  \echo [cloud-migration] ok    the tenant identity writer role holds exactly the declared grants
\else
  \echo [cloud-migration] FAIL the tenant identity writer role does not hold exactly the declared grants: :table_set
  DO $tw$ BEGIN RAISE EXCEPTION 'the tenant identity writer role does not hold exactly the declared grants after they ran'; END $tw$;
\endif
