-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- tenant_identity_writer grants on omninode_cloud (OMN-26471).
--
-- Applied by scripts/run-forward-migrations.sh on every compose up, from the
-- login-only role grant seam beside gateway_runtime_grants.sql, as the lane's
-- superuser connected to the runner's own database (psql -X -q -v ON_ERROR_STOP=1
-- -f this file). It lives OUTSIDE docker/migrations/forward/ on purpose: that
-- directory is the ordered, checksummed flat corpus. This is a re-asserted grant
-- script, not a migration.
--
-- The role (tenant_identity_writer) is minted LOGIN-only by
-- 000_create_multiple_databases.sh on a fresh volume and by the runner's credential
-- seam on a warm one; this file issues no credential and no role attribute. The
-- runtime's tenant and key nodes write the cloud identity tables' rows as it.
--
-- THE DECLARED GRANTS, AND NOTHING ELSE, on database omninode_cloud, schema public:
--   CONNECT on the database; USAGE (not CREATE) on the schema
--   tenants:             SELECT, INSERT; UPDATE (provisioning_state,
--                        kc_patch_attempt_count, kc_patch_last_error, principal_id, name)
--   plan_tiers:          SELECT
--   tenant_plans:        SELECT, INSERT
--   tenant_api_keys:     SELECT, INSERT; UPDATE (is_active, revoked_at)
--   tenant_event_outbox: SELECT, INSERT; UPDATE (published_at, attempts, last_error)
--   USAGE on the outbox id sequence
-- No DELETE, no TRUNCATE, no CREATE, no TRIGGER, no REFERENCES, no grant option.
--
-- GATES (a named skip, re-asserted on the next run): the role is not provisioned, the
-- database is absent, one of the five tables does not exist yet (the cloud-migration
-- one-shot owns them), or the outbox id sequence does not exist. A role that holds
-- SUPERUSER or BYPASSRLS, or an effective privilege set (including any delegation
-- right) that differs from the declared list after the grants ran, FAILS the run.

SELECT
    EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'tenant_identity_writer') AS role_present,
    EXISTS (SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'omninode_cloud') AS db_present
\gset

\if :role_present
  \if :db_present
    DO $tenant_identity_escalation$
    BEGIN
      IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles
                  WHERE rolname = 'tenant_identity_writer' AND (rolsuper OR rolbypassrls)) THEN
        RAISE EXCEPTION 'tenant_identity_writer holds SUPERUSER or BYPASSRLS; no grant list constrains it. A role administrator must remove those attributes.';
      END IF;
    END
    $tenant_identity_escalation$;

    GRANT CONNECT ON DATABASE omninode_cloud TO "tenant_identity_writer";

    \connect omninode_cloud

    SELECT count(*) = 5 AS tables_present,
           to_regclass('public.tenant_event_outbox_id_seq') IS NOT NULL AS sequence_present
    FROM unnest(ARRAY['tenants', 'plan_tiers', 'tenant_plans', 'tenant_api_keys',
                      'tenant_event_outbox']) AS t(name)
    WHERE to_regclass('public.' || t.name) IS NOT NULL
    \gset

    \if :tables_present
      \if :sequence_present
        BEGIN;
        REVOKE ALL ON ALL TABLES IN SCHEMA public FROM "tenant_identity_writer";
        REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM "tenant_identity_writer";
        GRANT USAGE ON SCHEMA public TO "tenant_identity_writer";
        GRANT SELECT, INSERT ON public.tenants TO "tenant_identity_writer";
        GRANT UPDATE (provisioning_state, kc_patch_attempt_count, kc_patch_last_error, principal_id, name) ON public.tenants TO "tenant_identity_writer";
        GRANT SELECT ON public.plan_tiers TO "tenant_identity_writer";
        GRANT SELECT, INSERT ON public.tenant_plans TO "tenant_identity_writer";
        GRANT SELECT, INSERT ON public.tenant_api_keys TO "tenant_identity_writer";
        GRANT UPDATE (is_active, revoked_at) ON public.tenant_api_keys TO "tenant_identity_writer";
        GRANT SELECT, INSERT ON public.tenant_event_outbox TO "tenant_identity_writer";
        GRANT UPDATE (published_at, attempts, last_error) ON public.tenant_event_outbox TO "tenant_identity_writer";
        GRANT USAGE ON SEQUENCE public.tenant_event_outbox_id_seq TO "tenant_identity_writer";
        COMMIT;

        -- Read the effective privileges back and compare them with the declared list.
        -- A GRANT without grant option on the object warns and returns success, so the
        -- readback is the only thing separating a real grant from a silent no-op. Every
        -- set is exact: an extra privilege, including one inherited from PUBLIC or from
        -- a role the writer belongs to, makes it differ. Table-level sets hold the
        -- table-wide privileges, column_set holds the column-only ones (a column the
        -- role reaches only through a column grant), and the *_option set holds any
        -- privilege the role could pass on (WITH GRANT OPTION forms of every probe).
        DO $tenant_identity_readback$
        DECLARE
          table_set text;
          column_set text;
          sequence_set text;
          option_set text;
        BEGIN
          SELECT coalesce(string_agg(c.relname || ':' || p.priv, ',' ORDER BY c.relname COLLATE "C", p.priv COLLATE "C"), '')
            INTO table_set
            FROM pg_catalog.pg_class c
           CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER']) AS p(priv)
           WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
             AND has_table_privilege('tenant_identity_writer', c.oid, p.priv);
          SELECT coalesce(string_agg(c.relname || '.' || a.attname || ':' || p.priv, ',' ORDER BY c.relname COLLATE "C", a.attname COLLATE "C", p.priv COLLATE "C"), '')
            INTO column_set
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
           CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'REFERENCES']) AS p(priv)
           WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
             AND has_column_privilege('tenant_identity_writer', c.oid, a.attnum, p.priv)
             AND NOT has_table_privilege('tenant_identity_writer', c.oid, p.priv);
          SELECT coalesce(string_agg(c.relname || ':' || p.priv, ',' ORDER BY c.relname COLLATE "C", p.priv COLLATE "C"), '')
            INTO sequence_set
            FROM pg_catalog.pg_class c
           CROSS JOIN unnest(ARRAY['USAGE', 'SELECT', 'UPDATE']) AS p(priv)
           WHERE c.relnamespace = 'public'::regnamespace AND c.relkind = 'S'
             AND has_sequence_privilege('tenant_identity_writer', c.oid, p.priv);
          SELECT coalesce(string_agg(found, ','), '')
            INTO option_set
            FROM (
              SELECT 'table:' || c.relname || ':' || p.priv AS found
                FROM pg_catalog.pg_class c
               CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER']) AS p(priv)
               WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                 AND has_table_privilege('tenant_identity_writer', c.oid, p.priv || ' WITH GRANT OPTION')
              UNION ALL
              SELECT 'column:' || c.relname || '.' || a.attname || ':' || p.priv
                FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
               CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'REFERENCES']) AS p(priv)
               WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                 AND has_column_privilege('tenant_identity_writer', c.oid, a.attnum, p.priv || ' WITH GRANT OPTION')
              UNION ALL
              SELECT 'sequence:' || c.relname || ':' || p.priv
                FROM pg_catalog.pg_class c
               CROSS JOIN unnest(ARRAY['USAGE', 'SELECT', 'UPDATE']) AS p(priv)
               WHERE c.relnamespace = 'public'::regnamespace AND c.relkind = 'S'
                 AND has_sequence_privilege('tenant_identity_writer', c.oid, p.priv || ' WITH GRANT OPTION')
              UNION ALL
              SELECT 'database:' || p.priv
                FROM unnest(ARRAY['CONNECT', 'CREATE', 'TEMPORARY']) AS p(priv)
               WHERE has_database_privilege('tenant_identity_writer', 'omninode_cloud', p.priv || ' WITH GRANT OPTION')
              UNION ALL
              SELECT 'schema:' || p.priv
                FROM unnest(ARRAY['USAGE', 'CREATE']) AS p(priv)
               WHERE has_schema_privilege('tenant_identity_writer', 'public', p.priv || ' WITH GRANT OPTION')
            ) delegable;
          IF NOT has_database_privilege('tenant_identity_writer', 'omninode_cloud', 'CONNECT')
             OR has_database_privilege('tenant_identity_writer', 'omninode_cloud', 'CREATE')
             OR NOT has_schema_privilege('tenant_identity_writer', 'public', 'USAGE')
             OR has_schema_privilege('tenant_identity_writer', 'public', 'CREATE')
             OR table_set IS DISTINCT FROM 'plan_tiers:SELECT,tenant_api_keys:INSERT,tenant_api_keys:SELECT,tenant_event_outbox:INSERT,tenant_event_outbox:SELECT,tenant_plans:INSERT,tenant_plans:SELECT,tenants:INSERT,tenants:SELECT'
             OR column_set IS DISTINCT FROM 'tenant_api_keys.is_active:UPDATE,tenant_api_keys.revoked_at:UPDATE,tenant_event_outbox.attempts:UPDATE,tenant_event_outbox.last_error:UPDATE,tenant_event_outbox.published_at:UPDATE,tenants.kc_patch_attempt_count:UPDATE,tenants.kc_patch_last_error:UPDATE,tenants.name:UPDATE,tenants.principal_id:UPDATE,tenants.provisioning_state:UPDATE'
             OR sequence_set IS DISTINCT FROM 'tenant_event_outbox_id_seq:USAGE'
             OR option_set <> '' THEN
            RAISE EXCEPTION 'tenant_identity_writer on omninode_cloud does not hold exactly the declared grants after they ran (tables=%, columns=%, sequences=%, delegable=%). A GRANT without grant option on the object warns and returns success; re-run as a role that owns the relations.', table_set, column_set, sequence_set, option_set;
          END IF;
        END
        $tenant_identity_readback$;
        \echo [forward-migration]   ok    tenant_identity_writer holds exactly the declared grants on omninode_cloud
      \else
        \echo [forward-migration]   skip  tenant_identity_writer grants (the outbox id sequence tenant_event_outbox_id_seq does not exist in omninode_cloud yet; re-asserted on the next run)
      \endif
    \else
      \echo [forward-migration]   skip  tenant_identity_writer grants (the five identity tables do not all exist in omninode_cloud yet -- the cloud-migration one-shot owns them; re-asserted on the next run)
    \endif
  \else
    \echo [forward-migration]   skip  tenant_identity_writer grants (database omninode_cloud absent on this lane)
  \endif
\else
  \echo [forward-migration]   skip  tenant_identity_writer grants (role not provisioned on this lane)
\endif
