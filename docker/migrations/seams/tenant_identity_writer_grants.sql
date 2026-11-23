-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- tenant_identity_writer: the login and the least-grant set on omninode_cloud (OMN-26471).
--
-- Applied by scripts/run-forward-migrations.sh on every compose up, in the same psql run as
-- gateway_runtime_grants.sql (psql -X -q -v ON_ERROR_STOP=1 -f ... -f this file), as the lane's
-- superuser. It lives OUTSIDE docker/migrations/forward/ on purpose, for the reason
-- gateway_runtime_grants.sql gives: that directory is the ordered, checksummed flat corpus.
--
-- ONE FILE, TWO JOBS, NO BASELINED LIST. The role is NOT a LOGIN_ONLY_ROLE_MAP member. The
-- credential is minted and re-asserted here, from the environment variable
-- TENANT_IDENTITY_WRITER_PASSWORD (psql \getenv: the value reaches the server as a quoted
-- literal built by format(%L), is never echoed and is never an argument), so the four
-- baselined shell and python files that hold the other login roles' lists do not grow. The
-- same attributes as the runner's credential seam: LOGIN, NOSUPERUSER, NOBYPASSRLS, NOCREATEDB,
-- NOCREATEROLE, NOREPLICATION; a pre-existing role has LOGIN and PASSWORD set and nothing else.
--
-- THE GRANTS, EXACTLY, on database omninode_cloud, schema public:
--   CONNECT on the database; USAGE on the schema and not CREATE
--   tenants:             SELECT, INSERT; UPDATE (provisioning_state, kc_patch_attempt_count,
--                        kc_patch_last_error, principal_id, name)
--   plan_tiers:          SELECT
--   tenant_plans:        SELECT, INSERT
--   tenant_api_keys:     SELECT, INSERT; UPDATE (is_active, revoked_at)
--   tenant_event_outbox: SELECT, INSERT; UPDATE (published_at, attempts, last_error)
--   USAGE on the sequence behind tenant_event_outbox.id
-- No DELETE, no TRUNCATE, no REFERENCES, no TRIGGER, no grant option, no CREATE.
-- Expiry on tenant_api_keys is set at insert only, so INSERT covers it.
--
-- GATES (a named skip, re-asserted on the next run): the password is unset, the database is
-- absent, or a table, a column or the sequence named above does not exist yet (the tenant
-- identity schema seam owns them). A password that is set and not hex FAILS the run: a skip
-- is right for an absent credential and wrong for a malformed one. A role that holds
-- SUPERUSER or BYPASSRLS, or an effective privilege set that differs from the list after the
-- grants ran (including any WITH GRANT OPTION form), FAILS the run.
--
-- The session may arrive connected to omninode_cloud (gateway_runtime_grants.sql ran first and
-- \connect-ed there), so everything before the \connect below reads cluster-wide catalogs only.

\set writer_pw ''
\getenv writer_pw TENANT_IDENTITY_WRITER_PASSWORD

SELECT
    :'writer_pw' <> '' AS pw_set,
    :'writer_pw' ~ '^[0-9a-fA-F]+$' AS pw_hex,
    EXISTS (SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'omninode_cloud') AS db_present
\gset

\if :pw_set
  \if :pw_hex
    \if :db_present
      SELECT format('CREATE ROLE tenant_identity_writer WITH LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD %L', :'writer_pw')
       WHERE NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'tenant_identity_writer')
      \gexec
      SELECT format('ALTER ROLE tenant_identity_writer WITH LOGIN PASSWORD %L', :'writer_pw')
       WHERE EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'tenant_identity_writer')
      \gexec
      \set writer_pw ''
      \echo [forward-migration]   ok    tenant_identity_writer LOGIN credential asserted

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

      SELECT
        (SELECT count(*) = 6
           FROM unnest(ARRAY['tenants', 'plan_tiers', 'tenant_plans', 'tenant_api_keys', 'tenant_event_outbox', 'tenant_event_outbox_id_seq']) AS t(name)
          WHERE to_regclass('public.' || t.name) IS NOT NULL)
        AND (SELECT count(*) = 10
               FROM pg_catalog.pg_attribute a
               JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
              WHERE c.relnamespace = 'public'::regnamespace AND NOT a.attisdropped AND a.attnum > 0
                AND (c.relname, a.attname) IN (
                      ('tenants', 'provisioning_state'), ('tenants', 'kc_patch_attempt_count'),
                      ('tenants', 'kc_patch_last_error'), ('tenants', 'principal_id'), ('tenants', 'name'),
                      ('tenant_api_keys', 'is_active'), ('tenant_api_keys', 'revoked_at'),
                      ('tenant_event_outbox', 'published_at'), ('tenant_event_outbox', 'attempts'),
                      ('tenant_event_outbox', 'last_error')))
        AS objects_present
      \gset

      \if :objects_present
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
        -- The sequence is resolved from the owned table's id column, not named: the ownership
        -- manifests declare the table, and the readback below pins the resulting name.
        SELECT format('GRANT USAGE ON SEQUENCE %s TO "tenant_identity_writer"', pg_catalog.pg_get_serial_sequence('public.tenant_event_outbox', 'id'))
        \gexec
        COMMIT;

        -- Read the effective privileges back and compare them with the list. A GRANT by a role
        -- that is not the owner warns and returns success, so the readback is the only thing
        -- separating a real grant from a silent no-op. Four sets, each exact: the table-level
        -- privileges over every relation in public (an extra one, including one inherited from
        -- PUBLIC, makes it differ); the same privileges at any column (a column-level SELECT or
        -- INSERT the table-level set does not show); the UPDATE columns; and the sequences. Then
        -- every WITH GRANT OPTION form of the same functions: any delegation right raises.
        DO $tenant_identity_readback$
        DECLARE
          table_set text;
          any_column_set text;
          update_columns text;
          sequence_set text;
          delegated text;
        BEGIN
          SELECT coalesce(string_agg(c.relname || ':' || p.priv, ',' ORDER BY c.relname, p.priv), '')
            INTO table_set
            FROM pg_catalog.pg_class c
           CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER']) AS p(priv)
           WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
             AND has_table_privilege('tenant_identity_writer', c.oid, p.priv);
          SELECT coalesce(string_agg(c.relname || ':' || p.priv, ',' ORDER BY c.relname, p.priv), '')
            INTO any_column_set
            FROM pg_catalog.pg_class c
           CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'REFERENCES']) AS p(priv)
           WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
             AND has_any_column_privilege('tenant_identity_writer', c.oid, p.priv);
          SELECT coalesce(string_agg(c.relname || '.' || a.attname, ',' ORDER BY c.relname, a.attname), '')
            INTO update_columns
            FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
           WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
             AND a.attnum > 0 AND NOT a.attisdropped
             AND has_column_privilege('tenant_identity_writer', a.attrelid, a.attnum, 'UPDATE')
             AND NOT has_table_privilege('tenant_identity_writer', c.oid, 'UPDATE');
          SELECT coalesce(string_agg(c.relname || ':' || p.priv, ',' ORDER BY c.relname, p.priv), '')
            INTO sequence_set
            FROM pg_catalog.pg_class c
           CROSS JOIN unnest(ARRAY['USAGE', 'SELECT', 'UPDATE']) AS p(priv)
           WHERE c.relnamespace = 'public'::regnamespace AND c.relkind = 'S'
             AND has_sequence_privilege('tenant_identity_writer', c.oid, p.priv);
          SELECT coalesce(string_agg(d.what, ','), '')
            INTO delegated
            FROM (
              SELECT 'table:' || c.relname || ':' || p.priv AS what
                FROM pg_catalog.pg_class c
               CROSS JOIN unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER']) AS p(priv)
               WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                 AND (has_table_privilege('tenant_identity_writer', c.oid, p.priv || ' WITH GRANT OPTION')
                      OR CASE WHEN p.priv IN ('SELECT', 'INSERT', 'UPDATE', 'REFERENCES')
                              THEN has_any_column_privilege('tenant_identity_writer', c.oid, p.priv || ' WITH GRANT OPTION')
                              ELSE false END)
              UNION ALL
              SELECT 'sequence:' || c.relname || ':' || p.priv
                FROM pg_catalog.pg_class c
               CROSS JOIN unnest(ARRAY['USAGE', 'SELECT', 'UPDATE']) AS p(priv)
               WHERE c.relnamespace = 'public'::regnamespace AND c.relkind = 'S'
                 AND has_sequence_privilege('tenant_identity_writer', c.oid, p.priv || ' WITH GRANT OPTION')
              UNION ALL
              SELECT 'schema:public:' || p.priv
                FROM unnest(ARRAY['USAGE', 'CREATE']) AS p(priv)
               WHERE has_schema_privilege('tenant_identity_writer', 'public', p.priv || ' WITH GRANT OPTION')
              UNION ALL
              SELECT 'database:omninode_cloud:' || p.priv
                FROM unnest(ARRAY['CONNECT', 'CREATE', 'TEMPORARY']) AS p(priv)
               WHERE has_database_privilege('tenant_identity_writer', 'omninode_cloud', p.priv || ' WITH GRANT OPTION')
            ) AS d;
          IF NOT has_database_privilege('tenant_identity_writer', 'omninode_cloud', 'CONNECT')
             OR has_database_privilege('tenant_identity_writer', 'omninode_cloud', 'CREATE')
             OR NOT has_schema_privilege('tenant_identity_writer', 'public', 'USAGE')
             OR has_schema_privilege('tenant_identity_writer', 'public', 'CREATE')
             OR table_set IS DISTINCT FROM 'plan_tiers:SELECT,tenant_api_keys:INSERT,tenant_api_keys:SELECT,tenant_event_outbox:INSERT,tenant_event_outbox:SELECT,tenant_plans:INSERT,tenant_plans:SELECT,tenants:INSERT,tenants:SELECT'
             OR any_column_set IS DISTINCT FROM 'plan_tiers:SELECT,tenant_api_keys:INSERT,tenant_api_keys:SELECT,tenant_api_keys:UPDATE,tenant_event_outbox:INSERT,tenant_event_outbox:SELECT,tenant_event_outbox:UPDATE,tenant_plans:INSERT,tenant_plans:SELECT,tenants:INSERT,tenants:SELECT,tenants:UPDATE'
             OR update_columns IS DISTINCT FROM 'tenant_api_keys.is_active,tenant_api_keys.revoked_at,tenant_event_outbox.attempts,tenant_event_outbox.last_error,tenant_event_outbox.published_at,tenants.kc_patch_attempt_count,tenants.kc_patch_last_error,tenants.name,tenants.principal_id,tenants.provisioning_state'
             OR sequence_set IS DISTINCT FROM 'tenant_event_outbox_id_seq:USAGE'
             OR delegated <> '' THEN
            RAISE EXCEPTION 'tenant_identity_writer on omninode_cloud does not hold exactly the declared grants after they ran (tables=%, any_column=%, update_columns=%, sequences=%, delegable=%). A GRANT by a role that does not own the relations warns and returns success; re-run as a role that owns them.', table_set, any_column_set, update_columns, sequence_set, delegated;
          END IF;
        END
        $tenant_identity_readback$;
        \echo [forward-migration]   ok    tenant_identity_writer holds exactly the declared grants on omninode_cloud
      \else
        \echo [forward-migration]   skip  tenant_identity_writer grants (a declared table, column or sequence does not exist in omninode_cloud yet -- the tenant identity schema seam owns them; re-asserted on the next run)
      \endif
    \else
      \echo [forward-migration]   skip  tenant_identity_writer (database omninode_cloud absent on this lane)
    \endif
  \else
    \echo [forward-migration]   FAIL  TENANT_IDENTITY_WRITER_PASSWORD is not hex -- refusing to set the tenant_identity_writer credential
    DO $tenant_identity_password$
    BEGIN
      RAISE EXCEPTION 'TENANT_IDENTITY_WRITER_PASSWORD is set and is not hexadecimal ([0-9a-fA-F]+); generate one with openssl rand -hex 32';
    END
    $tenant_identity_password$;
  \endif
\else
  \echo [forward-migration]   skip  tenant_identity_writer (TENANT_IDENTITY_WRITER_PASSWORD not set)
\endif
