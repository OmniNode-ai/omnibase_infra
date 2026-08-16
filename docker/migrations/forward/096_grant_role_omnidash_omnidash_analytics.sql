-- =============================================================================
-- TOMBSTONE (OMN-15846, 2026-08-10): this file is UNDELIVERABLE via the k8s
-- Job that applies docker/migrations/forward/*.sql
-- (omninode_infra/k8s/migrations/omnibase-infra-migrate.yaml) and, on the
-- one target where it matters (onex-dev RDS), UNNEEDED besides.
--
-- UNDELIVERABLE: that Job owns only the omnibase_infra database; its flat
-- loop's `psql -f` apply is gated on `directive_db == $DB_NAME` and is
-- UNREACHABLE for this file's `\connect omnidash_analytics` below, in that
-- loop or any other in the runner. It has never executed there --
-- live-confirmed 2026-08-10 (omninode-dev-postgres RDS):
-- `pg_default_acl` for omnidash_analytics/public is EMPTY (zero rows) --
-- this file's own unconditional step 6
-- `ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ... TO role_omnidash`
-- would ALWAYS leave a pg_default_acl entry if it had ever run to
-- completion under ANY executing role; none exists. omnibase_infra's own
-- schema_migrations carried a false "applied" row for this file
-- (applied_at 2026-08-01T22:00:18Z, checksum byte-identical to this file's
-- live content) -- the same OMN-15819 masking class OMN-15846's
-- classification-ordering fix unmasked for 098/099.
--
-- UNNEEDED ON RDS: this file's OWN header below documents its purpose as a
-- FORCE-RLS non-owner-pool repair for the `.201 lab lane` specifically
-- (compose project `omnibase-infra`) -- read the "PURPOSE" section that
-- follows, unchanged. On onex-dev RDS, role_omnidash is not a marginal
-- non-owner role at all: it is the RDS migration principal (OMN-15335
-- two-owner split) and OWNS 89 of 90 tables in omnidash_analytics.public
-- (live-verified via pg_tables.tableowner, 2026-08-10) -- an owner is
-- exempt from RLS regardless of rolsuper/rolbypassrls (this file's own
-- step 7 assertion states the same fact), and an owner needs no SELECT/
-- INSERT/UPDATE grant this file would add. role_omnidash's CREATE on
-- schema public and its one non-owned-table grant (generation_events) are
-- both live-verified as granted BY `pg_database_owner`/omninodeadmin (the
-- provisioning-seam path), not by role_omnibase_infra (the identity this
-- file would have executed under) -- a second, independent confirmation
-- this file never delivered anything on this database. OMN-15355 (P1, "In
-- Review" as of 2026-08-10) is the tracked, systematic successor: it
-- generates the complete ACL/default-privilege matrix from contracts
-- across every database/schema/table, explicitly scoped to include
-- role_omnidash-owned domains and app_dashboard's access boundary --
-- superseding this file's stopgap intent rather than leaving a gap next
-- to it.
--
-- Kept in place, byte-unchanged below this header, as ledgered history
-- (migration files are append-only) -- do NOT delete it and do NOT try to
-- make it deliverable in place; it remains applicable, unchanged, to the
-- .201 lab lane it was authored for (that lane's own runner,
-- scripts/run-forward-migrations.sh, is a different code path from the
-- k8s Job this tombstone concerns and is unaffected by it). Static
-- pre-merge enforcement in THIS repo:
-- tests/ci/test_flat_migration_no_foreign_connect_gate.py /
-- docker/migrations/forward/cross-database-flat-migrations.yaml.
-- =============================================================================
-- MIGRATION: role_omnidash authorization on omnidash_analytics (warm-volume safe)
-- =============================================================================
-- Ticket: OMN-15363 (open question 1 — "should the compose lane provision
--         role_omnidash, as the k8s/RDS path does?"  Answer: yes, and the
--         AUTHORIZATION half belongs in the migration chain.)
-- Related: OMN-15416 (P0/P3 gate — prove FORCE-RLS with real non-owner pools),
--          OMN-15351 (0027's role_omnidash guard), OMN-14899/094 (app_dashboard)
-- Version: 1.0.0
--
-- PURPOSE
--   Postgres exempts a table's OWNER and any role with SUPERUSER or BYPASSRLS
--   from row-level security unconditionally — FORCE included.  So on a lane
--   whose only connecting role is `postgres` (superuser + rolbypassrls), the
--   RLS/FORCE state on a table is INERT: every policy evaluates against nothing.
--
--   Live readback, .201 lab lane (compose project `omnibase-infra`), DB
--   `omnidash_analytics`, 2026-07-31T03:33Z — 400 consecutive samples of
--   pg_stat_activity returned exactly ONE client, `postgres` from 172.19.0.10:
--
--     select rolname, rolsuper, rolbypassrls from pg_roles where rolcanlogin;
--       postgres     | t | t
--       role_omniweb | f | f
--
--   node_service_registry, projection_delegation_inference_response_text and
--   savings_estimates all carry `relrowsecurity` AND `relforcerowsecurity` on
--   that lane and are all owned by `postgres`.  The lane has therefore been
--   observed "clean under FORCE" for ~10h47m without ever having exercised the
--   mechanism against a single non-exempt connection.
--
--   This migration provisions the AUTHORIZATION half of the repair: the
--   non-owner, non-superuser, non-BYPASSRLS role the runtime can connect as.
--   The DSN half (which role the lab lane actually connects with) is
--   `docker/docker-compose.dev-lane.yml`, and the CREDENTIAL half
--   (LOGIN + password) stays deployment-owned — see "WHAT THIS FILE IS NOT".
--
-- WHY role_omnidash AND NOT A NEW ROLE
--   role_omnidash is already the declared per-service identity for
--   `omnidash_analytics`: `docker/migrations/forward/000_create_multiple_databases.sh`
--   maps `omnidash_analytics:role_omnidash:ROLE_OMNIDASH_PASSWORD`, node
--   migration 0027 grants it the generation_events writer set, and it is the
--   role the cloud RDS path already connects as.  Minting a second writer role
--   for the same shape would be the duplicate this repo's one-canonical-model
--   rule exists to prevent.
--
-- WHY A MIGRATION AND NOT JUST 000
--   000_create_multiple_databases.sh is the sanctioned bootstrap, and it is
--   correct — but it is mounted at `/docker-entrypoint-initdb.d` and therefore
--   runs ONLY when the postgres data directory is empty.  Every warm lane (the
--   .201 lab lane included, whose volume long predates any ROLE_OMNIDASH_PASSWORD
--   being configured) can never receive those grants.  `ROLE_OMNIDASH_PASSWORD`
--   was unset when that volume was initialised, so 000 printed
--   `SKIP: role_omnidash — ROLE_OMNIDASH_PASSWORD not set` and skipped the role,
--   its grants, and its CONNECT privilege in one step.  This file re-establishes
--   the grant half on warm volumes through the forward-migration one-shot, which
--   IS part of the sanctioned deploy path.
--
-- WHAT THIS FILE IS NOT
--   It carries NO credential material and does NOT grant LOGIN.  That is the
--   same invariant migration 094 states for app_dashboard: the LOGIN + password
--   attach is deployment-owned (AWS Secrets Manager on the cloud path;
--   ROLE_OMNIDASH_PASSWORD consumed by 000 on compose lanes).  Re-asserting
--   NOLOGIN on a pre-existing role would REVOKE that deployment-owned attach, so
--   NOLOGIN here is a CREATE-TIME default only and is never re-asserted.
--
--   It also does NOT grant CREATE ON SCHEMA public.  A role that can create
--   tables OWNS them, and an owner is exempt from RLS — granting CREATE to the
--   application role would reopen, for every future table, exactly the bypass
--   this file exists to close.  It equally does NOT *revoke* CREATE: on cloud
--   RDS role_omnidash is the migration principal and performs DDL (OMN-15335), so
--   a blanket revoke here would break the cloud migration path as a side effect.
--   The invariant that is asserted instead — fail-closed, at the bottom of this
--   file — is that role_omnidash owns none of the RLS-covered tables.
--
-- DATABASE CONTEXT
--   The forward runner applies `docker/*.sql` against POSTGRES_DB
--   (`omnibase_infra`), so this file switches with `\connect omnidash_analytics`
--   partway through — the established in-repo pattern (see migration 083), and
--   the runner's `ensure_directive_database` handles the directive explicitly.
--   Role attributes and GRANT ... ON DATABASE are cluster-wide and are issued
--   BEFORE the switch; every schema/table grant is issued after it.
--
-- IDEMPOTENCY
--   Safe to re-run.  The CREATE is skipped when pg_roles already shows the role
--   (and keeps the duplicate_object/unique_violation handler for the genuine
--   concurrent-creation race).  Every ALTER ROLE runs only on an OBSERVED
--   divergence — an unconditional ALTER is not idempotent, it is a privilege
--   demand made on every apply, and Postgres gates SUPERUSER/BYPASSRLS on the
--   EXECUTING role's own attributes even when reasserting an already-correct
--   `false` (094's finding, and the reason the RDS migration principal can apply
--   this file at all).  GRANTs are idempotent by definition.
--
-- ROLLBACK
--   See rollback/rollback_096_grant_role_omnidash_omnidash_analytics.sql
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Role existence.  Guarded so a role WITHOUT CREATEROLE can apply this file:
--    Postgres checks create-role privilege BEFORE it checks whether the name is
--    taken, so an unconditional CREATE ROLE raises 42501 rather than the
--    duplicate_object the handler below is written for.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'role_omnidash') THEN
    BEGIN
      CREATE ROLE role_omnidash WITH
        NOLOGIN
        NOSUPERUSER
        NOBYPASSRLS
        NOREPLICATION;
    EXCEPTION
      WHEN duplicate_object OR unique_violation THEN
        NULL; -- created concurrently by another migration path
    END;
  END IF;
END;
$$;

-- -----------------------------------------------------------------------------
-- 2. The two flags that decide whether RLS applies to this role at all.
--    Deliberately scoped to rolsuper + rolbypassrls: rolcreatedb/rolcreaterole
--    are legitimately held by role_omnidash on cloud RDS, where it is the
--    migration principal (OMN-15335).  Correcting those here would break that
--    path; neither affects RLS exemption.
--
--    Gated on observed divergence.  If the flag IS set and the executing role
--    cannot correct it, fail loudly and name it — never silently succeed, never
--    silently leave the escalation in place.
-- -----------------------------------------------------------------------------
DO $$
DECLARE
  current_flags RECORD;
BEGIN
  SELECT rolsuper, rolbypassrls
    INTO current_flags
    FROM pg_roles
   WHERE rolname = 'role_omnidash';

  IF NOT FOUND THEN
    RAISE EXCEPTION
      'role_omnidash does not exist and could not be created — the executing '
      'role lacks CREATEROLE. On a managed instance the role is provisioned at '
      'the provisioning seam; this migration refuses to record itself against a '
      'role that is not there.';
  END IF;

  IF current_flags.rolsuper OR current_flags.rolbypassrls THEN
    BEGIN
      EXECUTE 'ALTER ROLE role_omnidash NOSUPERUSER NOBYPASSRLS';
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION
          'role_omnidash carries an RLS-exempting flag (rolsuper=%, '
          'rolbypassrls=%) and the executing role lacks privilege to correct '
          'it — a true superuser must fix this role before any FORCE ROW LEVEL '
          'SECURITY state on omnidash_analytics can be trusted',
          current_flags.rolsuper, current_flags.rolbypassrls;
    END;
  END IF;
END;
$$;

-- -----------------------------------------------------------------------------
-- 3. CONNECT on the target database.  Issued from the current (omnibase_infra)
--    context because GRANT ... ON DATABASE is cluster-wide, and guarded on the
--    database existing so this file is valid on a cluster that has not been
--    through 000 (e.g. a bare CI Postgres).
--    OMN-15297 is the same defect for app_dashboard: policy + table grants
--    without CONNECT is a role that cannot open a session at all.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omnidash_analytics') THEN
    EXECUTE 'GRANT CONNECT ON DATABASE omnidash_analytics TO role_omnidash';
  END IF;
END;
$$;

\connect omnidash_analytics

-- -----------------------------------------------------------------------------
-- 4. Schema resolution.  USAGE only — see "WHAT THIS FILE IS NOT" above for why
--    CREATE is neither granted nor revoked here.
-- -----------------------------------------------------------------------------
GRANT USAGE ON SCHEMA public TO role_omnidash;

-- -----------------------------------------------------------------------------
-- 5. Named, least-privilege DML on the tables that are RLS-FORCED today.
--    These three are spelled out rather than left to the blanket grant in step 6
--    because they are the tables whose isolation this ticket is about: a future
--    narrowing of step 6 must not silently drop the writer's access to them, and
--    a reader of this file should be able to see the exact privilege set the
--    proving ground runs under without deriving it.
--
--    SELECT, INSERT, UPDATE mirrors node migration 0027's writer set for
--    generation_events.  No DELETE, no TRUNCATE, no REFERENCES, no TRIGGER: a
--    projection writer upserts, it does not reshape the table.  Each grant is
--    guarded on the table existing, because the registration trio is fenced off
--    every lane except the lab (OMN-15379) and savings/inference-response tables
--    arrive from their own node migration chains.
-- -----------------------------------------------------------------------------
DO $$
DECLARE
  forced_table text;
BEGIN
  FOREACH forced_table IN ARRAY ARRAY[
    'node_service_registry',
    'projection_delegation_inference_response_text',
    'savings_estimates'
  ] LOOP
    IF to_regclass('public.' || forced_table) IS NOT NULL THEN
      EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON public.%I TO role_omnidash',
        forced_table
      );
    ELSE
      RAISE NOTICE
        'skipping grant on public.% — table not present on this lane', forced_table;
    END IF;
  END LOOP;
END;
$$;

-- -----------------------------------------------------------------------------
-- 6. The rest of the analytics surface.
--    Not decoration: `omnidash_analytics` carries ~55 projection tables on the
--    lab lane and the runtime writes across them.  Cutting the connection to a
--    non-owner role without these grants would trade a silent RLS bypass for a
--    loud 42501 on every projection that is NOT RLS-covered — which proves
--    nothing about isolation and takes the lane's whole projection path down.
--    This is the same grant set 000_create_multiple_databases.sh applies to
--    every per-service role; it is reproduced here so warm volumes converge on
--    the same state a fresh volume gets.
--
--    ALTER DEFAULT PRIVILEGES applies only to objects created BY THE EXECUTING
--    ROLE (same caveat 000 carries).  On compose lanes that is `postgres`, which
--    is also the migration/table owner, so future projection tables inherit the
--    grant.  A lane whose migrations run as a different principal must set its
--    own default privileges at its provisioning seam.
-- -----------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO role_omnidash;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO role_omnidash;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO role_omnidash;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO role_omnidash;

-- -----------------------------------------------------------------------------
-- 6b. Re-narrow the RLS-FORCED tables after the blanket grant.
--     Step 5's named grant is SELECT/INSERT/UPDATE, but step 6's
--     `ON ALL TABLES IN SCHEMA public` is a SUPERSET of it and silently re-adds
--     DELETE to the very three tables step 5 was careful to exclude it from.
--     Order matters and this is the only order that works: a REVOKE placed
--     before the blanket GRANT would simply be overwritten.
--
--     Verified live on the lab lane 2026-07-31T03:56Z: without this block,
--     `information_schema.role_table_grants` reported
--     `DELETE,INSERT,SELECT,UPDATE` on all three — the named grant read as
--     least-privilege while the effective privilege set was not. A claim that is
--     true of a statement but false of the resulting state is the failure mode
--     this block removes.
--
--     TRUNCATE / REFERENCES / TRIGGER are revoked in the same breath: they are
--     not in the blanket grant today, so these are no-ops that pin the intent
--     against a future widening of step 6.
-- -----------------------------------------------------------------------------
DO $$
DECLARE
  forced_table text;
BEGIN
  FOREACH forced_table IN ARRAY ARRAY[
    'node_service_registry',
    'projection_delegation_inference_response_text',
    'savings_estimates'
  ] LOOP
    IF to_regclass('public.' || forced_table) IS NOT NULL THEN
      EXECUTE format(
        'REVOKE DELETE, TRUNCATE, REFERENCES, TRIGGER ON public.%I FROM role_omnidash',
        forced_table
      );
    END IF;
  END LOOP;
END;
$$;

-- -----------------------------------------------------------------------------
-- 7. Post-conditions.
--    Grants are not the isolation control — ownership and the two role flags
--    are.  If role_omnidash owns an RLS-covered table, every policy on that
--    table is inert for it no matter what this file granted, and the lane would
--    report a false clean.  Assert both, do not assume either.
--
--    The two assertions have DELIBERATELY DIFFERENT severities, following the
--    OMN-15351 split: a fact this file itself just established is FATAL when it
--    is wrong, an environment-provisioned fact this file does not own is a
--    WARNING that ENUMERATES what it found.
--      * role flags — set by step 2 of this same file.  EXCEPTION.
--      * table ownership — decided at each lane's provisioning seam.  On cloud
--        RDS role_omnidash owns part of the schema by design (OMN-15335's
--        two-owner split), so a RAISE here would wedge the entire cloud
--        migration chain behind a lab-lane ticket.  WARNING, named table by
--        table, so the state is logged rather than silently tolerated.
-- -----------------------------------------------------------------------------
DO $$
DECLARE
  owned_rls_tables text;
  flags RECORD;
BEGIN
  SELECT string_agg(c.relname, ', ' ORDER BY c.relname)
    INTO owned_rls_tables
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
   WHERE c.relkind = 'r'
     AND n.nspname = 'public'
     AND c.relrowsecurity
     AND pg_get_userbyid(c.relowner) = 'role_omnidash';

  IF owned_rls_tables IS NOT NULL THEN
    RAISE WARNING
      'role_omnidash OWNS RLS-covered table(s) in omnidash_analytics: %. An '
      'owner is exempt from row-level security (FORCE included), so RLS on '
      'these tables is INERT against the application role and any "clean under '
      'RLS" reading taken from them is a false clean. Reassign ownership at the '
      'provisioning seam before citing RLS evidence from this database.',
      owned_rls_tables;
  END IF;

  SELECT rolsuper, rolbypassrls INTO flags
    FROM pg_roles WHERE rolname = 'role_omnidash';

  IF flags.rolsuper OR flags.rolbypassrls THEN
    RAISE EXCEPTION
      'role_omnidash still carries rolsuper=% / rolbypassrls=% after this '
      'migration — RLS would be inert for it',
      flags.rolsuper, flags.rolbypassrls;
  END IF;
END;
$$;
