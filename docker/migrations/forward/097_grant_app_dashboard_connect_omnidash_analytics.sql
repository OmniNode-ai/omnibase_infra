-- onex-create-database: omnidash_analytics
-- =============================================================================
-- MIGRATION: app_dashboard CONNECT + policy-gated read grants on omnidash_analytics
-- =============================================================================
-- Ticket: OMN-15297 (the grant chain never grants CONNECT — the read role
--         cannot open a session at all)
-- Related: OMN-14899/094 (the role), OMN-14894/0023 (the RLS policies + table
--          grants), OMN-15363/096 (same repair for the writer role_omnidash),
--          OMN-15355 (revoke PUBLIC's CONNECT — the change that makes this
--          latent defect fatal)
-- Version: 1.0.0
--
-- THE DEFECT
--   094 creates app_dashboard. Node migration 0023 grants USAGE ON SCHEMA
--   public and SELECT on the two RLS-covered delegation tables. Nothing in
--   that chain ever grants CONNECT ON DATABASE. On a stock Postgres CONNECT is
--   held by PUBLIC, so the gap is LATENT and every test passes. On a database
--   where PUBLIC's CONNECT has been revoked the role cannot open a session at
--   all and every grant behind it is unreachable:
--
--     FATAL:  permission denied for database "omnidash_analytics"
--     DETAIL:  User does not have CONNECT privilege.
--
--   Live readback, .201 dev lane 2026-07-28 (compose project
--   `omnibase-infra`, db `omnidash_analytics`), with LOGIN attached:
--
--     rolname       | rolcanlogin | rolsuper | rolbypassrls | can_connect
--     app_dashboard | t           | f        | f            | f
--
--   This reads like a credential problem and is a missing grant. OMN-15355
--   revokes PUBLIC's CONNECT platform-wide by design, so this is not an
--   edge case being defended against — it is the declared target state.
--
-- WHY AN ADDITIVE FILE AND NOT AN EDIT TO 0023
--   0023 is a VENDORED node migration: the authoritative copy lives in
--   omnimarket (`src/omnimarket/nodes/node_projection_delegation/migrations/`)
--   and is mirrored here by `scripts/sync-node-migrations.sh`. Editing the
--   mirror would be silently reverted by the next sync. 0023 is also on the
--   operator fence (`FENCED_NODE_MIGRATION_IDS`, OMN-15336), so a fix placed
--   inside it would not execute on any fenced lane — the repair has to live
--   where it can actually run.
--
-- WHY app_dashboard AND NOT role_omnidash
--   Two roles, two paths, both real: role_omnidash is the projection WRITER
--   (096) and app_dashboard is the dashboard READ role (094). This file is the
--   read half only. It grants no DML anywhere, and step 4 revokes write
--   privileges explicitly rather than trusting that none were granted.
--
-- WHAT THIS FILE IS NOT
--   * It carries NO credential material and does NOT grant LOGIN. The
--     LOGIN + password attach is deployment-owned and operator-gated (AWS
--     Secrets Manager, `omninode/staging/rds/app-dashboard`), exactly as 094
--     states. Re-asserting NOLOGIN would REVOKE that attach.
--   * It does NOT revoke PUBLIC's CONNECT. That is OMN-15355, it has real
--     blast radius across every role and database on the instance, and it is
--     not this ticket's to take unilaterally.
--   * It does NOT grant CREATE ON DATABASE or CREATE ON SCHEMA. A role that
--     can create objects OWNS them, and an owner is exempt from row-level
--     security — FORCE included. Granting CREATE to the read role would
--     reopen, for every future table, precisely the bypass OMN-14894 exists
--     to close.
--   * It does NOT grant SELECT on the projection VIEWS (0010). Postgres
--     evaluates RLS against the VIEW OWNER, so reading through an owner's view
--     silently bypasses tenant isolation until the views are recreated with
--     `security_invoker = true`. Step 3 filters to `relkind = 'r'` for that
--     reason, not by accident.
--
-- FAIL-CLOSED TABLE GRANTS
--   Step 3b does not grant SELECT on every table in the schema. It grants
--   only the explicit tenant-isolated delegation tables that 0023 owns:
--   public.delegation_events and public.delegation_budget_state. A blanket
--   `GRANT SELECT ON ALL TABLES IN SCHEMA public` here would make ~55
--   projection tables readable with no tenant predicate at all — a cross-
--   tenant read that every RLS test in this repo would still report green,
--   because none of them look at the uncovered tables. Readable-before-policy
--   is the ordering hazard 094's header names; the explicit table list keeps
--   the SQL statically provable by the OMN-15361 application database gate.
--
-- DATABASE CONTEXT
--   The forward runner applies `docker/migrations/forward/*.sql` against
--   POSTGRES_DB (`omnibase_infra`), so this file switches with
--   `\connect omnidash_analytics` partway through — the established in-repo
--   pattern (083, 096). GRANT ... ON DATABASE is cluster-wide and is issued
--   BEFORE the switch; every schema/table grant is issued after it. The
--   `onex-create-database` directive on line 1 is honoured by
--   `ensure_directive_database` in `scripts/run-forward-migrations.sh`, so the
--   `\connect` cannot fail on a cluster that has not been through
--   `000_create_multiple_databases.sh`.
--
-- IDEMPOTENCY
--   Safe to re-run. GRANT/REVOKE are idempotent by definition, and every
--   statement that is not is gated on an observed catalog read. No ALTER ROLE
--   appears anywhere in this file: an unconditional ALTER is not idempotent,
--   it is a privilege demand made on every apply, and it is what stopped 094
--   applying under the RDS master (OMN-14899) and then under the ordinary
--   service role the k8s Job uses (OMN-15343).
--
-- EXECUTING-ROLE REQUIREMENTS
--   GRANT requires the executing role to hold the privilege WITH GRANT OPTION
--   or to be the object owner. On compose lanes that is `postgres`; on the
--   managed instance it is the per-database migration principal, which owns
--   the database and the projection tables (OMN-15335). No CREATEROLE and no
--   superuser is required by this file.
--
-- ROLLBACK
--   See rollback/rollback_097_grant_app_dashboard_connect_omnidash_analytics.sql
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. The read role must already exist. It is created by 094, which runs first
--    (filename order) and which itself refuses to record against a role it
--    could neither find nor create. Failing loudly here rather than granting
--    into the void keeps the two files' contract explicit.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_dashboard') THEN
    RAISE EXCEPTION
      'app_dashboard role missing — apply forward migration '
      '094_create_app_dashboard_role.sql (OMN-14899) before this grant '
      'migration. Granting the read path before the constrained role exists '
      'is the ordering this work exists to prevent.';
  END IF;
END;
$$;

-- -----------------------------------------------------------------------------
-- 2. CONNECT on the target database — the defect this ticket names.
--    Issued from the current (omnibase_infra) context because
--    GRANT ... ON DATABASE is cluster-wide, and guarded on the database
--    existing so this file is valid on a cluster that has not been through
--    000_create_multiple_databases.sh.
--
--    The database name is a literal, not current_database(): a plain GRANT has
--    no current_database() shorthand, and the forward runner is connected to
--    omnibase_infra at this point, so resolving it dynamically would grant
--    CONNECT on the WRONG database.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omnidash_analytics') THEN
    EXECUTE 'GRANT CONNECT ON DATABASE omnidash_analytics TO app_dashboard';
  ELSE
    RAISE NOTICE
      'omnidash_analytics is not present on this cluster — skipping the '
      'CONNECT grant. This is expected on a bare CI cluster and never on a '
      'lane the dashboard reads from.';
  END IF;
END;
$$;

\connect omnidash_analytics

-- -----------------------------------------------------------------------------
-- 3. Schema resolution. USAGE only — CREATE is neither granted (see "WHAT THIS
--    FILE IS NOT") nor revoked (revoking it from the migration principal would
--    break the cloud migration path as a side effect, OMN-15335). USAGE alone
--    confers no data access; every row still goes through step 3b.
-- -----------------------------------------------------------------------------
GRANT USAGE ON SCHEMA public TO app_dashboard;

-- -----------------------------------------------------------------------------
-- 3b. Policy-gated SELECT. Fail-closed by construction: only the two tables
--     whose tenant_isolation policies are established by 0023 are granted.
--
--     Dynamic catalog-driven GRANT would be runtime-correct but is not
--     statically provable by the application-database authority gate. The
--     explicit list below is narrower and tied to the vendored 0023 seam.
--
--     0023 already grants these two tables on lanes where it has run. The loop
--     is not redundant with it: 0023 is operator-fenced (OMN-15336) and is a
--     vendored node migration, so on any lane where the fence holds or the
--     node chain has not run, this is the only grant the read role gets — and
--     on lanes where 0023 HAS run, GRANT is idempotent.
-- -----------------------------------------------------------------------------
GRANT SELECT ON TABLE public.delegation_events TO app_dashboard;
GRANT SELECT ON TABLE public.delegation_budget_state TO app_dashboard;

-- Least privilege is the EFFECTIVE privilege set, not the statement text:
-- 096 step 6b found a named SELECT/INSERT/UPDATE grant reading as
-- least-privilege while information_schema reported DELETE as well, re-added
-- by a blanket grant elsewhere in the chain. Revoking the write verbs in the
-- same breath pins the intent against that class.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
  ON TABLE public.delegation_events FROM app_dashboard;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
  ON TABLE public.delegation_budget_state FROM app_dashboard;

-- -----------------------------------------------------------------------------
-- 4. Post-conditions. Grants are not the isolation control — the two role
--    flags and ownership are — so both halves are asserted, not assumed.
--
--    Severities follow 096's split (and OMN-15351's): a fact THIS FILE just
--    established is FATAL when it is wrong; an environment-provisioned fact
--    this file does not own is a WARNING that ENUMERATES what it found, so it
--    is logged rather than silently tolerated and cannot wedge the migration
--    chain behind a fact only the provisioning seam can fix.
-- -----------------------------------------------------------------------------
DO $$
DECLARE
  flags RECORD;
  owned_rls_tables TEXT;
BEGIN
  -- Established by this file (step 2). FATAL.
  IF NOT has_database_privilege('app_dashboard', current_database(), 'CONNECT') THEN
    RAISE EXCEPTION
      'app_dashboard still has no CONNECT on % after this migration — the '
      'read path cannot open a session and every grant behind it is '
      'unreachable (OMN-15297)', current_database();
  END IF;

  -- Established by 094, re-read here because THIS file is what makes the role
  -- reachable: a role that can now connect and still carries an RLS-exempting
  -- flag is strictly worse than one that could not connect at all. FATAL.
  SELECT rolsuper, rolbypassrls INTO flags
    FROM pg_roles WHERE rolname = 'app_dashboard';

  IF flags.rolsuper OR flags.rolbypassrls THEN
    RAISE EXCEPTION
      'app_dashboard carries rolsuper=% / rolbypassrls=% — row-level security '
      'is INERT for this role, so granting it CONNECT would open an '
      'unfiltered cross-tenant read. Fix the role at the provisioning seam '
      'before this migration may grant it access.',
      flags.rolsuper, flags.rolbypassrls;
  END IF;

  -- Decided at each lane's provisioning seam, not here. WARNING, enumerated.
  SELECT string_agg(c.relname, ', ' ORDER BY c.relname)
    INTO owned_rls_tables
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
   WHERE c.relkind = 'r'
     AND n.nspname = 'public'
     AND c.relrowsecurity
     AND pg_get_userbyid(c.relowner) = 'app_dashboard';

  IF owned_rls_tables IS NOT NULL THEN
    RAISE WARNING
      'app_dashboard OWNS RLS-covered table(s): %. An owner is exempt from '
      'row-level security (FORCE included), so any "clean under RLS" reading '
      'taken from those tables is a false clean. Reassign ownership at the '
      'provisioning seam before citing isolation evidence from this database.',
      owned_rls_tables;
  END IF;
END;
$$;
