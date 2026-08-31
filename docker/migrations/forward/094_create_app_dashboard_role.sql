-- =============================================================================
-- MIGRATION: Create app_dashboard role (NOSUPERUSER, NOBYPASSRLS, non-owner)
-- =============================================================================
-- Ticket: OMN-14899 (blocks OMN-14894 — RLS across the projection tables)
--         OMN-15343 (must apply under an ordinary, non-CREATEROLE role on RDS)
-- Version: 1.1.0
--
-- PURPOSE:
--   app_dashboard is the RUNTIME connection role for the dashboard/projection
--   read path against omnidash_analytics. Postgres silently bypasses
--   row-level security for a table's owner and for any role with BYPASSRLS
--   or SUPERUSER — so the connecting role, not the policy, is the actual
--   isolation control. This migration creates that role with both bypass
--   flags off and NO table ownership, so the RLS policies landed under
--   OMN-14894 are enforced against every read the dashboard makes.
--
-- DESIGN INVARIANTS:
--   * NOSUPERUSER + NOBYPASSRLS are ENFORCED on every run (guarded ALTER
--     after the guarded CREATE), not just requested at create time. A
--     pre-existing app_dashboard role with either flag set is corrected,
--     never trusted.
--   * EVERY privileged statement is gated on an OBSERVED DIVERGENCE. Postgres
--     requires role-administration rights for ALTER ROLE, and the executing
--     role's OWN attributes additionally gate SUPERUSER / REPLICATION /
--     BYPASSRLS — even reasserting an already-correct `false`. An
--     unconditional ALTER is therefore not "idempotent": it is a privilege
--     demand made on every apply. This file reads pg_roles first and issues a
--     statement only when observed state differs from required state. Two
--     consequences, both proven by execution in
--     tests/integration/migrations/test_094_app_dashboard_role.py:
--       - RDS's master account (CREATEROLE + CREATEDB but explicitly
--         NOSUPERUSER / NOREPLICATION / NOBYPASSRLS) applies this file
--         cleanly on a fresh cluster.
--       - An ORDINARY service role with NO CREATEROLE at all applies it
--         cleanly when the role already exists in the required state. That is
--         the live cloud case (OMN-15343): the k8s migration Job holds no
--         cluster-admin credential on the managed instance by design, and the
--         managed instance has no `postgres` role at all (live readback
--         2026-07-29: `select count(*) from pg_roles where rolname='postgres'`
--         -> 0), so this file has to be a true no-op there or it can never be
--         recorded — and a migration that cannot be recorded blocks every
--         later migration behind it.
--     When a divergence IS observed and the executing role cannot correct it,
--     this migration raises an explicit, actionable exception naming the flag.
--     It never silently succeeds and never silently leaves an escalation in
--     place.
--   * NOLOGIN is a CREATE-TIME default, deliberately NOT re-asserted on a
--     pre-existing role. No credential material ever lives in a migration: the
--     LOGIN + password attach is a deployment-owned, operator-gated step (AWS
--     Secrets Manager per OMN-14899; local lanes may ALTER ROLE ... LOGIN with
--     lane-local credentials). Re-asserting NOLOGIN would REVOKE that
--     deployment-owned attach — on the cloud instance app_dashboard already
--     carries LOGIN (live readback 2026-07-29: rolcanlogin = t), so a blanket
--     `ALTER ROLE app_dashboard NOLOGIN` here would break the dashboard's
--     runtime connection as a side effect of recording a migration. LOGIN is
--     not the isolation control for this role in any case: NOSUPERUSER,
--     NOBYPASSRLS and non-ownership are, and all three are still enforced.
--   * app_dashboard must NEVER own tables. Table creation stays with the
--     migration/runtime role (postgres on compose lanes). Ownership would
--     silently bypass ENABLE ROW LEVEL SECURITY.
--   * Role-only migration: roles are cluster-wide, so this file is valid in
--     any database context (it deliberately contains no \connect and no
--     GRANT). Schema USAGE and per-table SELECT grants ride WITH the RLS
--     migrations in omnidash_analytics (OMN-14894 tranches) so a table is
--     never readable by app_dashboard before its tenant_isolation policy
--     exists.
--
-- IDEMPOTENCY:
--   Safe to re-run: the CREATE is skipped when pg_roles already shows the role
--   AND still carries the duplicate_object / unique_violation guard for the
--   genuine race (roles are cluster-wide and two migration paths may race, see
--   omnidash 0001's OMN-10875 note); every ALTER runs only on an observed
--   divergence, so a correct role touches none of them.
--
-- ROLLBACK:
--   See rollback/rollback_094_create_app_dashboard_role.sql
-- =============================================================================

-- Guarded CREATE. The pg_roles pre-check is what makes this file runnable by a
-- role WITHOUT CREATEROLE: Postgres checks create-role privilege BEFORE it
-- checks whether the name is already taken, so an unconditional CREATE ROLE
-- raises `permission denied to create role` (42501) rather than the
-- duplicate_object the handler below is written for. The handler is retained
-- for the genuine race: two migration paths creating the role concurrently.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_dashboard') THEN
    BEGIN
      CREATE ROLE app_dashboard WITH
        NOLOGIN
        NOSUPERUSER
        NOBYPASSRLS
        NOCREATEDB
        NOCREATEROLE
        NOREPLICATION;
    EXCEPTION
      WHEN duplicate_object OR unique_violation THEN
        NULL; -- role already exists (created concurrently)
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION USING
          ERRCODE = 'insufficient_privilege',
          MESSAGE = format(
            'app_dashboard does not exist on this cluster and the executing role %I '
            'cannot create it: CREATE ROLE requires the CREATEROLE attribute.',
            current_user),
          DETAIL =
            'Roles are cluster-scoped. Every migration identity on the managed '
            'lane (role_omnibase_infra for the flat loop, role_omnidash for the '
            'node loop) is provisioned NOCREATEROLE by contract, and the '
            'instance has no superuser role this Job can authenticate as '
            '(OMN-15343). No loop in this corpus can create the role. This file already RAISEd a provisioning-seam message from its NEXT block, but the raw CREATE above aborted the file before that block was ever reached (OMN-17301).',
          HINT =
            'Provision it once at the seam that holds the privilege, then '
            're-run: from omninode_infra, scripts/provision-cluster-roles.sh '
            '--apply. This migration is an idempotent no-op once the role '
            'exists. Ticket: OMN-17301.';
    END;
  END IF;
END;
$$;

-- CREATEDB / CREATEROLE: not gated by the executing role's own attributes the
-- way the three below are, but ALTER ROLE still demands role-administration
-- rights, so this is gated on an observed divergence too. Immediately after the
-- guarded CREATE, and on every re-run against a correct role, both flags are
-- already false and no statement is issued at all.
DO $$
DECLARE
  current_flags RECORD;
BEGIN
  SELECT rolcreatedb, rolcreaterole
    INTO current_flags
    FROM pg_roles
   WHERE rolname = 'app_dashboard';

  IF NOT FOUND THEN
    RAISE EXCEPTION
      'app_dashboard role does not exist and could not be created — the '
      'executing role lacks CREATEROLE. On a managed instance the role is '
      'provisioned at the provisioning seam (OMN-15343); this migration '
      'refuses to record itself against a role that is not there.';
  END IF;

  IF current_flags.rolcreatedb OR current_flags.rolcreaterole THEN
    BEGIN
      EXECUTE 'ALTER ROLE app_dashboard NOCREATEDB NOCREATEROLE';
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION
          'app_dashboard carries an unexpected privilege (rolcreatedb=%, '
          'rolcreaterole=%) and the executing role lacks the role-administration '
          'rights to correct it — fix this role at the provisioning seam before '
          'the dashboard read path can be trusted',
          current_flags.rolcreatedb, current_flags.rolcreaterole;
    END;
  END IF;
END;
$$;

-- SUPERUSER/BYPASSRLS/REPLICATION are privilege-gated by Postgres core: the
-- executing role must already hold an attribute to ALTER it, even to
-- reassert an already-correct `false`. Only attempt the ALTER when one of
-- the three is actually set (the common case — immediately after the
-- guarded CREATE above, and on every idempotent re-run — never reaches
-- this block at all, so it never trips the RDS permission error that a
-- blanket ALTER did). If the flag is set AND the executing role lacks the
-- privilege to correct it, fail loudly and name the problem — never
-- silently succeed and never silently leave the escalation in place.
DO $$
DECLARE
  current_flags RECORD;
BEGIN
  SELECT rolsuper, rolbypassrls, rolreplication
    INTO current_flags
    FROM pg_roles
   WHERE rolname = 'app_dashboard';

  IF current_flags.rolsuper
     OR current_flags.rolbypassrls
     OR current_flags.rolreplication THEN
    BEGIN
      EXECUTE 'ALTER ROLE app_dashboard NOSUPERUSER NOBYPASSRLS NOREPLICATION';
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION
          'app_dashboard has an escalated flag (rolsuper=%, rolbypassrls=%, '
          'rolreplication=%) and the executing role lacks privilege to '
          'correct it — a true Postgres superuser must fix this role '
          'manually before RLS on app_dashboard-read tables can be trusted',
          current_flags.rolsuper, current_flags.rolbypassrls,
          current_flags.rolreplication;
    END;
  END IF;
END;
$$;
