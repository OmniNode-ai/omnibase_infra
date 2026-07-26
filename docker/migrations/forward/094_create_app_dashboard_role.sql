-- =============================================================================
-- MIGRATION: Create app_dashboard role (NOSUPERUSER, NOBYPASSRLS, non-owner)
-- =============================================================================
-- Ticket: OMN-14899 (blocks OMN-14894 — RLS across the projection tables)
-- Version: 1.0.0
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
--   * The SUPERUSER/REPLICATION/BYPASSRLS ALTER is privilege-guarded: only
--     the executing role's OWN attributes gate ALTER ROLE's ability to
--     change these three (Postgres core behavior, not this migration's
--     choice) — even reasserting an already-correct `false` requires the
--     executing role to already hold the attribute. RDS's master account
--     is CREATEROLE + CREATEDB but explicitly NOSUPERUSER/NOREPLICATION/
--     NOBYPASSRLS, so an unconditional ALTER of those three flags fails
--     `permission denied to alter role` on EVERY real RDS apply, not just
--     when the role pre-exists with an escalated flag. This migration only
--     attempts that ALTER when pg_roles shows one of the three already
--     set, and raises an explicit, actionable exception (naming the
--     escalated flag) rather than silently failing or silently succeeding
--     if the executing role also lacks the privilege to correct it — that
--     case needs a true superuser, by design, and must never pass quietly.
--   * NOLOGIN here: no credential material ever lives in a migration. The
--     LOGIN + password attach is a deployment-owned, operator-gated step
--     (AWS Secrets Manager per OMN-14899; local lanes may ALTER ROLE ...
--     LOGIN with lane-local credentials). Same convention as the
--     omnidash_app role in omnidash/db/migrations/0001_tenant_rls.sql.
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
--   Safe to re-run: guarded CREATE ROLE (duplicate_object / unique_violation
--   both caught — roles are cluster-wide and two migration paths may race,
--   see omnidash 0001's OMN-10875 note); the unconditional ALTER is
--   idempotent; the privilege-gated ALTER only runs its EXECUTE branch when
--   pg_roles shows an actual escalation, so a correct role never touches it.
--
-- ROLLBACK:
--   See rollback/rollback_094_create_app_dashboard_role.sql
-- =============================================================================

DO $$
BEGIN
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
      NULL; -- role already exists (possibly created concurrently)
  END;
END;
$$;

-- Enforce the non-privilege-gated flags unconditionally — these never
-- require the executing role to already hold them (RDS master account
-- compatible: CREATEROLE alone is sufficient).
ALTER ROLE app_dashboard
  NOLOGIN
  NOCREATEDB
  NOCREATEROLE;

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
