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
--   * NOSUPERUSER + NOBYPASSRLS are ENFORCED on every run (ALTER after the
--     guarded CREATE), not just requested at create time. A pre-existing
--     app_dashboard role with either flag set is corrected, never trusted.
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
--   see omnidash 0001's OMN-10875 note), ALTER ROLE is idempotent.
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

-- Enforce the security-critical flags even when the role pre-existed.
-- Presence is not the property that matters — the flags are.
ALTER ROLE app_dashboard
  NOLOGIN
  NOSUPERUSER
  NOBYPASSRLS
  NOCREATEDB
  NOCREATEROLE
  NOREPLICATION;
