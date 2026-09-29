-- OMN-15358: SELECT for app_dashboard on delegation_routing_tenant_overlay.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_delegation_routing_reducer
--
-- ============================================================================
-- THE PRIVILEGE THAT WAS OWNERSHIP, NOT A GRANT
-- ============================================================================
--   `0001_create_delegation_routing_tenant_overlay.sql` creates this relation
--   under `role_omnidash`, and until 2026-09-23 `role_omnidash` was also the
--   principal the dashboard read path connected as. The dashboard could read
--   this table for the same reason it could drop it: it OWNED it. No grant was
--   ever needed and none was ever written.
--
--   OMN-15358 cut that read path to `app_dashboard`, a non-owner NOBYPASSRLS
--   reader. Ownership does not travel, so the read went with the old identity.
--   Measured on onex-dev RDS before the cutover: `role_omnidash` could SELECT
--   11 relations in `public`, `app_dashboard` could SELECT 22, and this table
--   was in the first set and not the second.
--
--   The failure this prevents is quiet. The dashboard does not refuse to
--   start; one panel returns `permission denied for table
--   delegation_routing_tenant_overlay` while everything around it works.
--
-- ============================================================================
-- WHY THIS FILE EXISTS AT ALL, GIVEN THE GRANT IS ALREADY LIVE
-- ============================================================================
--   The grant was hand-applied on 2026-09-23 to unblock the cutover, as
--   `role_omnidash` (the owner), and recorded on OMN-15358. Hand-applied state
--   is not declared state: a rebuild of `omnidash_analytics` from the
--   migration corpus would come back without it. This file is the declaration
--   catching up to the database, and it is written to be a no-op against the
--   instance it was derived from.
--
--   It is a NODE migration and not a flat one on purpose.
--   `097_grant_app_dashboard_connect_omnidash_analytics.sql` is a tombstone
--   because the k8s Job that applies `docker/migrations/forward/*.sql` owns
--   only the `omnibase_infra` database and cannot reach this one, and
--   `tests/ci/test_flat_migration_no_foreign_connect_gate.py` refuses new flat
--   files that try. The node runner reaches NODE_POSTGRES_DB directly.
--
-- SELECT ONLY. `app_dashboard` holds zero non-SELECT privileges anywhere and
-- this file does not change that. No DML, no DDL, no sequence grant -- a
-- sequence grant rides INSERT, which is not granted here. Nothing in this file
-- touches RLS, ownership, or any role attribute.
--
-- Idempotency: GRANT is idempotent; re-running is a no-op.

-- ---------------------------------------------------------------------------
-- 1. The read grant.
-- ---------------------------------------------------------------------------
GRANT SELECT ON TABLE public.delegation_routing_tenant_overlay TO app_dashboard;

-- ---------------------------------------------------------------------------
-- 2. Assertion: fail the migration if the grant did not take.
--
--    A GRANT that silently does not apply is the defect class this corpus
--    keeps rediscovering (OMN-15701, OMN-16993, OMN-17374). Asserting the
--    privilege that was actually missing is the only thing that closes it.
-- ---------------------------------------------------------------------------
SELECT 1 / count(*) AS delegation_routing_tenant_overlay_app_dashboard_select_assertion
WHERE has_table_privilege(
          'app_dashboard',
          'public.delegation_routing_tenant_overlay',
          'SELECT'
      );
