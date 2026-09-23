-- OMN-15358: SELECT for app_dashboard on projection_delegation_savings_series.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_projection_savings
--
-- ============================================================================
-- THE SAME OWNERSHIP GAP, THE SECOND RELATION
-- ============================================================================
--   This relation is owned by `role_omnidash`, which was the dashboard read
--   path's connection identity until OMN-15358 cut it to `app_dashboard`. The
--   old role read it by ownership; the new role, a non-owner NOBYPASSRLS
--   reader, had no grant and no path to one.
--
--   Read the companion file
--   `nodes/node_delegation_routing_reducer/0006_grant_app_dashboard_overlay.sql`
--   for the full reasoning -- these two relations were the entire difference
--   between what the outgoing role could read and what the incoming one could,
--   measured on onex-dev RDS before the cutover.
--
--   This table is empty today (0 bytes at the time of writing), which is
--   exactly why it is worth declaring now rather than when it is not: a
--   missing grant on an empty relation produces no symptom until the first row
--   lands, and then produces one that looks like a data bug.
--
-- ============================================================================
-- WHY A NODE MIGRATION
-- ============================================================================
--   The flat forward runner cannot reach `omnidash_analytics` -- the k8s Job
--   owns only the `omnibase_infra` database, which is why
--   `097_grant_app_dashboard_connect_omnidash_analytics.sql` is a tombstone
--   and why `tests/ci/test_flat_migration_no_foreign_connect_gate.py` refuses
--   new flat files that try to `\connect` across. The node runner applies
--   against NODE_POSTGRES_DB directly.
--
-- SELECT ONLY, on a TABLE. `088_savings_views_invoker_scoped.sql` made the
-- savings VIEWS `security_invoker`; this file grants on the relation and not
-- on a view, so RLS is evaluated against the querying role either way. No DML,
-- no DDL, no sequence grant, no RLS or ownership change.
--
-- Idempotency: GRANT is idempotent; re-running is a no-op.
--
-- The grant was hand-applied on 2026-09-23 to unblock the cutover and is
-- recorded on OMN-15358. This file is the declaration catching up to the
-- database, and is a no-op against the instance it was derived from.

-- ---------------------------------------------------------------------------
-- 1. The read grant.
-- ---------------------------------------------------------------------------
GRANT SELECT ON TABLE public.projection_delegation_savings_series TO app_dashboard;

-- ---------------------------------------------------------------------------
-- 2. Assertion: fail the migration if the grant did not take.
-- ---------------------------------------------------------------------------
SELECT 1 / count(*) AS projection_delegation_savings_series_app_dashboard_select_assertion
WHERE has_table_privilege(
          'app_dashboard',
          'public.projection_delegation_savings_series',
          'SELECT'
      );
