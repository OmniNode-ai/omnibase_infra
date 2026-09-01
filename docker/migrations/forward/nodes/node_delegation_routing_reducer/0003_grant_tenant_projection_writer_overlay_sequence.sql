-- OMN-17447: sequence USAGE for tenant_projection_writer behind delegation_routing_tenant_overlay.id.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_delegation_routing_reducer
--
-- ============================================================================
-- THE HALF A TABLE GRANT DOES NOT REACH
-- ============================================================================
--   `0001_create_delegation_routing_tenant_overlay.sql` declares
--   `id BIGSERIAL PRIMARY KEY`. PostgreSQL rewrites that into a plain
--   `nextval()` DEFAULT over a STANDALONE sequence, and it checks that
--   sequence's OWN acl on every INSERT. `GRANT INSERT ON TABLE` does not
--   reach it.
--
--   So `tenant_projection_writer` can hold a complete, correct
--   SELECT/INSERT/UPDATE grant on `delegation_routing_tenant_overlay` and STILL fail
--   every single write with:
--
--     InsufficientPrivilege: permission denied for sequence
--     delegation_routing_tenant_overlay_id_seq
--
--   (An IDENTITY column would NOT need this: its sequence is owned by the
--   column and rides the table's own INSERT privilege. This column is
--   BIGSERIAL, not IDENTITY, which is exactly the distinction that makes the
--   separate grant necessary.)
--
--
-- ============================================================================
-- WHY THIS IS A CLASS, NOT AN INCIDENT
-- ============================================================================
--   OMN-15423 re-pointed projections from `database: omnidash_analytics`
--   (principal `role_omnidash`, which inherits the blanket `GRANT USAGE,
--   SELECT ON ALL SEQUENCES IN SCHEMA public` from infra migration 096) to
--   `database_ref: application`. The application principals carry NO blanket
--   sequence grant anywhere in either repo, so every re-pointed projection
--   with a sequence-backed key lost the privilege silently and kept it lost.
--
--   OMN-17379 proved the consequence on the real wired path for
--   `pr_merged_events`: replay of offsets 94-96 produced three
--   InsufficientPrivilege errors, zero rows written, and the offset committed
--   anyway -- so the projection sat 24 days behind its topic while its
--   consumer group reported Stable at LAG 0. This file is the same fix for
--   the same class, one relation at a time, in the lineage that owns it --
--   the `node_pr_merged_projection/0002` convention.
--
--   The gate that makes this a bound rather than a comment is
--   `scripts/validation/check_topology_grant_delivery.py`, which OMN-17447
--   extends to DERIVE this requirement: for every declared TABLE grant
--   carrying INSERT, it resolves the relation's sequence-backed columns from
--   the applied end state of the migration corpus and requires a matching
--   sequence grant. It is not a hand list, so a new BIGSERIAL projection
--   cannot be added without either its grant or a gate failure.
--
-- Idempotency: GRANT is idempotent; re-running is a no-op. Nothing here
-- touches RLS, ownership, or any role attribute.

-- ---------------------------------------------------------------------------
-- 1. Sequence grant.
-- ---------------------------------------------------------------------------
GRANT USAGE ON SEQUENCE public.delegation_routing_tenant_overlay_id_seq TO tenant_projection_writer;

-- ---------------------------------------------------------------------------
-- 2. Assertion: fail the migration if the grant did not take.
--
--    Asserting only the TABLE privilege is precisely what let the broken state
--    ship for `pr_merged_events` -- it was TRUE for the whole 24-day outage.
--    This asserts the privilege that was actually missing.
-- ---------------------------------------------------------------------------
SELECT 1 / count(*) AS delegation_routing_tenant_overlay_id_sequence_usage_assertion
WHERE has_sequence_privilege(
          'tenant_projection_writer',
          'public.delegation_routing_tenant_overlay_id_seq',
          'USAGE'
      );
