-- OMN-17440: topology-derived omninode_runtime TABLE grants for node_projection_consumer_flow.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_projection_consumer_flow
--
-- ============================================================================
-- WHAT IS BROKEN
-- ============================================================================
--   All three topology instances
--   (`omnibase_infra/src/omnibase_infra/topology/instances/{local,onex-dev,
--   onex-prod}.yaml`) declare `consumer_flow_windows` and `topic_produce_windows` under
--   `databases.application.principals.omninode_runtime.grants[object_type:
--   TABLE, schema: omninode_internal]` with `[INSERT, SELECT, UPDATE]`. That list is
--   GENERATED from node contract `db_io.db_tables` declarations by
--   `scripts/generate_application_database_table_grants.py --write`, so it is
--   the platform's machine-derived statement of intent.
--
--   No migration in either repo has ever ISSUED them. The two halves --
--   what the platform declares, and what a database is actually told -- drifted
--   apart silently, and the drift only ever surfaces as a live outage on
--   whichever relation happens to take traffic next: `InsufficientPrivilege`
--   on every write while the runtime reports healthy and commits its offsets
--   (the OMN-16993 / OMN-17379 failure shape).
--
-- ============================================================================
-- MEASURED, NOT ASSUMED
-- ============================================================================
--   `scripts/validation/check_topology_grant_delivery.py` at omnibase_infra
--   dev 4053dc3c0 reported `topology grant delivery: 27 undelivered (of 65
--   declared)`, and `consumer_flow_windows`, `topic_produce_windows` were among them.
--
--   Read-only probe of the `.201` dev lane (compose project `omnibase-infra`,
--   database `omnidash_analytics`) on 2026-09-03 against
--   `information_schema.role_table_grants`: 63 of the 65 declared grants are
--   present live, but only 38 of them are issued by any migration. The other
--   26 -- `consumer_flow_windows`, `topic_produce_windows` included -- exist ONLY because they were granted
--   out of band, BY HAND. That is precisely why the gap stayed invisible: a
--   lane that has never been hand-patched (a fresh staging namespace, a rebuilt
--   onex-dev, prod) gets none of them.
--
-- ============================================================================
-- WHY THIS LINEAGE, AND NOT ONE BULK GRANT FILE
-- ============================================================================
--   This grant belongs next to the file that creates the relation, in the
--   lineage of the node that OWNS it -- the convention
--   `node_projection_session_replay/0002` (OMN-16993),
--   `node_projection_tenant_registry/0001` (OMN-17374) and the eight files the
--   first OMN-17440 tranche landed already follow.
--
--   Deliberately NOT a shared cross-node grant file. That is the shape
--   `0004_grant_tenant_projection_writer.sql` took under protest -- "Homing a
--   cross-node grant block under one node is a real ownership compromise" --
--   and it is exactly why a relation added to a node later silently misses its
--   grant. OMN-15701 is that failure: a pin regeneration reverted eight
--   house-tenant grants at once because they all lived in one place.
--
--   The flat corpus cannot carry it either. Flat migrations are applied by a
--   `psql -f` loop gated on `directive_db == "$DB_NAME"`, while the node-owned
--   loop is the sanctioned path that connects directly to omnidash_analytics
--   (NODE_POSTGRES_DB). So the AUTHORIZATION rides here.
--
-- ============================================================================
-- PRIVILEGES
-- ============================================================================
--   SELECT, INSERT, UPDATE and deliberately NO DELETE -- a projection writer
--   upserts, it does not reshape the table. Same invariant flat 096 states for
--   role_omnidash and flat 099 for this principal on
--   `omninode_internal.live_events`. PostgreSQL requires SELECT alongside
--   INSERT/UPDATE for the adapter's `INSERT ... ON CONFLICT DO UPDATE`, which
--   is why the write set is three privileges and not two.
--
--   This widens NOTHING beyond what the generated topology already declares.
--   It grants only the relations THIS node owns.
--
--   Physical schema is `omninode_internal`: `0000_create_consumer_flow_windows.sql` is the file that creates
--   them, and a grant must name the PHYSICAL schema.
--
--   `0000_create_consumer_flow_windows.sql` itself is NOT edited. It is applied on the `.201` dev
--   lane with a recorded content_sha256, so repairing its grant block in place
--   would raise "conflicting migration checksum in canonical node history"
--   (the OMN-16705 constraint). The grant is therefore a forward ADD.
--
-- Idempotency: GRANT is idempotent; re-running is a no-op. Nothing here
-- touches RLS, ownership, table shape, or any role attribute.

-- ---------------------------------------------------------------------------
-- 1. Schema USAGE, mirroring topology
--    `principals.omninode_runtime.grants[object_type: SCHEMA, schema: omninode_internal]`.
--    Idempotent, and re-asserted here for the same reason flat 099 re-asserts
--    the omninode_internal one: a migration must not assume a sibling file ran.
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;

-- ---------------------------------------------------------------------------
-- 2. Table grants (topology-derived, OMN-17440)
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE ON omninode_internal.consumer_flow_windows TO omninode_runtime;
GRANT SELECT, INSERT, UPDATE ON omninode_internal.topic_produce_windows TO omninode_runtime;

-- ---------------------------------------------------------------------------
-- 3. Assertions: fail the migration if a grant did not take. Division by zero
--    when the grant is absent -- the same fail-loud shape flat 099,
--    node_log_persistence_effect/0000, node_projection_session_replay/0002 and
--    node_projection_tenant_registry/0001 already use.
--
--    BOTH directions are asserted per relation, not just INSERT. A lane where
--    the INSERT landed and the SELECT did not would accept writes and still
--    fail every `ON CONFLICT DO UPDATE` read-back, which is the harder of the
--    two failures to diagnose.
-- ---------------------------------------------------------------------------
SELECT 1 / count(*) AS consumer_flow_windows_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'consumer_flow_windows'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS consumer_flow_windows_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'consumer_flow_windows'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'SELECT';

SELECT 1 / count(*) AS topic_produce_windows_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'topic_produce_windows'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS topic_produce_windows_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'topic_produce_windows'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'SELECT';
