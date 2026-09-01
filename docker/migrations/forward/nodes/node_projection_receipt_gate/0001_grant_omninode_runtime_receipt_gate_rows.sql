-- OMN-17440: topology-derived omninode_runtime TABLE grants for node_projection_receipt_gate.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_projection_receipt_gate
--
-- ============================================================================
-- WHAT IS BROKEN
-- ============================================================================
--   The topology ALREADY declares these grants. All three instances
--   (`omnibase_infra/src/omnibase_infra/topology/instances/{local,onex-dev,
--   onex-prod}.yaml`) carry `receipt_gate_rows` in
--   `databases.application.principals.omninode_runtime.grants[object_type:
--   TABLE, schema: public]` with `[INSERT, SELECT, UPDATE]`, and that list is
--   GENERATED from node contract `db_io.db_tables` declarations by
--   `scripts/generate_application_database_table_grants.py --write`.
--
--   No migration has ever ISSUED them. The two halves -- what the platform
--   declares, and what a database is actually told -- drifted apart silently,
--   and the drift is only ever discovered as a live outage on whichever
--   relation happens to take traffic next.
--
--   The `.201` dev lane holds most of these grants live, which is precisely
--   what kept the gap invisible: they were granted OUT OF BAND, BY HAND. A
--   lane that has never been hand-patched -- a fresh staging namespace, a
--   rebuilt onex-dev, prod -- gets NONE of them, and every projection writing
--   these relations refuses on first traffic with `InsufficientPrivilege`
--   while the runtime reports healthy and commits its offsets.
--
--   PROVEN LIVE, not inferred: `receipt_gate_rows`
--   is among the relations OMN-17377's dead-table survey found sitting at
--   ZERO ROWS, and OMN-17379 reproduced the cause on the real wired path --
--   replay of a real retained event produced `InsufficientPrivilege`, zero
--   rows written, and the offset committed anyway. The silent-ack half of that
--   defect is owned by OMN-17379 and is NOT repaired here; this file removes
--   the CAUSE.
--
-- ============================================================================
-- WHY THIS LINEAGE, AND NOT ONE BULK GRANT FILE
-- ============================================================================
--   This grant belongs next to the file that creates the relation, in the
--   lineage of the node that OWNS it -- the convention
--   `node_projection_session_replay/0002` (OMN-16993),
--   `node_projection_tenant_registry/0001` (OMN-17374) and
--   `node_log_persistence_effect/0000` already follow.
--
--   Deliberately NOT a shared cross-node grant file. That is the shape
--   `0004_grant_tenant_projection_writer.sql` took under protest -- "Homing a
--   cross-node grant block under one node is a real ownership compromise" --
--   and it is exactly why a relation added to a node later silently misses its
--   grant. OMN-15701 is that failure: a pin regeneration reverted eight
--   house-tenant grants at once because they all lived in one place.
--
--   `omnibase_infra` flat migration 099 creates the role and grants it CONNECT
--   plus USAGE on `omninode_internal`. It cannot carry the `public`-schema
--   table half for a node-owned relation: the flat corpus is applied by a
--   `psql -f` loop gated on `directive_db == "$DB_NAME"`, while the node-owned
--   loop is the sanctioned path that connects directly to omnidash_analytics
--   (NODE_POSTGRES_DB). So the AUTHORIZATION rides here.
--
-- ============================================================================
-- PRIVILEGES
-- ============================================================================
--   SELECT, INSERT, UPDATE and deliberately NO DELETE -- a projection writer
--   upserts, it does not reshape the table. Same invariant 096 states for
--   role_omnidash, 099 for this principal on `omninode_internal.live_events`,
--   and `node_projection_tenant_registry/0001` for this principal on the
--   registry mirror. PostgreSQL requires SELECT alongside INSERT/UPDATE for
--   the adapter's `INSERT ... ON CONFLICT DO UPDATE`, which is why the write
--   set is three privileges and not two.
--
--   This widens NOTHING beyond what the generated topology already declares.
--   It grants only the relations THIS node owns and claims nothing about the
--   corpus-wide residual, which stays measured and ratcheted by
--   `scripts/validation/check_topology_grant_delivery.py`.
--
--   Physical location is bare `public`: 0000_create_receipt_gate_projection_table.sql issues an
--   unqualified CREATE TABLE, like every other relation still awaiting the
--   OMN-15359 schema cutover. A grant must name the PHYSICAL schema, so it
--   says `public`.
--
-- Idempotency: GRANT is idempotent; re-running is a no-op. Nothing here
-- touches RLS, ownership, or any role attribute.
--
-- NOTE (OMN-17447):
--   a TABLE grant alone is still not sufficient to write these relations:
--   `receipt_gate_rows` is sequence-backed, and an INSERT that omits the
--   key fails at the SEQUENCE before it ever reaches the table. That half
--   is OMN-17447 and lands in this same lineage; neither half works
--   alone.

-- ---------------------------------------------------------------------------
-- 1. Schema USAGE, mirroring topology
--    `principals.omninode_runtime.grants[object_type: SCHEMA, schema: public]`.
--    Idempotent, and re-asserted here for the same reason 099 re-asserts the
--    omninode_internal one: a migration must not assume a sibling file ran.
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA public TO omninode_runtime;

-- ---------------------------------------------------------------------------
-- 2. Table grants (topology-derived, OMN-17440)
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE ON public.receipt_gate_rows TO omninode_runtime;

-- ---------------------------------------------------------------------------
-- 3. Assertions: fail the migration if a grant did not take. Division by zero
--    when the grant is absent -- the same fail-loud shape 099,
--    node_log_persistence_effect/0000, node_projection_session_replay/0002 and
--    node_projection_tenant_registry/0001 already use.
--
--    BOTH directions are asserted per relation, not just INSERT. A lane where
--    the INSERT landed and the SELECT did not would accept writes and still
--    fail every `ON CONFLICT DO UPDATE` read-back, which is the harder of the
--    two failures to diagnose.
-- ---------------------------------------------------------------------------
SELECT 1 / count(*) AS receipt_gate_rows_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'public'
  AND table_name = 'receipt_gate_rows'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS receipt_gate_rows_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'public'
  AND table_name = 'receipt_gate_rows'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'SELECT';
