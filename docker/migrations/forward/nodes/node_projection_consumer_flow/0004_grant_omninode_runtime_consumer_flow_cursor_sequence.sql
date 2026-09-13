-- OMN-18353: sequence USAGE for omninode_runtime behind
-- consumer_flow_windows.projection_cursor.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_projection_consumer_flow
--
-- ============================================================================
-- THE HALF A TABLE GRANT DOES NOT REACH
-- ============================================================================
--   `0001_add_projection_cursor.sql` (OMN-18043) adds
--
--       ALTER TABLE omninode_internal.consumer_flow_windows
--           ADD COLUMN IF NOT EXISTS projection_cursor BIGSERIAL;
--
--   PostgreSQL rewrites BIGSERIAL into a plain `nextval()` DEFAULT over a
--   STANDALONE sequence, and it checks that sequence's OWN acl on every
--   INSERT. `GRANT INSERT ON TABLE` does not reach it.
--
--   So `omninode_runtime` held the complete, correct SELECT/INSERT/UPDATE
--   grant `0003_grant_omninode_runtime_consumer_flow_tables.sql` (OMN-17440)
--   delivers, and still failed every single write. Adding a pagination cursor
--   removed the writer's ability to insert at all.
--
--   (An IDENTITY column would NOT need this: its sequence is owned by the
--   column and rides the table's own INSERT privilege. This column is
--   BIGSERIAL, which is exactly the distinction that makes the separate grant
--   necessary.)
--
-- ============================================================================
-- PROVEN LIVE, NOT INFERRED
-- ============================================================================
--   `.201` dev lane (compose project `omnibase-infra`), read 2026-09-13T23:14Z.
--
--   `omninode-runtime`, on every heartbeat:
--
--     [ERROR] omnibase_infra.runtime.auto_wiring.handler_wiring: Projection
--     handler error: handler=ConsumerFlowProjectionWriter
--     topic=onex.evt.platform.node-heartbeat.v1
--     error_type=InsufficientPrivilegeError
--     error=permission denied for sequence
--     consumer_flow_windows_projection_cursor_seq
--
--   omnidash_analytics:
--
--     select max(window_end), count(*) from omninode_internal.consumer_flow_windows;
--       2026-09-10 19:48:02.495172+00 | 19078405
--     \dp omninode_internal.consumer_flow_windows
--       omninode_runtime=arw/postgres         <- table grant present
--     \dp omninode_internal.consumer_flow_windows_projection_cursor_seq
--       (access privileges EMPTY)             <- sequence grant absent
--
--   The freeze is total rather than partial because OMN-17379's
--   `ProjectionNotMaterializedError` correctly refuses to advance the offset on
--   a failed write. That is the right behaviour, and it is why this presented
--   as a hard stop at a fixed timestamp instead of as a silently discarding
--   consumer -- the shape `pr_merged_events` had for 24 days before OMN-17379.
--
-- ============================================================================
-- WHY THE GATE DID NOT CATCH IT
-- ============================================================================
--   `scripts/validation/check_topology_grant_delivery.py` has DERIVED this
--   requirement since OMN-17447, and reported `0 undelivered` throughout. Its
--   derivation read `CREATE TABLE` bodies only, so a SERIAL column added by
--   ALTER was invisible to it. Every ALTER of that shape already in the corpus
--   sits in the same file as a CREATE declaring the same column (the idempotent
--   reconcile pattern), so the blind spot was covered by accident everywhere it
--   existed -- until OMN-18043 landed a file carrying the ALTER and nothing
--   else. The gate is extended in the same change that lands this file; with
--   the extension and without this file it reports exactly this one relation.
--
-- ============================================================================
-- LINEAGE AND PRIVILEGES
-- ============================================================================
--   This grant belongs next to the files that own the relation, following
--   `node_pr_merged_projection/0002` (OMN-17379) and
--   `node_merge_state_projection/0003` (OMN-17447). Deliberately NOT a shared
--   cross-node grant file: that shape is what lets a relation added to a node
--   later silently miss its grant.
--
--   `0001_add_projection_cursor.sql` itself is NOT edited. It is applied on the
--   `.201` dev lane with a recorded content_sha256, so repairing it in place
--   would raise "conflicting migration checksum in canonical node history"
--   (the OMN-16705 constraint). The grant is a forward ADD.
--
--   USAGE only, which permits `nextval()` and `currval()` and
--   nothing else. Deliberately NOT `UPDATE`, which would additionally permit
--   `setval()`: the projection cursor's strict monotonicity is the contract
--   OMN-18043's `?since=<cursor>` paginated read depends on, and a writer that
--   can rewind the sequence can silently violate it.
--
--   The sequence is resolved through `pg_get_serial_sequence` rather than by
--   spelling `consumer_flow_windows_projection_cursor_seq`, so a restore, a
--   rename or an out-of-band apply that produced a differently-named sequence
--   still converges. A NULL return contradicts `0001`'s BIGSERIAL declaration
--   and fails loud rather than no-opping into another silent half-grant.
--
-- Idempotency: GRANT is idempotent; re-running is a no-op. Nothing here touches
-- RLS, ownership, table shape, or any role attribute.

-- ---------------------------------------------------------------------------
-- 1. Schema USAGE, mirroring topology
--    `principals.omninode_runtime.grants[object_type: SCHEMA,
--    schema: omninode_internal]`. Idempotent, and re-asserted here for the same
--    reason 0003 re-asserts it: a migration must not assume a sibling file ran.
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;

-- ---------------------------------------------------------------------------
-- 2. Sequence grant -- the half that was missing.
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    v_seq TEXT;
BEGIN
    v_seq := pg_get_serial_sequence(
        'omninode_internal.consumer_flow_windows', 'projection_cursor'
    );
    IF v_seq IS NULL THEN
        RAISE EXCEPTION
            'OMN-18353: omninode_internal.consumer_flow_windows.projection_cursor is not backed by a sequence, but 0001_add_projection_cursor.sql declares it BIGSERIAL. Refusing to grant a privilege on an object that does not exist -- reconcile the column shape first.';
    END IF;
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO omninode_runtime', v_seq);
END$$;

-- ---------------------------------------------------------------------------
-- 3. Assertions: fail the migration if either half did not take. Division by
--    zero when the grant is absent -- the fail-loud shape 0003 and the sibling
--    grant files already use.
--
--    The SECOND assertion is the one this ticket exists for. Asserting only the
--    table INSERT is exactly what let the broken state ship: it was TRUE for
--    the whole outage.
-- ---------------------------------------------------------------------------
SELECT 1 / count(*) AS consumer_flow_windows_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'consumer_flow_windows'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS consumer_flow_windows_sequence_usage_assertion
WHERE has_sequence_privilege(
          'omninode_runtime',
          pg_get_serial_sequence(
              'omninode_internal.consumer_flow_windows', 'projection_cursor'
          ),
          'USAGE'
      );
