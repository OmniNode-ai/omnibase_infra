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
--   The sequence is named literally rather than resolved through
--   `pg_get_serial_sequence` inside a `DO ... EXECUTE format(...)` block, which
--   is the shape `node_pr_merged_projection/0002` uses. That shape is
--   grandfathered and closed: `scripts/ci/check_application_database_sql.py`
--   rejects a new procedural block whose relation targets cannot be proven
--   statically, and it is right to -- a gate that reads SQL cannot audit a
--   string assembled at run time. `node_merge_state_projection/0003` is the
--   open precedent and this file follows it.
--
--   The property the dynamic form bought is kept by assertion instead: section
--   3 fails loud unless `pg_get_serial_sequence` resolves this exact column to
--   this exact sequence. A NULL return (the column is not sequence-backed, so
--   `0001`'s BIGSERIAL declaration is contradicted) and a differently-named
--   sequence (a restore, a rename, an out-of-band apply) both fail the
--   migration rather than no-opping into another silent half-grant.
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
GRANT USAGE ON SEQUENCE omninode_internal.consumer_flow_windows_projection_cursor_seq TO omninode_runtime;

-- ---------------------------------------------------------------------------
-- 3. Assertions: fail the migration if any of the three facts does not hold.
--    Division by zero when the fact is false -- the fail-loud shape 0003 and
--    the sibling grant files already use.
--
--    The FIRST assertion replaces what the dynamic form gave for free: that the
--    literal sequence named above is in fact the one this column drives. The
--    THIRD is the one this ticket exists for. Asserting only the table INSERT
--    (the second) is exactly what let the broken state ship: it was TRUE for
--    the whole outage.
-- ---------------------------------------------------------------------------
SELECT 1 / count(*) AS consumer_flow_windows_cursor_sequence_identity_assertion
WHERE pg_get_serial_sequence(
          'omninode_internal.consumer_flow_windows', 'projection_cursor'
      ) = 'omninode_internal.consumer_flow_windows_projection_cursor_seq';

SELECT 1 / count(*) AS consumer_flow_windows_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'consumer_flow_windows'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS consumer_flow_windows_sequence_usage_assertion
WHERE has_sequence_privilege(
          'omninode_runtime',
          'omninode_internal.consumer_flow_windows_projection_cursor_seq',
          'USAGE'
      );
