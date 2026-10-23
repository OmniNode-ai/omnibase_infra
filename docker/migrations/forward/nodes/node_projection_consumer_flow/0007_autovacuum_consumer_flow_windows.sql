-- =============================================================================
-- MIGRATION: vacuum consumer_flow_windows on its own scale, not the default
-- =============================================================================
-- Owner:   omnimarket.nodes.node_projection_consumer_flow
-- Ticket:  OMN-17427 (M4 criterion C28)
--
-- WHAT WAS BROKEN
--   The consumer-flow exposure serves the newest window of every
--   (consumer_group, topic) pair through a skip scan of the primary key: one
--   index probe to reach each next pair and one backward probe for its newest
--   window, so the read follows the pair count (838 pairs), not the row count.
--   Both probes are index-only scans, and an index-only scan is only index-only
--   on heap pages the visibility map marks all-visible. Pages are marked by
--   VACUUM.
--
--   With the default thresholds (autovacuum_vacuum_scale_factor and
--   autovacuum_vacuum_insert_scale_factor 0.2) a table of 11.47M rows waits for
--   2.3M dead or 2.3M newly inserted tuples before autovacuum visits it. On the
--   lab dev lane on 2026-10-22 the last autovacuum was 2026-10-21 02:52Z, with
--   2,090,711 dead tuples outstanding, and the exposure read measured:
--
--     cold   24.0 s  (838 skip-scan steps at 27 ms, 230,578 heap fetches)
--     warm    1.2 s  (the same plan once the first read had marked the dead
--                     index entries; heap fetches 3)
--
--   The projection API's command timeout is 15 s, so the cold read answered
--   503 projection_database_unavailable and the C28 board probe could not look
--   (rc=2).
--
-- WHAT THIS DOES
--   Sets per-table autovacuum thresholds at 2% of the table (about 230k tuples
--   at today's size, a few hours of windows) for dead tuples, for inserts and
--   for analyze. Insert-driven vacuum (PostgreSQL 13 and later) is the part that
--   matters for the read: it marks the pages new windows land on all-visible,
--   so the newest-window probes stay index-only. Dead-tuple vacuum removes the
--   index entries a replayed or gap-then-observed window leaves behind, which
--   the skip-scan step otherwise walks.
--
--   Storage parameters only. No row, column, index or grant changes, and
--   ALTER TABLE ... SET takes SHARE UPDATE EXCLUSIVE, which does not block
--   reads or the writer's upserts.
--
-- WHAT THIS DOES NOT DO
--   It does not bound the table. Retention is node_consumer_flow_prune_effect's
--   (OMN-19658, 30-day floor), and the writer's catch-up is the
--   ConsumerFlowProjectionWriter change that keeps one pool and batches each
--   heartbeat.
-- =============================================================================

ALTER TABLE omninode_internal.consumer_flow_windows SET (
    autovacuum_vacuum_scale_factor = 0.02,
    autovacuum_vacuum_insert_scale_factor = 0.02,
    autovacuum_analyze_scale_factor = 0.02
);
