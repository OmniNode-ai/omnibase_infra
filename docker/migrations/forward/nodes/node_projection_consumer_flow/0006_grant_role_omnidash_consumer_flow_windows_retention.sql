-- OMN-17427: role_omnidash grant for the consumer_flow_windows retention prune.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_projection_consumer_flow
--
-- omnimarket node_retention_cleanup (contract 1.3.0, omnimarket#4035) prunes
-- omninode_internal.consumer_flow_windows to 7 days, in batches of
-- DELETE ... WHERE ctid = ANY(ARRAY(SELECT ctid ... WHERE window_start < ...)).
-- Its retention_policy.schedule.database_url_env is OMNIDASH_ANALYTICS_DB_URL,
-- which connects as role_omnidash on the dev lane. role_omnidash held schema
-- USAGE (node_projection_registration/0008) and nothing on this relation, so
-- every run was refused: the dev lane's Postgres log at 10:03Z and 10:38Z
-- on 2026-10-17 shows the prune's DELETE refused, first for the schema, then,
-- after 0008 landed, with "permission denied for table consumer_flow_windows".
--
-- Nothing has ever pruned the relation. Read on the dev lane at
-- 2026-10-17T10:50Z: 32 GB with indexes, about 13.6M live rows. The
-- projection-api's 15 s command timeout on the first page of
-- onex.snapshot.projection.consumer-flow.v1 cancels under host load
-- ("canceling statement due to user request" at 08:31:04Z and 10:18:14Z),
-- which the api answers 503 and the C28 board probe grades INDETERMINATE.
--
-- Only this relation is granted, and only what the prune statement uses:
-- SELECT (its WHERE and subquery read window_start, projection_cursor and
-- ctid) and DELETE. No INSERT, UPDATE, TRUNCATE or CREATE, and no other
-- relation. The writer principal omninode_runtime keeps SELECT/INSERT/UPDATE
-- and no DELETE (0003): the projection writer upserts, the retention node
-- prunes.
--
-- The role must already exist; a plain GRANT fails loud where it does not, like
-- node_projection_registration/0008. Idempotency: GRANT is idempotent.

GRANT USAGE ON SCHEMA omninode_internal TO role_omnidash;

GRANT SELECT, DELETE
    ON omninode_internal.consumer_flow_windows
    TO role_omnidash;

-- Assertions: fail the migration if a grant did not take.
SELECT 1 / count(*) AS consumer_flow_windows_role_omnidash_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'consumer_flow_windows'
  AND grantee = 'role_omnidash'
  AND privilege_type = 'SELECT';

SELECT 1 / count(*) AS consumer_flow_windows_role_omnidash_delete_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'consumer_flow_windows'
  AND grantee = 'role_omnidash'
  AND privilege_type = 'DELETE';
