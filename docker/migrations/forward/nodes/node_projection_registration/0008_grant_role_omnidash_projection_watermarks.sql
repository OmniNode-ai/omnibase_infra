-- OMN-19998: role_omnidash grant on the projection watermark table.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_projection_registration
--
-- BaseProjectionRunner._update_watermark upserts
-- omninode_internal.projection_watermarks after every committed batch. A
-- standalone writer that resolves its connection through
-- OMNIDASH_ANALYTICS_DB_URL connects as role_omnidash on the dev lane, and
-- role_omnidash held nothing in omninode_internal, so each upsert failed with
-- "permission denied for schema omninode_internal" and the replay cursor was
-- never persisted. The live-events writer is the one such writer left: it
-- writes public.live_events, a relation omninode_runtime cannot be granted, so
-- it cannot move to the omninode_runtime principal the way the registration and
-- tenant-registry writers did (OMN-18109). 2026-10-16 read on the dev lane's
-- Postgres log: about 6,200 "permission denied for schema omninode_internal"
-- errors in six hours, the sampled statements being the watermark upsert, still
-- arriving at 20:04Z from a role_omnidash session of the live-events writer.
--
-- Only the watermark table is granted, and only what the upsert uses: schema
-- USAGE (no CREATE, so no table can be owned and no RLS bypassed), then
-- SELECT/INSERT/UPDATE on projection_watermarks. No DELETE, no other relation.
--
-- The role must already exist; a plain GRANT fails loud where it does not, like
-- 0005 for omninode_runtime. Idempotency: GRANT is idempotent.

GRANT USAGE ON SCHEMA omninode_internal TO role_omnidash;

GRANT SELECT, INSERT, UPDATE
    ON omninode_internal.projection_watermarks
    TO role_omnidash;

-- Assertions: fail the migration if a grant did not take.
SELECT 1 / count(*) AS projection_watermarks_role_omnidash_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'projection_watermarks'
  AND grantee = 'role_omnidash'
  AND privilege_type = 'SELECT';

SELECT 1 / count(*) AS projection_watermarks_role_omnidash_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'projection_watermarks'
  AND grantee = 'role_omnidash'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS projection_watermarks_role_omnidash_update_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'projection_watermarks'
  AND grantee = 'role_omnidash'
  AND privilege_type = 'UPDATE';
