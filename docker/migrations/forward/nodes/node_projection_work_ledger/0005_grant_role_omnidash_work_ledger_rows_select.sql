-- OMN-17427: role_omnidash read grant on omninode_internal.work_ledger_rows.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_projection_work_ledger
--
-- omnimarket node_lab_job_check_effect (handlers/postgres_lab_job_check_store.py)
-- reads omninode_internal.lab_job_state, omninode_internal.work_ledger_rows,
-- omninode_internal.pr_state and public.hook_events, and only reads: its guard
-- refuses any statement carrying a write verb. The store's pool is the shared
-- AsyncpgAdapter the runtime binds, which on the dev lane's
-- omninode-runtime-effects connects as role_omnidash. role_omnidash held schema
-- USAGE (node_projection_registration/0008) and SELECT on public.hook_events,
-- but nothing on the three omninode_internal relations, so every lab job check
-- was refused: the dev lane Postgres log read 201 "permission denied" lines in
-- the 24h to 2026-10-22T06:51Z, 200 of them "permission denied for table
-- lab_job_state" (the SELECT of every non-terminal job, arriving every five
-- minutes), the last at 06:48Z.
--
-- Only this relation is granted, and only SELECT. No INSERT, UPDATE, DELETE or
-- TRUNCATE, and no other relation. The writer principal omninode_runtime keeps
-- its own grants; the check node reads what the projection wrote.
--
-- The role must already exist; a plain GRANT fails loud where it does not, like
-- node_projection_registration/0008. Idempotency: GRANT is idempotent.

GRANT USAGE ON SCHEMA omninode_internal TO role_omnidash;

GRANT SELECT
    ON omninode_internal.work_ledger_rows
    TO role_omnidash;

-- Assertions: fail the migration if a grant did not take.
SELECT 1 / count(*) AS work_ledger_rows_role_omnidash_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'work_ledger_rows'
  AND grantee = 'role_omnidash'
  AND privilege_type = 'SELECT';
