-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- OMN-22260: omninode_runtime grants for omninode_internal.transcript_mirror_runs.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_projection_transcript_mirror_run
--
-- WHY A GRANT FILE AT ALL
--   The topology DECLARES the grant (its TABLE list is generated from the
--   declaration of this table), but declaring a grant is not issuing one.
--   Without this file every write fails with InsufficientPrivilege while the
--   consumer group reports Stable and commits its offsets (OMN-17377,
--   OMN-17379). The grant rides in the owning node's lineage, beside the file
--   that creates the relation, never in a shared cross-node grant file
--   (OMN-15701).
--
-- WHY SELECT, INSERT, UPDATE AND NOTHING ELSE
--   The reducer is the only writer. It reads the stored row of an attempt,
--   inserts the start row and updates it with the terminal state; the
--   orchestrator reads the newest start time through the same runtime role.
--   Nothing deletes an attempt: the table is the record of what the mirror did.
--   The table has no sequence, so there is no sequence grant. No other role is
--   granted anything here.
--
-- Idempotency: GRANT is idempotent; re-running is a no-op.

GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;

GRANT SELECT, INSERT, UPDATE
    ON omninode_internal.transcript_mirror_runs
    TO omninode_runtime;

-- Assertions: fail the migration if a grant did not take.
SELECT 1 / count(*) AS transcript_mirror_runs_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'transcript_mirror_runs'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'SELECT';

SELECT 1 / count(*) AS transcript_mirror_runs_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'transcript_mirror_runs'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS transcript_mirror_runs_update_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'transcript_mirror_runs'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'UPDATE';
