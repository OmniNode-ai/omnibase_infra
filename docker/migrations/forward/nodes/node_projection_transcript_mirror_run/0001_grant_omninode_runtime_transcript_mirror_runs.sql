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

BEGIN;

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

-- Existing permissions must already respect the writer boundary. Refuse an
-- incompatible ACL without revoking it, and roll back this file's grants too.
-- Inspect column ACLs as well as table ACLs; column access must not provide
-- another principal with a path around the table-level boundary.
DO $privileges$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class c
        CROSS JOIN LATERAL (
            SELECT acl.*
            FROM pg_catalog.aclexplode(
                COALESCE(c.relacl, pg_catalog.acldefault('r', c.relowner))
            ) AS acl
            UNION
            SELECT acl.*
            FROM pg_catalog.pg_attribute a
            CROSS JOIN LATERAL pg_catalog.aclexplode(a.attacl) AS acl
            WHERE a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
        ) AS acl
        WHERE c.oid = 'omninode_internal.transcript_mirror_runs'::regclass
          AND acl.grantee <> c.relowner
          AND (
              acl.grantee <> 'omninode_runtime'::regrole
              OR acl.privilege_type NOT IN ('SELECT', 'INSERT', 'UPDATE')
              OR acl.is_grantable
          )
    ) OR EXISTS (
        -- Effective permissions also include ownership and inherited grants.
        SELECT 1
        FROM pg_catalog.pg_class c
        CROSS JOIN LATERAL pg_catalog.aclexplode(
            pg_catalog.acldefault('r', c.relowner)
        ) AS owner_privilege
        WHERE c.oid = 'omninode_internal.transcript_mirror_runs'::regclass
          AND owner_privilege.privilege_type NOT IN ('SELECT', 'INSERT', 'UPDATE')
          AND pg_catalog.has_table_privilege(
              'omninode_runtime', c.oid, owner_privilege.privilege_type
          )
    ) THEN
        RAISE EXCEPTION 'transcript_mirror_runs: incompatible privileges; expected only owner and runtime writer permissions';
    END IF;
END;
$privileges$;

COMMIT;
