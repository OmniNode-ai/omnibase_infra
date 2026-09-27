-- =============================================================================
-- MIGRATION: retire jake_ro and lakshman_ro, database half 2 of 2
--            (omnidash_analytics), then drop the roles
-- =============================================================================
-- Ticket: OMN-17886, AC2 step 3 (principal removal). Half 1 is flat
-- 108_drop_owned_by_retired_read_roles.sql, which the flat loop runs first, in
-- omnibase_infra; its header carries the rulings and the .201 readback.
--
-- WHY IT LIVES UNDER THIS NODE
--   The node loop is the only sanctioned path into omnidash_analytics
--   (NODE_POSTGRES_DB in scripts/run-forward-migrations.sh). The two roles
--   belong to no node. This stream's 0002 is the node-loop owner of the
--   omninode_internal schema, the ACL surface OMN-17886 governs and the one on
--   which jake_ro held its undeclared grants, so the removal is homed here, as
--   Jonah Gray proposed on OMN-17886 (comment e9011752, 2026-09-26). Homing a
--   cross-node principal change under one node is the ownership compromise
--   node_projection_delegation_inference_response/0004 also makes, and it is
--   named rather than hidden.
--
-- WHAT IT DOES
--   1. DROP OWNED BY each role that exists, in this database.
--   2. Refuse, naming the database, while pg_shdepend still records any
--      dependency of either role anywhere in the cluster. Flat 108 has already
--      cleared omnibase_infra and refused on any database outside the two, so
--      on a lane where both files ran, nothing should remain.
--   3. DROP ROLE for each role that exists.
--   Then a static post-condition: division by zero while either role exists.
--
-- EXECUTING ROLE
--   DROP OWNED BY needs the privileges of the role, and DROP ROLE needs
--   CREATEROLE and admin rights over it. The compose lanes run this as
--   postgres. The managed (RDS) lane's migrate Job has neither (103's header):
--   where the roles are absent there this file does nothing; where they are
--   present it fails with the remediation, and nothing is ledgered.
--
-- omninode_runtime and every declared principal are untouched: only these two
-- role names are named anywhere below. Re-running is a no-op once the roles
-- are gone.
--
-- Class: forward-only (DROP). Rollback:
-- rollback/rollback_node_projection_live_events_0003.sql (manual, recreates the
-- roles without their grants).

DO $$
DECLARE
  executing_role text := current_user;
  remaining text;
BEGIN
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'jake_ro') THEN
    BEGIN
      DROP OWNED BY jake_ro;
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION USING
          ERRCODE = 'insufficient_privilege',
          MESSAGE = format(
            'the executing role %I cannot DROP OWNED BY jake_ro: it is neither '
            'a superuser nor a member of jake_ro.', executing_role),
          HINT =
            'Remove the role at the seam that holds the privilege (the instance '
            'master), then re-run this deploy; this file then does nothing for '
            'it. Ticket: OMN-17886.';
    END;
  END IF;

  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lakshman_ro') THEN
    BEGIN
      DROP OWNED BY lakshman_ro;
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION USING
          ERRCODE = 'insufficient_privilege',
          MESSAGE = format(
            'the executing role %I cannot DROP OWNED BY lakshman_ro: it is '
            'neither a superuser nor a member of lakshman_ro.', executing_role),
          HINT =
            'Remove the role at the seam that holds the privilege (the instance '
            'master), then re-run this deploy; this file then does nothing for '
            'it. Ticket: OMN-17886.';
    END;
  END IF;

  SELECT string_agg(DISTINCT coalesce(d.datname, '(shared objects)'), ', ')
    INTO remaining
    FROM pg_catalog.pg_shdepend s
    LEFT JOIN pg_catalog.pg_database d ON d.oid = s.dbid
    JOIN pg_catalog.pg_roles r ON r.oid = s.refobjid
   WHERE s.refclassid = 'pg_catalog.pg_authid'::regclass
     AND r.rolname IN ('jake_ro', 'lakshman_ro');
  IF remaining IS NOT NULL THEN
    RAISE EXCEPTION USING
      ERRCODE = 'dependent_objects_still_exist',
      MESSAGE = format(
        'jake_ro or lakshman_ro still depends on objects in %s; the roles were '
        'not dropped.', remaining),
      HINT =
        'Flat 108 clears omnibase_infra and this file clears omnidash_analytics. '
        'A dependency anywhere else needs DROP OWNED BY there as a superuser or '
        'a member of the role, then a re-run of this deploy. Ticket: OMN-17886.';
  END IF;

  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'jake_ro') THEN
    BEGIN
      DROP ROLE jake_ro;
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION USING
          ERRCODE = 'insufficient_privilege',
          MESSAGE = format(
            'the executing role %I cannot DROP ROLE jake_ro: that needs '
            'CREATEROLE and admin rights over the role.', executing_role),
          HINT =
            'Remove the role at the seam that holds the privilege (the instance '
            'master), then re-run this deploy. Ticket: OMN-17886.';
    END;
  END IF;

  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lakshman_ro') THEN
    BEGIN
      DROP ROLE lakshman_ro;
    EXCEPTION
      WHEN insufficient_privilege THEN
        RAISE EXCEPTION USING
          ERRCODE = 'insufficient_privilege',
          MESSAGE = format(
            'the executing role %I cannot DROP ROLE lakshman_ro: that needs '
            'CREATEROLE and admin rights over the role.', executing_role),
          HINT =
            'Remove the role at the seam that holds the privilege (the instance '
            'master), then re-run this deploy. Ticket: OMN-17886.';
    END;
  END IF;
END
$$;

-- Post-condition: division by zero while either role still exists.
SELECT 1 / (count(*) = 0)::int AS retired_read_roles_absent_assertion
FROM pg_catalog.pg_roles
WHERE rolname IN ('jake_ro', 'lakshman_ro');
