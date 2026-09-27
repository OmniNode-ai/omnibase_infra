-- =============================================================================
-- MIGRATION: retire jake_ro and lakshman_ro, database half 1 of 2 (omnibase_infra)
-- =============================================================================
-- Ticket: OMN-17886, AC2 step 3 (principal removal).
-- Rulings: Jonah Gray in #standups 2026-09-24 22:31 IST ("remove lakshman_ro,
--   since OMN-16963 is done") and 22:55 IST ("fold that into OMN-17886 along
--   with lakshman_ro"), after Jake Brower at 22:44 IST ("We can remove
--   jake_ro. I don't use it."). Shape: OMN-17886 comment e9011752
--   (2026-09-26): forward-only; the node-owned half under
--   node_projection_live_events.
--
-- WHAT IS WRONG
--   Two read-only login roles were created by hand on the .201 lanes, are
--   declared in no topology instance, and hold grants the live-ACL gate reports
--   as undeclared: jake_ro held SELECT on every omninode_internal table through
--   a postgres-owned default-privilege rule. Read-only readback of .201 on
--   2026-09-25T02:10Z: jake_ro has dependencies in omnibase_infra (79) and
--   omnidash_analytics (97) plus database privileges; lakshman_ro has 2 in
--   omnibase_infra plus database privileges; neither is a superuser or a
--   member of any role.
--
-- WHY TWO FILES
--   A role's owned objects and grants are per database, and DROP OWNED BY acts
--   only on the database it runs in (plus shared objects: database CONNECT and
--   the like). The flat loop connects to omnibase_infra and runs first; the
--   node loop connects to omnidash_analytics and runs after it
--   (scripts/run-forward-migrations.sh sections 2 and 3). So this file clears
--   omnibase_infra, and nodes/node_projection_live_events/0003 clears
--   omnidash_analytics and then drops the roles.
--
-- NOTHING HALF-REMOVED
--   Before it drops anything, this file refuses if either role still depends
--   on an object in any database other than those two, and names it. A
--   dependency there is one no migration here can reach, and removing the
--   omnibase_infra half first would leave a role with grants gone in one
--   database and present in another.
--
-- EXECUTING ROLE
--   DROP OWNED BY needs the privileges of the role being dropped: a superuser,
--   or a member of it. The compose lanes run this as postgres. The managed
--   (RDS) lane's migrate Job holds only NOCREATEROLE identities that are not
--   members of these roles (see 103's header). Where a role is absent this
--   file does nothing for it; where it is present and cannot be dropped by the
--   executing identity, it fails with the remediation, and nothing is ledgered.
--
-- WHY 108, AND THE MANAGED-LANE CONDITION ON MERGING IT
--   Ordinals 104 and 107 are burned (_ledger/retired-flat-migrations.tsv), and
--   for this file's own failure shape: each refused on the onex-dev serving RDS
--   for want of a privilege the migrate Job does not hold, and because the flat
--   loop is migration-order 1 of 6 of deploy-onex-staging, the refusal stopped
--   every staging deploy until the file was retired. This file refuses only
--   where one of the two roles EXISTS and cannot be dropped. So it merges only
--   after a read-only readback of the onex-dev RDS shows neither role in
--   pg_roles there (OMN-17886 comment e9011752 allows that readback). If
--   either role is present there, it is removed at the master-credential seam
--   first, and this file then does nothing on that lane.
--
-- omninode_runtime and every declared principal are untouched: only these two
-- role names are named anywhere below.
--
-- Class: forward-only (DROP; a role's dropped grants cannot be reconstructed
-- by any down script, because they were issued by hand). No rollback file: the
-- grants this file removes were issued by hand and nothing records them, so a
-- down script could only guess. The roles themselves are recreated, without
-- grants, by rollback/rollback_node_projection_live_events_0003.sql.

DO $$
DECLARE
  executing_role text := current_user;
  unreachable text;
BEGIN
  SELECT string_agg(DISTINCT d.datname, ', ' ORDER BY d.datname)
    INTO unreachable
    FROM pg_catalog.pg_shdepend s
    JOIN pg_catalog.pg_database d ON d.oid = s.dbid
    JOIN pg_catalog.pg_roles r ON r.oid = s.refobjid
   WHERE s.refclassid = 'pg_catalog.pg_authid'::regclass
     AND r.rolname IN ('jake_ro', 'lakshman_ro')
     AND d.datname NOT IN ('omnibase_infra', 'omnidash_analytics');
  IF unreachable IS NOT NULL THEN
    RAISE EXCEPTION USING
      ERRCODE = 'dependent_objects_still_exist',
      MESSAGE = format(
        'jake_ro or lakshman_ro still depends on objects in database(s) %s, '
        'which no forward migration reaches; nothing was dropped.', unreachable),
      HINT =
        'Run DROP OWNED BY for the role in each named database as a superuser '
        'or a member of the role, then re-run this deploy. Ticket: OMN-17886.';
  END IF;

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
END
$$;
