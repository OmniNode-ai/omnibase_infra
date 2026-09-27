-- OMN-17886: Rollback for
-- nodes/node_projection_live_events/0003_drop_retired_read_roles.sql (and, with
-- it, flat 108_drop_owned_by_retired_read_roles.sql).
--
-- Recreates jake_ro and lakshman_ro as NOLOGIN roles with nothing granted.
-- Manual execution only -- never auto-applied (rollback/ is not mounted to
-- docker-entrypoint-initdb.d and no runner reads it).
--
-- It restores the names only. Their LOGIN attribute and passwords, and every
-- grant and default-privilege entry the forward files removed, were issued by
-- hand on the lanes that had them and are recorded nowhere, so no script can
-- put them back; re-issue them by hand from the lane's readback if they are
-- wanted. Both forward files are forward-only in config/migration_classes.yaml
-- for this reason.
--
-- It does not remove the forward files' ledger rows, so the loops will not
-- drop the roles again unless those rows are removed too.
--
-- Run as a role with CREATEROLE (on the compose lanes, postgres).

CREATE ROLE jake_ro NOLOGIN;
CREATE ROLE lakshman_ro NOLOGIN;
