-- OMN-17886: Rollback for
-- nodes/node_gateway_link_health_write_effect/0002_revoke_omninode_runtime_gateway_link_health_status.sql.
--
-- Gives omninode_runtime back the privileges the forward file removed from the
-- gateway_link_health_status view: INSERT, SELECT and UPDATE, the set the
-- default-privilege rule conferred on a from-empty build. Manual execution
-- only -- never auto-applied (rollback/ is not mounted to
-- docker-entrypoint-initdb.d and no runner reads it).
--
-- Run it only on a lane where the forward file actually removed something. On
-- a lane where the role held nothing on the view, the forward file changed
-- nothing, and this file would add grants that the topology does not declare
-- and the live-ACL gate reports as UNDECLARED_GRANT.
--
-- It does not remove the forward file's ledger row, so the node loop will not
-- revoke again unless that row is removed too.
--
-- Run as the view's owner, a member of the owner role, or a superuser.

GRANT SELECT, INSERT, UPDATE ON omninode_internal.gateway_link_health_status TO omninode_runtime;
