-- OMN-16759: Rollback for
-- nodes/node_gateway_link_health_write_effect/0001_create_gateway_link_health.sql.
--
-- Reverses creation of the gateway_link_health_status view, the
-- gateway_link_health latest-known-state projection table, and its
-- last_seen_at index, in the application database (omnidash_analytics), where
-- the node loop applies that migration. Manual execution only -- never
-- auto-applied (rollback/ is not mounted to docker-entrypoint-initdb.d).
--
-- Does NOT drop the omninode_internal schema: this node migration asserts that
-- schema rather than creating it, so it does not own it and must not remove it.

DROP VIEW IF EXISTS omninode_internal.gateway_link_health_status;
DROP INDEX IF EXISTS omninode_internal.idx_gateway_link_health_last_seen_at;
DROP TABLE IF EXISTS omninode_internal.gateway_link_health;
