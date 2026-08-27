-- OMN-15570: Rollback for 100_create_gateway_link_health.sql.
--
-- OMN-16759: 100 is now a superseded no-op (see its own header). It creates
-- nothing, so on any lane that has not already applied its ORIGINAL bytes there
-- is nothing here to reverse and running this file is a harmless no-op.
--
-- What this file is still for: the COMPOSE lanes did apply the original bytes.
-- There the forward runner connects as the postgres superuser, so the
-- since-removed `CREATE SCHEMA IF NOT EXISTS omninode_internal` succeeded and
-- these relations physically exist inside the *omnibase_infra* database -- the
-- wrong database, which is the defect OMN-16759 corrected. The live copy the
-- runtime actually reads and writes now lives in the application database
-- (omnidash_analytics), created by
-- nodes/node_gateway_link_health_write_effect/0001_create_gateway_link_health.sql
-- and reversed by rollback_node_gateway_link_health_0001.sql -- do not confuse
-- the two: running THIS file against omnidash_analytics would drop the live
-- relation.
--
-- Manual execution only, against omnibase_infra only -- never auto-applied
-- (rollback/ is not mounted to docker-entrypoint-initdb.d). Dropping the
-- orphaned compose-lane copy is operator-discretionary cleanup, not something
-- any migration does unattended.

DROP VIEW IF EXISTS omninode_internal.gateway_link_health_status;
DROP INDEX IF EXISTS omninode_internal.idx_gateway_link_health_last_seen_at;
DROP TABLE IF EXISTS omninode_internal.gateway_link_health;
