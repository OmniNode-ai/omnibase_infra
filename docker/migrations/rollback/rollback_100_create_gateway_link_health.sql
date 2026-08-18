-- OMN-15570: Rollback for 100_create_gateway_link_health.sql.
--
-- Reverses creation of the gateway_link_health_status view, the
-- gateway_link_health latest-known-state projection table, and its
-- last_seen_at index. Manual execution only -- never auto-applied (rollback/
-- is not mounted to docker-entrypoint-initdb.d).

DROP VIEW IF EXISTS public.gateway_link_health_status;
DROP INDEX IF EXISTS public.idx_gateway_link_health_last_seen_at;
DROP TABLE IF EXISTS public.gateway_link_health;
