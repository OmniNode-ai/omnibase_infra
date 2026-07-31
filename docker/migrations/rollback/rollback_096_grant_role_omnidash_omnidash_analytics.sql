-- =============================================================================
-- ROLLBACK: 096_grant_role_omnidash_omnidash_analytics.sql
-- =============================================================================
-- Ticket: OMN-15363
--
-- SCOPE
--   Revokes the grants migration 095 issued.  It deliberately does NOT drop
--   role_omnidash: the role predates 095 on every lane that has one
--   (000_create_multiple_databases.sh on compose, out-of-band provisioning on
--   cloud RDS), and dropping it would take down the cloud migration principal.
--
-- WARNING BEFORE RUNNING THIS
--   Any lane whose analytics DSN connects AS role_omnidash — the .201 lab lane
--   after docker/docker-compose.dev-lane.yml — loses its projection write path
--   the moment these revokes land.  Repoint that DSN FIRST.  Rolling back to
--   the `postgres` DSN also restores the superuser/BYPASSRLS connection, which
--   makes every RLS/FORCE state on this database inert again; that is the
--   condition OMN-15363 exists to remove, so this rollback is an emergency
--   surface, not a maintenance one.
-- =============================================================================

\connect omnidash_analytics

ALTER DEFAULT PRIVILEGES IN SCHEMA public
  REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM role_omnidash;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  REVOKE USAGE, SELECT ON SEQUENCES FROM role_omnidash;

REVOKE SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public FROM role_omnidash;
REVOKE USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public FROM role_omnidash;
REVOKE USAGE ON SCHEMA public FROM role_omnidash;

\connect omnibase_infra

REVOKE CONNECT ON DATABASE omnidash_analytics FROM role_omnidash;
