-- Ensure warm Postgres volumes have the Keycloak database before the
-- keycloak service starts. The forward-migration runner executes SQL files
-- with psql, so \gexec can run CREATE DATABASE outside a transaction while
-- remaining idempotent.
SELECT 'CREATE DATABASE keycloak'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'keycloak'
)\gexec
