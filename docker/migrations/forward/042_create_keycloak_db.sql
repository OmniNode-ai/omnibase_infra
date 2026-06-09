-- onex-create-database: keycloak
-- Ensure warm Postgres volumes have the Keycloak database before the
-- keycloak service starts. The migration runners interpret the directive
-- above before executing this SQL body so the file stays valid SQL for both
-- psql and asyncpg-based migration paths.
SELECT 1;
