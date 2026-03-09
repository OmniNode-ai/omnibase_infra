-- Placeholder: keycloak database creation is handled by provision-keycloak.py
-- (or by docker-entrypoint-initdb.d on fresh volumes). CREATE DATABASE cannot
-- run inside a transaction block, so it is not executable by the migration
-- runner. This file reserves migration number 041 for documentation purposes.
DO $$
BEGIN
  RAISE NOTICE 'Migration 041: keycloak DB creation handled by provision-keycloak.py (no-op in migration runner)';
END
$$;
