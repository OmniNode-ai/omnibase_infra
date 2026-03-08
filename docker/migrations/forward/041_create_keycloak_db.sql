-- Creates keycloak database on first postgres container boot.
-- init scripts only run on empty postgres_data volume.
-- For existing volumes, provision-keycloak.py handles it via TCP.
SELECT 'CREATE DATABASE keycloak'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'keycloak'
)\gexec
