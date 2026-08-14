-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- Synthetic cluster bootstrap shared by the fresh and legacy fixtures.
-- No password, secret, dump, customer identifier, or live catalog value occurs here.

\set ON_ERROR_STOP on

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'omninodeadmin') THEN
    CREATE ROLE omninodeadmin NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'role_omnibase') THEN
    CREATE ROLE role_omnibase LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'role_omnidash') THEN
    CREATE ROLE role_omnidash LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'role_omniweb') THEN
    CREATE ROLE role_omniweb LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_dashboard') THEN
    CREATE ROLE app_dashboard LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'onex_api') THEN
    CREATE ROLE onex_api LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
  END IF;
END
$$;

SELECT 'CREATE DATABASE omnibase_infra'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omnibase_infra')\gexec
SELECT 'CREATE DATABASE omnidash_analytics'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omnidash_analytics')\gexec
SELECT 'CREATE DATABASE omninode_cloud'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omninode_cloud')\gexec
SELECT 'CREATE DATABASE omniintelligence'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omniintelligence')\gexec
SELECT 'CREATE DATABASE omniclaude'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omniclaude')\gexec
SELECT 'CREATE DATABASE omnimemory'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omnimemory')\gexec
SELECT 'CREATE DATABASE omniweb'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omniweb')\gexec
SELECT 'CREATE DATABASE infisical_db'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'infisical_db')\gexec
SELECT 'CREATE DATABASE keycloak'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'keycloak')\gexec

GRANT CONNECT ON DATABASE omnibase_infra TO role_omnibase;
GRANT CONNECT ON DATABASE omnidash_analytics TO role_omnidash, app_dashboard;
GRANT CONNECT ON DATABASE omninode_cloud TO onex_api;

\connect omnidash_analytics
GRANT USAGE, CREATE ON SCHEMA public TO role_omnidash;

\connect omniweb
GRANT USAGE, CREATE ON SCHEMA public TO role_omniweb;
