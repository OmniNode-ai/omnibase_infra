-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- Synthetic unsafe ACL baseline. No dump, live identifier, or credential occurs here.

\set ON_ERROR_STOP on

CREATE ROLE owner_onex_tenant NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE owner_omninode_internal NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE owner_platform_catalog NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE onex_api LOGIN PASSWORD 'acl-proof-only' CREATEDB NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOREPLICATION; -- pragma: allowlist secret
CREATE ROLE tenant_projection_writer LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER BYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION; -- pragma: allowlist secret
CREATE ROLE app_dashboard LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB CREATEROLE NOREPLICATION; -- pragma: allowlist secret
CREATE ROLE omninode_runtime LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE REPLICATION; -- pragma: allowlist secret
CREATE ROLE untrusted_login LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION; -- pragma: allowlist secret
CREATE ROLE db_migrator NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE probe_migrator NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT;
CREATE ROLE legacy_probe_login LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT; -- pragma: allowlist secret
CREATE ROLE rls_admin NOLOGIN NOSUPERUSER BYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE external_connect_parent NOLOGIN NOSUPERUSER BYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE keycloak_service LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT; -- pragma: allowlist secret
CREATE ROLE omnibase_infra_service LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT; -- pragma: allowlist secret
CREATE ROLE omninode_cloud_service LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT; -- pragma: allowlist secret
CREATE ROLE omniclaude_service LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT; -- pragma: allowlist secret
CREATE ROLE omniintelligence_service LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT; -- pragma: allowlist secret
CREATE ROLE omnimemory_service LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT; -- pragma: allowlist secret
CREATE ROLE umami_service LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT; -- pragma: allowlist secret
CREATE ROLE shadow_login LOGIN PASSWORD 'acl-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION; -- pragma: allowlist secret

CREATE DATABASE keycloak;
CREATE DATABASE omnibase_infra;
CREATE DATABASE omninode_cloud;
CREATE DATABASE omniclaude;
CREATE DATABASE omniintelligence;
CREATE DATABASE omnimemory;
CREATE DATABASE umami;
CREATE DATABASE acl_scaffold_probe;

GRANT CREATE ON DATABASE keycloak TO app_dashboard;
GRANT CREATE, TEMPORARY ON DATABASE omnibase_infra TO onex_api;
GRANT CONNECT ON DATABASE keycloak TO external_connect_parent;
GRANT external_connect_parent TO rls_admin WITH ADMIN OPTION;
SET ROLE rls_admin;
GRANT external_connect_parent TO keycloak_service;
RESET ROLE;
GRANT keycloak_service TO shadow_login;
GRANT CONNECT, CREATE ON DATABASE omnibase_infra TO rls_admin WITH GRANT OPTION;
SET ROLE rls_admin;
GRANT CONNECT ON DATABASE omnibase_infra TO keycloak_service;
RESET ROLE;

ALTER DATABASE omnidash_analytics OWNER TO app_dashboard;

GRANT owner_onex_tenant TO tenant_projection_writer;
GRANT owner_omninode_internal TO omninode_runtime;
GRANT rls_admin TO app_dashboard WITH ADMIN TRUE, INHERIT TRUE, SET TRUE;
GRANT rls_admin TO owner_onex_tenant WITH ADMIN FALSE, INHERIT FALSE, SET TRUE;
GRANT owner_onex_tenant TO db_migrator
  WITH ADMIN TRUE, INHERIT TRUE, SET FALSE;

CREATE SCHEMA tenant AUTHORIZATION app_dashboard;
CREATE SCHEMA omninode_internal AUTHORIZATION tenant_projection_writer;
CREATE SCHEMA platform_catalog AUTHORIZATION onex_api;
CREATE SCHEMA legacy_acl_sentinel AUTHORIZATION owner_onex_tenant;

CREATE TABLE tenant.tenant_accounts (
  account_id BIGINT PRIMARY KEY,
  display_name TEXT NOT NULL
);
CREATE SEQUENCE tenant.delegation_events_id_seq;
CREATE TABLE tenant.delegation_events (
  id BIGINT PRIMARY KEY DEFAULT nextval('tenant.delegation_events_id_seq'),
  payload TEXT NOT NULL
);
CREATE TABLE tenant.partitioned_events (
  event_id BIGINT NOT NULL,
  occurred_at DATE NOT NULL
) PARTITION BY RANGE (occurred_at);
ALTER SEQUENCE tenant.delegation_events_id_seq OWNED BY tenant.delegation_events.id;
CREATE FUNCTION tenant.delegation_event_count()
RETURNS BIGINT
LANGUAGE sql
AS $$SELECT count(*) FROM tenant.delegation_events$$;
CREATE FUNCTION tenant.hostile_signature(IN "arg'name%" INTEGER)
RETURNS INTEGER
LANGUAGE sql
AS $$SELECT "arg'name%"$$;
CREATE PROCEDURE tenant.record_delegation(p_payload TEXT)
LANGUAGE sql
AS $$INSERT INTO tenant.delegation_events (payload) VALUES (p_payload)$$;
CREATE VIEW tenant.tenant_account_names AS
SELECT account_id, display_name FROM tenant.tenant_accounts;
CREATE TYPE tenant.account_ref AS (
  account_id BIGINT,
  display_name TEXT
);
CREATE TYPE tenant.account_id_span AS RANGE (
  subtype = BIGINT,
  multirange_type_name = tenant.account_id_span_set
);

CREATE TYPE omninode_internal.runtime_status AS ENUM ('ready', 'blocked');
CREATE DOMAIN omninode_internal.runtime_code AS TEXT
  CHECK (VALUE <> '');
CREATE TABLE omninode_internal.runtime_state (
  state_id BIGINT PRIMARY KEY,
  status omninode_internal.runtime_status NOT NULL
);
CREATE TABLE platform_catalog.plan_tiers (
  code TEXT PRIMARY KEY,
  display_name TEXT NOT NULL
);
CREATE MATERIALIZED VIEW platform_catalog.plan_tier_snapshot AS
SELECT code, display_name FROM platform_catalog.plan_tiers;

ALTER TABLE tenant.tenant_accounts OWNER TO onex_api;
ALTER TABLE tenant.delegation_events OWNER TO app_dashboard;
ALTER TABLE tenant.partitioned_events OWNER TO onex_api;
ALTER SEQUENCE tenant.delegation_events_id_seq OWNER TO app_dashboard;
ALTER FUNCTION tenant.delegation_event_count() OWNER TO app_dashboard;
ALTER FUNCTION tenant.hostile_signature(IN "arg'name%" INTEGER) OWNER TO app_dashboard;
ALTER PROCEDURE tenant.record_delegation(TEXT) OWNER TO app_dashboard;
ALTER VIEW tenant.tenant_account_names OWNER TO app_dashboard;
ALTER TYPE tenant.account_ref OWNER TO onex_api;
ALTER TYPE tenant.account_id_span OWNER TO onex_api;
ALTER TYPE tenant.account_id_span_set OWNER TO onex_api;
ALTER FUNCTION tenant.account_id_span(BIGINT, BIGINT) OWNER TO onex_api;
ALTER FUNCTION tenant.account_id_span(BIGINT, BIGINT, TEXT) OWNER TO onex_api;
ALTER FUNCTION tenant.account_id_span_set() OWNER TO onex_api;
ALTER FUNCTION tenant.account_id_span_set(VARIADIC tenant.account_id_span[]) OWNER TO onex_api;
ALTER FUNCTION tenant.account_id_span_set(tenant.account_id_span) OWNER TO onex_api;
ALTER TYPE omninode_internal.runtime_status OWNER TO omninode_runtime;
ALTER DOMAIN omninode_internal.runtime_code OWNER TO omninode_runtime;
ALTER TABLE omninode_internal.runtime_state OWNER TO tenant_projection_writer;
ALTER TABLE platform_catalog.plan_tiers OWNER TO onex_api;
ALTER MATERIALIZED VIEW platform_catalog.plan_tier_snapshot OWNER TO onex_api;

GRANT ALL PRIVILEGES ON DATABASE omnidash_analytics TO app_dashboard;
GRANT CONNECT, TEMPORARY ON DATABASE omnidash_analytics TO untrusted_login;
GRANT CREATE, USAGE ON SCHEMA tenant, omninode_internal, platform_catalog TO PUBLIC;
GRANT USAGE ON SCHEMA tenant TO untrusted_login;
GRANT USAGE ON SCHEMA omninode_internal TO app_dashboard;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA tenant, omninode_internal, platform_catalog TO app_dashboard;
GRANT SELECT ON tenant.tenant_accounts TO untrusted_login;
GRANT SELECT (display_name) ON tenant.tenant_accounts TO untrusted_login WITH GRANT OPTION;
GRANT UPDATE (display_name) ON tenant.tenant_accounts TO rls_admin WITH GRANT OPTION;
SET ROLE rls_admin;
GRANT UPDATE (display_name) ON tenant.tenant_accounts TO untrusted_login;
RESET ROLE;
GRANT SELECT ON tenant.tenant_accounts TO app_dashboard WITH GRANT OPTION;
GRANT SELECT ON tenant.tenant_accounts TO rls_admin WITH GRANT OPTION;
SET ROLE rls_admin;
GRANT SELECT ON tenant.tenant_accounts TO untrusted_login;
RESET ROLE;
GRANT SELECT ON tenant.partitioned_events TO rls_admin WITH GRANT OPTION;
SET ROLE rls_admin;
GRANT SELECT ON tenant.partitioned_events TO untrusted_login;
RESET ROLE;
GRANT TRIGGER ON tenant.tenant_accounts TO onex_api;
GRANT USAGE ON SCHEMA omninode_internal TO tenant_projection_writer;
GRANT SELECT ON omninode_internal.runtime_state TO tenant_projection_writer;
GRANT EXECUTE ON FUNCTION tenant.delegation_event_count() TO PUBLIC;
GRANT USAGE ON TYPE omninode_internal.runtime_status TO PUBLIC;

ALTER DEFAULT PRIVILEGES FOR ROLE owner_onex_tenant IN SCHEMA tenant
  GRANT SELECT ON TABLES TO app_dashboard WITH GRANT OPTION;
ALTER DEFAULT PRIVILEGES FOR ROLE owner_onex_tenant IN SCHEMA tenant
  GRANT EXECUTE ON FUNCTIONS TO PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE owner_onex_tenant
  GRANT SELECT ON TABLES TO app_dashboard;
ALTER DEFAULT PRIVILEGES FOR ROLE owner_onex_tenant
  GRANT SELECT ON TABLES TO untrusted_login;
ALTER DEFAULT PRIVILEGES FOR ROLE owner_onex_tenant IN SCHEMA tenant
  GRANT SELECT ON TABLES TO keycloak_service WITH GRANT OPTION;
ALTER DEFAULT PRIVILEGES FOR ROLE owner_onex_tenant IN SCHEMA legacy_acl_sentinel
  GRANT SELECT ON TABLES TO untrusted_login;

INSERT INTO tenant.tenant_accounts VALUES (1, 'Synthetic');
INSERT INTO tenant.delegation_events (payload) VALUES ('before-matrix');
INSERT INTO omninode_internal.runtime_state VALUES (1, 'ready');
INSERT INTO platform_catalog.plan_tiers VALUES ('beta', 'Beta');
REFRESH MATERIALIZED VIEW platform_catalog.plan_tier_snapshot;

\connect acl_scaffold_probe
CREATE TABLE public.legacy_scaffold_data(id integer PRIMARY KEY);
INSERT INTO public.legacy_scaffold_data VALUES (1);
GRANT SELECT, INSERT ON public.legacy_scaffold_data TO legacy_probe_login;
\connect omnidash_analytics
