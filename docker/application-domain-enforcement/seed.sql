-- SPDX-License-Identifier: MIT

CREATE ROLE owner_onex_tenant NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE owner_omninode_internal NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE owner_platform_catalog NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
CREATE ROLE tenant_control_admin NOLOGIN NOSUPERUSER BYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;

CREATE ROLE onex_api LOGIN PASSWORD 'domain-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION; -- pragma: allowlist secret
CREATE ROLE tenant_projection_writer LOGIN PASSWORD 'domain-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION; -- pragma: allowlist secret
CREATE ROLE app_dashboard LOGIN PASSWORD 'domain-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION; -- pragma: allowlist secret
CREATE ROLE omninode_runtime LOGIN PASSWORD 'domain-proof-only' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION; -- pragma: allowlist secret

CREATE SCHEMA tenant AUTHORIZATION owner_onex_tenant;
CREATE SCHEMA omninode_internal AUTHORIZATION owner_omninode_internal;
CREATE SCHEMA platform_catalog AUTHORIZATION owner_platform_catalog;

CREATE TABLE tenant.tenants (
    id UUID PRIMARY KEY,
    tenant_name TEXT NOT NULL
);
ALTER TABLE tenant.tenants OWNER TO owner_onex_tenant;
ALTER TABLE tenant.tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant.tenants FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_identity_isolation ON tenant.tenants
    AS PERMISSIVE FOR ALL TO PUBLIC
    USING (id = current_setting('app.tenant_id', true)::uuid)
    WITH CHECK (id = current_setting('app.tenant_id', true)::uuid);

INSERT INTO tenant.tenants (id, tenant_name) VALUES
    ('11111111-1111-1111-1111-111111111111', 'tenant-a'),
    ('22222222-2222-2222-2222-222222222222', 'tenant-b');

CREATE TABLE tenant.events (
    event_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    payload TEXT NOT NULL
);
ALTER TABLE tenant.events OWNER TO owner_onex_tenant;
ALTER TABLE tenant.events ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant.events FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON tenant.events
    AS PERMISSIVE FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

INSERT INTO tenant.events (event_id, tenant_id, payload) VALUES
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1', '11111111-1111-1111-1111-111111111111', 'tenant-a-first'),
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa2', '11111111-1111-1111-1111-111111111111', 'tenant-a-second'),
    ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbb1', '22222222-2222-2222-2222-222222222222', 'tenant-b-only');

CREATE VIEW tenant.events_view WITH (security_invoker = true) AS
    SELECT event_id, tenant_id, payload FROM tenant.events;
ALTER VIEW tenant.events_view OWNER TO owner_onex_tenant;

CREATE FUNCTION tenant.safe_report() RETURNS INTEGER
    LANGUAGE sql
    SECURITY DEFINER
    SET search_path = pg_catalog, tenant, pg_temp
    AS 'SELECT count(*)::integer FROM tenant.events';
ALTER FUNCTION tenant.safe_report() OWNER TO owner_onex_tenant;
REVOKE ALL ON FUNCTION tenant.safe_report() FROM PUBLIC;

CREATE TABLE omninode_internal.runtime_state (
    state_id UUID PRIMARY KEY,
    source_tenant_id UUID NULL,
    payload TEXT NOT NULL
);
ALTER TABLE omninode_internal.runtime_state OWNER TO owner_omninode_internal;

CREATE TABLE platform_catalog.feature_flags (
    flag_id UUID PRIMARY KEY,
    flag_name TEXT NOT NULL UNIQUE
);
ALTER TABLE platform_catalog.feature_flags OWNER TO owner_platform_catalog;

GRANT CONNECT ON DATABASE omnidash_analytics TO onex_api, tenant_projection_writer, app_dashboard, omninode_runtime;
GRANT USAGE ON SCHEMA tenant TO onex_api, tenant_projection_writer, app_dashboard;
GRANT USAGE ON SCHEMA tenant TO tenant_control_admin;
GRANT SELECT, INSERT ON tenant.tenants TO onex_api, tenant_control_admin;
GRANT SELECT ON tenant.events TO onex_api, tenant_projection_writer, app_dashboard;
GRANT SELECT ON tenant.events_view TO onex_api, tenant_projection_writer, app_dashboard;
GRANT INSERT, UPDATE, DELETE ON tenant.events TO onex_api, tenant_projection_writer;
GRANT EXECUTE ON FUNCTION tenant.safe_report() TO app_dashboard;
GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON omninode_internal.runtime_state TO omninode_runtime;
GRANT USAGE ON SCHEMA platform_catalog TO onex_api, app_dashboard, omninode_runtime;
GRANT SELECT ON platform_catalog.feature_flags TO onex_api, app_dashboard, omninode_runtime;
