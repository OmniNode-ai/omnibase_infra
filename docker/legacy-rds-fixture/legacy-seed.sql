-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Wholly synthetic legacy-RDS fixture for OMN-15422.
-- Current application DB names: omnibase_infra, omnidash_analytics,
-- omninode_cloud. Representative roles: omninodeadmin, role_omnidash,
-- app_dashboard, onex_api. Values are invented controls, never dump-derived.

\set ON_ERROR_STOP on

-- Checksum-capable Python/compose ledger and the flat half of a dual producer.
\connect omnibase_infra
CREATE TABLE public.schema_migrations (
  migration_id TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  checksum TEXT NOT NULL,
  source_set TEXT NOT NULL
);
INSERT INTO public.schema_migrations (migration_id, checksum, source_set) VALUES
  ('docker/000_synthetic.sql', 'sha256:synthetic-control', 'docker');

CREATE TYPE cost_aggregation_window AS ENUM ('24h', '7d', '30d');
CREATE TABLE public.llm_cost_aggregates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  aggregation_key VARCHAR(512) NOT NULL,
  "window" cost_aggregation_window NOT NULL,
  total_cost_usd NUMERIC(14, 6) NOT NULL DEFAULT 0,
  total_tokens BIGINT NOT NULL DEFAULT 0,
  call_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE public.llm_cost_aggregates OWNER TO omninodeadmin;

CREATE TABLE public.flat_node_parity_control (
  id UUID PRIMARY KEY,
  payload JSONB NOT NULL
);
ALTER TABLE public.flat_node_parity_control OWNER TO omninodeadmin;

CREATE TABLE public.baselines_comparisons (
  id BIGSERIAL PRIMARY KEY,
  cohort TEXT NOT NULL,
  measured_at TIMESTAMPTZ NOT NULL
);
ALTER TABLE public.baselines_comparisons OWNER TO omninodeadmin;

-- Live-legacy omnidash ledger shape plus the post-OMN-15332 node ledger.
\connect omnidash_analytics
CREATE TABLE public.schema_migrations (
  filename TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO public.schema_migrations (filename, applied_at)
SELECT format('%04s_synthetic_legacy.sql', sequence),
       TIMESTAMPTZ '2026-01-01 00:00:00+00' + sequence * INTERVAL '1 minute'
FROM generate_series(0, 22) AS sequence;
ALTER TABLE public.schema_migrations OWNER TO omninodeadmin;

CREATE TABLE public.node_schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  checksum TEXT NOT NULL DEFAULT ''
);
INSERT INTO public.node_schema_migrations (version, checksum) VALUES
  ('node:synthetic:0001.sql', 'sha256:synthetic-node-control');
ALTER TABLE public.node_schema_migrations OWNER TO role_omnidash;

-- Deliberately old node-side shapes. These are empty so current reconciling
-- migrations can prove shape convergence without inventing row backfills.
CREATE TABLE public.llm_cost_aggregates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name TEXT,
  total_cost_usd NUMERIC(14, 6),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE public.llm_cost_aggregates OWNER TO omninodeadmin;

CREATE TABLE public.baselines_comparisons (
  id BIGSERIAL PRIMARY KEY
);
ALTER TABLE public.baselines_comparisons OWNER TO omninodeadmin;

CREATE TABLE public.flat_node_parity_control (
  id UUID PRIMARY KEY,
  payload JSONB NOT NULL
);
ALTER TABLE public.flat_node_parity_control OWNER TO omninodeadmin;

-- Text/varchar tenants, sentinels, FK/index/view/function dependencies, and
-- legacy ENABLE-only RLS. These rows are synthetic and intentionally invalid
-- under the target UUID/no-sentinel contract.
CREATE TABLE public.tenants_legacy (
  tenant_id VARCHAR(64) PRIMARY KEY,
  display_name TEXT NOT NULL
);
INSERT INTO public.tenants_legacy (tenant_id, display_name) VALUES
  ('legacy-acme', 'Synthetic tenant A'),
  ('omninode', 'Synthetic sentinel'),
  ('00000000-0000-0000-0000-000000000000', 'Synthetic zero UUID sentinel'),
  ('', 'Synthetic empty sentinel');
ALTER TABLE public.tenants_legacy OWNER TO omninodeadmin;

CREATE TABLE public.tenant_usage_legacy (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id VARCHAR(64) NOT NULL REFERENCES public.tenants_legacy (tenant_id),
  external_key TEXT NOT NULL,
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (tenant_id, external_key)
);
CREATE INDEX tenant_usage_legacy_tenant_idx
  ON public.tenant_usage_legacy (tenant_id);
ALTER TABLE public.tenant_usage_legacy ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_usage_legacy_policy ON public.tenant_usage_legacy
  USING (tenant_id = current_setting('app.tenant_id', true))
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
INSERT INTO public.tenant_usage_legacy (tenant_id, external_key) VALUES
  ('legacy-acme', 'usage-a'),
  ('omninode', 'usage-sentinel'),
  ('00000000-0000-0000-0000-000000000000', 'usage-zero');
ALTER TABLE public.tenant_usage_legacy OWNER TO omninodeadmin;

CREATE VIEW public.tenant_usage_legacy_view AS
SELECT id, tenant_id, external_key FROM public.tenant_usage_legacy;
ALTER VIEW public.tenant_usage_legacy_view OWNER TO omninodeadmin;

CREATE FUNCTION public.tenant_usage_legacy_count()
RETURNS BIGINT
LANGUAGE sql
SECURITY DEFINER
AS $$SELECT count(*) FROM public.tenant_usage_legacy$$;
ALTER FUNCTION public.tenant_usage_legacy_count() OWNER TO omninodeadmin;

GRANT SELECT, INSERT, UPDATE ON public.tenant_usage_legacy TO role_omnidash;
GRANT SELECT ON public.tenant_usage_legacy_view TO app_dashboard;
SET ROLE omninodeadmin;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT ON TABLES TO app_dashboard;
RESET ROLE;

-- Safe and RED controls. Every catalog detector in prove.sh is executed once
-- against each side of the pair; a detector that cannot distinguish them fails.
CREATE SCHEMA omn15422_fixture AUTHORIZATION omninodeadmin;

CREATE TABLE omn15422_fixture.mapping_positive (
  legacy_tenant_value TEXT NOT NULL,
  tenant_uuid UUID NOT NULL
);
INSERT INTO omn15422_fixture.mapping_positive VALUES
  ('legacy-a', '11111111-1111-4111-8111-111111111111'),
  ('legacy-b', '22222222-2222-4222-8222-222222222222');

CREATE TABLE omn15422_fixture.mapping_red (
  legacy_tenant_value TEXT NOT NULL,
  tenant_uuid UUID NOT NULL
);
INSERT INTO omn15422_fixture.mapping_red VALUES
  ('ambiguous', '33333333-3333-4333-8333-333333333333'),
  ('ambiguous', '44444444-4444-4444-8444-444444444444');

CREATE TABLE omn15422_fixture.checksum_positive (
  migration_id TEXT NOT NULL,
  checksum TEXT NOT NULL
);
INSERT INTO omn15422_fixture.checksum_positive VALUES
  ('stream:0001', 'sha256:positive');

CREATE TABLE omn15422_fixture.checksum_red (
  migration_id TEXT NOT NULL,
  checksum TEXT NOT NULL
);
INSERT INTO omn15422_fixture.checksum_red VALUES
  ('stream:0001', 'sha256:first'),
  ('stream:0001', 'sha256:conflict');

CREATE TABLE omn15422_fixture.owner_positive (id INTEGER PRIMARY KEY);
ALTER TABLE omn15422_fixture.owner_positive OWNER TO role_omnidash;
CREATE TABLE omn15422_fixture.owner_red (id INTEGER PRIMARY KEY);
ALTER TABLE omn15422_fixture.owner_red OWNER TO omninodeadmin;

CREATE TABLE omn15422_fixture.tenant_usage_safe (
  id UUID PRIMARY KEY,
  tenant_id UUID NOT NULL
);
ALTER TABLE omn15422_fixture.tenant_usage_safe ENABLE ROW LEVEL SECURITY;
ALTER TABLE omn15422_fixture.tenant_usage_safe FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_usage_safe_policy
  ON omn15422_fixture.tenant_usage_safe
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

CREATE VIEW omn15422_fixture.tenant_usage_safe_view
WITH (security_invoker = true) AS
SELECT id, tenant_id FROM omn15422_fixture.tenant_usage_safe;
CREATE VIEW omn15422_fixture.tenant_usage_red_view AS
SELECT id, tenant_id FROM omn15422_fixture.tenant_usage_safe;

CREATE FUNCTION omn15422_fixture.tenant_usage_safe_count()
RETURNS BIGINT
LANGUAGE sql
SECURITY INVOKER
SET search_path = pg_catalog, omn15422_fixture
AS $$SELECT count(*) FROM tenant_usage_safe$$;
REVOKE ALL ON FUNCTION omn15422_fixture.tenant_usage_safe_count() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION omn15422_fixture.tenant_usage_safe_count() TO app_dashboard;

CREATE FUNCTION omn15422_fixture.tenant_usage_red_count()
RETURNS BIGINT
LANGUAGE sql
SECURITY DEFINER
AS $$SELECT count(*) FROM omn15422_fixture.tenant_usage_safe$$;

CREATE TABLE omn15422_fixture.transform_positive (
  source_id INTEGER PRIMARY KEY,
  target_key TEXT NOT NULL
);
INSERT INTO omn15422_fixture.transform_positive VALUES (1, 'target-a'), (2, 'target-b');
CREATE TABLE omn15422_fixture.transform_red (
  source_id INTEGER PRIMARY KEY,
  target_key TEXT NOT NULL
);
INSERT INTO omn15422_fixture.transform_red VALUES (1, 'collision'), (2, 'collision');

-- Third legacy ledger shape: version key plus nullable checksum.
\connect omninode_cloud
CREATE TABLE public.schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  checksum TEXT
);
INSERT INTO public.schema_migrations (version, checksum) VALUES
  ('legacy-cloud-0001', NULL);
ALTER TABLE public.schema_migrations OWNER TO omninodeadmin;
