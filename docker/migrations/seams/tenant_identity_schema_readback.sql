-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Readback of the tenant identity schema on omninode_cloud (OMN-26121, T14).
--
-- Applied by scripts/run-forward-migrations.sh after tenant_identity_schema.sql and
-- tenant_api_keys_expiry.sql, only where the rendered forward-migration service carries
-- ONEX_MIGRATION_GATEWAY_LANE. It compares, for each of the eight tables, every column
-- (name, data_type, is_nullable, column_default), every constraint (pg_get_constraintdef)
-- and every index (pg_get_indexdef, as pg_indexes.indexdef) with the final declared set,
-- tenant_api_keys.expires_at and its index included, and raises on any difference in
-- either direction. It only reads: a warm volume that drifted fails the run loudly and
-- is never altered.
--
-- The declared set below was read from the warm dev lane's omninode_cloud catalog
-- (omni-201, 2026-11-22), in the form each line is rendered here:
--   <table>|column|<name>|<data_type>|<is_nullable>|<column_default>
--   <table>|constraint|<name>|<pg_get_constraintdef>
--   <table>|index|<name>|<indexdef>

SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'omninode_cloud') AS db_present
\gset

\if :db_present
  \connect omninode_cloud

  DO $readback$
  DECLARE
    declared text[] := string_to_array($declared$
      gateway_rate_limit_events|column|burst_capacity|integer|NO|
      gateway_rate_limit_events|column|correlation_id|uuid|NO|
      gateway_rate_limit_events|column|event_id|uuid|NO|
      gateway_rate_limit_events|column|limit_per_minute|integer|NO|
      gateway_rate_limit_events|column|occurred_at|timestamp with time zone|NO|now()
      gateway_rate_limit_events|column|principal_id|text|NO|
      gateway_rate_limit_events|column|retry_after_seconds|integer|NO|
      gateway_rate_limit_events|column|tenant_id|uuid|NO|
      gateway_rate_limit_events|column|workflow_type|text|NO|
      gateway_rate_limit_events|constraint|gateway_rate_limit_events_pkey|PRIMARY KEY (event_id)
      gateway_rate_limit_events|constraint|gateway_rate_limit_events_tenant_id_fkey|FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
      gateway_rate_limit_events|index|gateway_rate_limit_events_pkey|CREATE UNIQUE INDEX gateway_rate_limit_events_pkey ON public.gateway_rate_limit_events USING btree (event_id)
      gateway_rate_limit_events|index|idx_gateway_rate_limit_events_tenant_time|CREATE INDEX idx_gateway_rate_limit_events_tenant_time ON public.gateway_rate_limit_events USING btree (tenant_id, occurred_at DESC)
      gateway_workflows|column|authenticated_user_id|text|NO|
      gateway_workflows|column|causation_id|uuid|NO|
      gateway_workflows|column|command_topic|text|NO|
      gateway_workflows|column|completed_at|timestamp with time zone|YES|
      gateway_workflows|column|contract_id|text|NO|
      gateway_workflows|column|correlation_id|uuid|NO|
      gateway_workflows|column|envelope_id|uuid|NO|
      gateway_workflows|column|principal_id|text|NO|
      gateway_workflows|column|result_content|text|YES|
      gateway_workflows|column|status|text|NO|
      gateway_workflows|column|submitted_at|timestamp with time zone|NO|now()
      gateway_workflows|column|tenant_id|uuid|NO|
      gateway_workflows|column|terminal_credential_source|text|YES|
      gateway_workflows|column|terminal_failure_class|text|YES|
      gateway_workflows|column|terminal_failure_code|text|YES|
      gateway_workflows|column|terminal_failure_reason|text|YES|
      gateway_workflows|column|terminal_latency_ms|integer|YES|
      gateway_workflows|column|terminal_model_used|text|YES|
      gateway_workflows|column|terminal_provider|text|YES|
      gateway_workflows|column|terminal_remediation|text|YES|
      gateway_workflows|column|terminal_route|text|YES|
      gateway_workflows|column|terminal_rule_evaluations|jsonb|YES|
      gateway_workflows|column|terminal_total_tokens|integer|YES|
      gateway_workflows|column|updated_at|timestamp with time zone|NO|now()
      gateway_workflows|column|workflow_id|uuid|NO|
      gateway_workflows|column|workflow_type|text|NO|
      gateway_workflows|constraint|gateway_workflows_pkey|PRIMARY KEY (workflow_id)
      gateway_workflows|constraint|gateway_workflows_status_check|CHECK ((status = ANY (ARRAY['accepted'::text, 'published'::text, 'failed_publish'::text, 'completed'::text, 'failed'::text])))
      gateway_workflows|constraint|gateway_workflows_tenant_id_fkey|FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
      gateway_workflows|index|gateway_workflows_pkey|CREATE UNIQUE INDEX gateway_workflows_pkey ON public.gateway_workflows USING btree (workflow_id)
      gateway_workflows|index|idx_gateway_workflows_correlation|CREATE INDEX idx_gateway_workflows_correlation ON public.gateway_workflows USING btree (correlation_id)
      gateway_workflows|index|idx_gateway_workflows_tenant_submitted|CREATE INDEX idx_gateway_workflows_tenant_submitted ON public.gateway_workflows USING btree (tenant_id, submitted_at DESC)
      plan_entitlements|column|created_at|timestamp with time zone|NO|now()
      plan_entitlements|column|hard_cap|boolean|NO|true
      plan_entitlements|column|limit_value|bigint|NO|
      plan_entitlements|column|metric_key|text|NO|
      plan_entitlements|column|plan_code|text|NO|
      plan_entitlements|column|unit|text|YES|
      plan_entitlements|column|updated_at|timestamp with time zone|NO|now()
      plan_entitlements|column|window_key|text|NO|
      plan_entitlements|constraint|plan_entitlements_pkey|PRIMARY KEY (plan_code, metric_key, window_key)
      plan_entitlements|constraint|plan_entitlements_plan_code_fkey|FOREIGN KEY (plan_code) REFERENCES plan_tiers(code) ON DELETE CASCADE
      plan_entitlements|index|plan_entitlements_pkey|CREATE UNIQUE INDEX plan_entitlements_pkey ON public.plan_entitlements USING btree (plan_code, metric_key, window_key)
      plan_tiers|column|code|text|NO|
      plan_tiers|column|created_at|timestamp with time zone|NO|now()
      plan_tiers|column|description|text|YES|
      plan_tiers|column|id|uuid|NO|gen_random_uuid()
      plan_tiers|column|is_default|boolean|NO|false
      plan_tiers|column|name|text|NO|
      plan_tiers|column|stripe_price_id|text|YES|
      plan_tiers|constraint|plan_tiers_code_key|UNIQUE (code)
      plan_tiers|constraint|plan_tiers_pkey|PRIMARY KEY (id)
      plan_tiers|index|idx_plan_tiers_stripe_price_id|CREATE INDEX idx_plan_tiers_stripe_price_id ON public.plan_tiers USING btree (stripe_price_id) WHERE (stripe_price_id IS NOT NULL)
      plan_tiers|index|plan_tiers_code_key|CREATE UNIQUE INDEX plan_tiers_code_key ON public.plan_tiers USING btree (code)
      plan_tiers|index|plan_tiers_pkey|CREATE UNIQUE INDEX plan_tiers_pkey ON public.plan_tiers USING btree (id)
      tenant_api_keys|column|created_at|timestamp with time zone|NO|now()
      tenant_api_keys|column|expires_at|timestamp with time zone|YES|
      tenant_api_keys|column|id|uuid|NO|gen_random_uuid()
      tenant_api_keys|column|is_active|boolean|NO|true
      tenant_api_keys|column|key_hash|text|NO|
      tenant_api_keys|column|last_used_at|timestamp with time zone|YES|
      tenant_api_keys|column|name|text|NO|
      tenant_api_keys|column|revoked_at|timestamp with time zone|YES|
      tenant_api_keys|column|tenant_id|uuid|NO|
      tenant_api_keys|constraint|chk_key_hash_valid|CHECK ((length(TRIM(BOTH FROM key_hash)) >= 32))
      tenant_api_keys|constraint|chk_name_valid|CHECK (((length(TRIM(BOTH FROM name)) > 0) AND (length(name) <= 255)))
      tenant_api_keys|constraint|chk_revocation_consistency|CHECK ((((is_active = true) AND (revoked_at IS NULL)) OR ((is_active = false) AND (revoked_at IS NOT NULL))))
      tenant_api_keys|constraint|tenant_api_keys_key_hash_key|UNIQUE (key_hash)
      tenant_api_keys|constraint|tenant_api_keys_pkey|PRIMARY KEY (id)
      tenant_api_keys|constraint|tenant_api_keys_tenant_id_fkey|FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
      tenant_api_keys|index|idx_tenant_api_keys_active|CREATE INDEX idx_tenant_api_keys_active ON public.tenant_api_keys USING btree (tenant_id, is_active) WHERE ((is_active = true) AND (revoked_at IS NULL))
      tenant_api_keys|index|idx_tenant_api_keys_expires_at|CREATE INDEX idx_tenant_api_keys_expires_at ON public.tenant_api_keys USING btree (expires_at) WHERE is_active
      tenant_api_keys|index|idx_tenant_api_keys_tenant_id|CREATE INDEX idx_tenant_api_keys_tenant_id ON public.tenant_api_keys USING btree (tenant_id)
      tenant_api_keys|index|tenant_api_keys_key_hash_key|CREATE UNIQUE INDEX tenant_api_keys_key_hash_key ON public.tenant_api_keys USING btree (key_hash)
      tenant_api_keys|index|tenant_api_keys_pkey|CREATE UNIQUE INDEX tenant_api_keys_pkey ON public.tenant_api_keys USING btree (id)
      tenant_api_keys|index|uq_tenant_api_keys_tenant_id_name_active|CREATE UNIQUE INDEX uq_tenant_api_keys_tenant_id_name_active ON public.tenant_api_keys USING btree (tenant_id, name) WHERE (is_active = true)
      tenant_event_outbox|column|attempts|integer|NO|0
      tenant_event_outbox|column|created_at|timestamp with time zone|NO|now()
      tenant_event_outbox|column|envelope|jsonb|NO|
      tenant_event_outbox|column|id|bigint|NO|nextval('tenant_event_outbox_id_seq'::regclass)
      tenant_event_outbox|column|last_error|text|YES|
      tenant_event_outbox|column|published_at|timestamp with time zone|YES|
      tenant_event_outbox|column|topic|text|NO|
      tenant_event_outbox|constraint|tenant_event_outbox_pkey|PRIMARY KEY (id)
      tenant_event_outbox|index|idx_tenant_event_outbox_pending|CREATE INDEX idx_tenant_event_outbox_pending ON public.tenant_event_outbox USING btree (created_at) WHERE (published_at IS NULL)
      tenant_event_outbox|index|tenant_event_outbox_pkey|CREATE UNIQUE INDEX tenant_event_outbox_pkey ON public.tenant_event_outbox USING btree (id)
      tenant_plans|column|ended_at|timestamp with time zone|YES|
      tenant_plans|column|is_active|boolean|NO|true
      tenant_plans|column|plan_code|text|NO|
      tenant_plans|column|started_at|timestamp with time zone|NO|now()
      tenant_plans|column|tenant_id|uuid|NO|
      tenant_plans|constraint|tenant_plans_pkey|PRIMARY KEY (tenant_id, plan_code, started_at)
      tenant_plans|index|idx_tenant_plans_active_lookup|CREATE INDEX idx_tenant_plans_active_lookup ON public.tenant_plans USING btree (tenant_id, started_at DESC) WHERE ((is_active = true) AND (ended_at IS NULL))
      tenant_plans|index|idx_tenant_plans_one_active_per_tenant|CREATE UNIQUE INDEX idx_tenant_plans_one_active_per_tenant ON public.tenant_plans USING btree (tenant_id) WHERE ((is_active = true) AND (ended_at IS NULL))
      tenant_plans|index|tenant_plans_pkey|CREATE UNIQUE INDEX tenant_plans_pkey ON public.tenant_plans USING btree (tenant_id, plan_code, started_at)
      tenant_plans|index|uniq_tenant_plans_active|CREATE UNIQUE INDEX uniq_tenant_plans_active ON public.tenant_plans USING btree (tenant_id) WHERE (is_active = true)
      tenants|column|acl_state|text|NO|'PENDING_MSK_RECONCILIATION'::text
      tenants|column|created_at|timestamp with time zone|NO|now()
      tenants|column|created_by_sub|text|YES|
      tenants|column|kc_patch_attempt_count|integer|NO|0
      tenants|column|kc_patch_last_error|text|YES|
      tenants|column|name|text|YES|
      tenants|column|owner_email|text|YES|
      tenants|column|principal_id|text|YES|
      tenants|column|provisioning_state|text|NO|'active'::text
      tenants|column|quota_state|text|NO|'ACTIVE'::text
      tenants|column|slug|text|YES|
      tenants|column|stripe_customer_id|text|YES|
      tenants|column|tenant_id|uuid|NO|gen_random_uuid()
      tenants|column|tenant_slug|text|NO|
      tenants|constraint|tenants_acl_state_check|CHECK ((acl_state = ANY (ARRAY['ACTIVE'::text, 'DEFERRED_PENDING_MSK_ACLS'::text, 'PENDING_MSK_RECONCILIATION'::text])))
      tenants|constraint|tenants_kc_patch_attempt_count_check|CHECK ((kc_patch_attempt_count >= 0))
      tenants|constraint|tenants_pkey|PRIMARY KEY (tenant_id)
      tenants|constraint|tenants_provisioning_state_check|CHECK ((provisioning_state = ANY (ARRAY['pending'::text, 'active'::text, 'suspended'::text, 'failed'::text, 'NEEDS_KC_PATCH'::text])))
      tenants|constraint|tenants_quota_state_check|CHECK ((quota_state = ANY (ARRAY['ACTIVE'::text, 'DEFERRED_PENDING_MSK_QUOTAS'::text])))
      tenants|constraint|tenants_tenant_slug_key|UNIQUE (tenant_slug)
      tenants|constraint|tenants_tenant_slug_unique|UNIQUE (tenant_slug)
      tenants|index|idx_tenants_created_by_sub_unique|CREATE UNIQUE INDEX idx_tenants_created_by_sub_unique ON public.tenants USING btree (created_by_sub) WHERE (created_by_sub IS NOT NULL)
      tenants|index|idx_tenants_owner_email|CREATE INDEX idx_tenants_owner_email ON public.tenants USING btree (owner_email) WHERE (owner_email IS NOT NULL)
      tenants|index|idx_tenants_principal_id|CREATE UNIQUE INDEX idx_tenants_principal_id ON public.tenants USING btree (principal_id) WHERE (principal_id IS NOT NULL)
      tenants|index|idx_tenants_provisioning_state_non_active|CREATE INDEX idx_tenants_provisioning_state_non_active ON public.tenants USING btree (provisioning_state) WHERE (provisioning_state <> 'active'::text)
      tenants|index|idx_tenants_stripe_customer_id|CREATE UNIQUE INDEX idx_tenants_stripe_customer_id ON public.tenants USING btree (stripe_customer_id) WHERE (stripe_customer_id IS NOT NULL)
      tenants|index|tenants_pkey|CREATE UNIQUE INDEX tenants_pkey ON public.tenants USING btree (tenant_id)
      tenants|index|tenants_tenant_slug_key|CREATE UNIQUE INDEX tenants_tenant_slug_key ON public.tenants USING btree (tenant_slug)
      tenants|index|tenants_tenant_slug_unique|CREATE UNIQUE INDEX tenants_tenant_slug_unique ON public.tenants USING btree (tenant_slug)
    $declared$, E'\n');
    actual text[];
    missing text;
    extra text;
  BEGIN
    SELECT array_agg(line) INTO actual FROM (
      WITH t(name) AS (VALUES ('tenants'), ('plan_tiers'), ('plan_entitlements'), ('tenant_plans'),
                              ('tenant_api_keys'), ('tenant_event_outbox'), ('gateway_workflows'),
                              ('gateway_rate_limit_events'))
      SELECT c.table_name || '|column|' || c.column_name || '|' || c.data_type || '|' || c.is_nullable
             || '|' || coalesce(c.column_default, '') AS line
        FROM information_schema.columns c JOIN t ON t.name = c.table_name
       WHERE c.table_schema = 'public'
      UNION ALL
      SELECT cl.relname || '|constraint|' || co.conname || '|' || regexp_replace(pg_get_constraintdef(co.oid), 'btrim\(([a-z_]+)\)', 'TRIM(BOTH FROM \1)', 'g')
        FROM pg_catalog.pg_constraint co
        JOIN pg_catalog.pg_class cl ON cl.oid = co.conrelid
        JOIN t ON t.name = cl.relname
       WHERE cl.relnamespace = 'public'::regnamespace
      UNION ALL
      SELECT i.tablename || '|index|' || i.indexname || '|' || i.indexdef
        FROM pg_catalog.pg_indexes i JOIN t ON t.name = i.tablename
       WHERE i.schemaname = 'public'
    ) s;

    SELECT string_agg(d, E'\n' ORDER BY d) INTO missing
      FROM (SELECT btrim(x) AS d FROM unnest(declared) x) dd
     WHERE d <> '' AND d <> ALL (coalesce(actual, ARRAY[]::text[]));
    SELECT string_agg(a, E'\n' ORDER BY a) INTO extra
      FROM unnest(coalesce(actual, ARRAY[]::text[])) a
     WHERE a <> ALL (ARRAY(SELECT btrim(x) FROM unnest(declared) x WHERE btrim(x) <> ''));

    IF missing IS NOT NULL OR extra IS NOT NULL THEN
      RAISE EXCEPTION E'tenant identity schema in omninode_cloud differs from the declared one.\nDeclared but absent or different:\n%\nPresent but not declared:\n%',
        coalesce(missing, '<none>'), coalesce(extra, '<none>');
    END IF;
  END
  $readback$;
  \echo [forward-migration]   ok    tenant identity schema matches the declared one
\else
  \echo [forward-migration]   skip  tenant identity schema readback (database omninode_cloud absent on this lane)
\endif
