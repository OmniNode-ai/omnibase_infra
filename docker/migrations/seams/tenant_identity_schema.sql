-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- The tenant identity and gateway schema on a fresh omninode_cloud (OMN-26121, T14).
--
-- Applied by scripts/run-forward-migrations.sh as the lane's superuser
-- (psql -X -q -v ON_ERROR_STOP=1 -f this file), only where the rendered
-- forward-migration service carries ONEX_MIGRATION_GATEWAY_LANE (dev and
-- the stability lane). It lives OUTSIDE docker/migrations/forward/ for the same reason
-- gateway_runtime_grants.sql does: it is not an ordered, checksummed migration.
--
-- WHY IT EXISTS: the one-shot that applied the cloud corpus to a dev or stability
-- volume is gone, so a fresh volume holds an empty omninode_cloud and the gateway's
-- grants seam skips ("the seven gateway tables do not all exist"). Every authenticated
-- submit then answers 503. This file creates the eight tables the gateway and the
-- tenant identity writer read and write, with the columns, checks and indexes the warm
-- dev volume holds today (read from its catalog, not from memory).
--
-- THE CREATE STEP NEVER ALTERS A TABLE. Each table is created, with its own indexes,
-- only when it is absent at the start of the run. A table that already exists is left
-- exactly as it is, including a drifted one: tenant_identity_schema_readback.sql
-- compares the final catalog with the declared one and fails the run on any difference.
-- tenant_api_keys.expires_at is NOT here: tenant_api_keys_expiry.sql is the only place
-- that adds it, so fresh and warm volumes end identical.
--
-- Ownership: the tables are owned by the connecting superuser. Grants to the gateway
-- and writer roles belong to their own seams; this file issues none.

SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'omninode_cloud') AS db_present
\gset

\if :db_present
  \connect omninode_cloud

  SELECT
      to_regclass('public.tenants') IS NULL AS new_tenants,
      to_regclass('public.plan_tiers') IS NULL AS new_plan_tiers,
      to_regclass('public.plan_entitlements') IS NULL AS new_plan_entitlements,
      to_regclass('public.tenant_plans') IS NULL AS new_tenant_plans,
      to_regclass('public.tenant_api_keys') IS NULL AS new_tenant_api_keys,
      to_regclass('public.tenant_event_outbox') IS NULL AS new_tenant_event_outbox,
      to_regclass('public.gateway_workflows') IS NULL AS new_gateway_workflows,
      to_regclass('public.gateway_rate_limit_events') IS NULL AS new_gateway_rate_limit_events
  \gset

  BEGIN;

  \if :new_tenants
    CREATE TABLE public.tenants (
        tenant_id uuid DEFAULT gen_random_uuid() NOT NULL,
        tenant_slug text NOT NULL,
        slug text,
        name text,
        created_by_sub text,
        provisioning_state text DEFAULT 'active'::text NOT NULL,
        kc_patch_attempt_count integer DEFAULT 0 NOT NULL,
        kc_patch_last_error text,
        owner_email text,
        principal_id text,
        stripe_customer_id text,
        acl_state text DEFAULT 'PENDING_MSK_RECONCILIATION'::text NOT NULL,
        created_at timestamp with time zone DEFAULT now() NOT NULL,
        quota_state text DEFAULT 'ACTIVE'::text NOT NULL,
        CONSTRAINT tenants_pkey PRIMARY KEY (tenant_id),
        CONSTRAINT tenants_tenant_slug_key UNIQUE (tenant_slug),
        CONSTRAINT tenants_acl_state_check CHECK (acl_state = ANY (ARRAY['ACTIVE'::text, 'DEFERRED_PENDING_MSK_ACLS'::text, 'PENDING_MSK_RECONCILIATION'::text])),
        CONSTRAINT tenants_kc_patch_attempt_count_check CHECK (kc_patch_attempt_count >= 0),
        CONSTRAINT tenants_provisioning_state_check CHECK (provisioning_state = ANY (ARRAY['pending'::text, 'active'::text, 'suspended'::text, 'failed'::text, 'NEEDS_KC_PATCH'::text])),
        CONSTRAINT tenants_quota_state_check CHECK (quota_state = ANY (ARRAY['ACTIVE'::text, 'DEFERRED_PENDING_MSK_QUOTAS'::text]))
    );
    -- The warm volume carries this second, identical UNIQUE constraint beside tenants_tenant_slug_key.
    -- CREATE TABLE merges duplicate constraints into one, so it is added after, on the table this
    -- run just created (the readback declares both, and a warm volume holds both).
    ALTER TABLE public.tenants ADD CONSTRAINT tenants_tenant_slug_unique UNIQUE (tenant_slug);
    CREATE UNIQUE INDEX idx_tenants_created_by_sub_unique ON public.tenants USING btree (created_by_sub) WHERE (created_by_sub IS NOT NULL);
    CREATE INDEX idx_tenants_owner_email ON public.tenants USING btree (owner_email) WHERE (owner_email IS NOT NULL);
    CREATE UNIQUE INDEX idx_tenants_principal_id ON public.tenants USING btree (principal_id) WHERE (principal_id IS NOT NULL);
    CREATE INDEX idx_tenants_provisioning_state_non_active ON public.tenants USING btree (provisioning_state) WHERE (provisioning_state <> 'active'::text);
    CREATE UNIQUE INDEX idx_tenants_stripe_customer_id ON public.tenants USING btree (stripe_customer_id) WHERE (stripe_customer_id IS NOT NULL);
  \endif

  \if :new_plan_tiers
    CREATE TABLE public.plan_tiers (
        id uuid DEFAULT gen_random_uuid() NOT NULL,
        code text NOT NULL,
        name text NOT NULL,
        description text,
        is_default boolean DEFAULT false NOT NULL,
        created_at timestamp with time zone DEFAULT now() NOT NULL,
        stripe_price_id text,
        CONSTRAINT plan_tiers_pkey PRIMARY KEY (id),
        CONSTRAINT plan_tiers_code_key UNIQUE (code)
    );
    CREATE INDEX idx_plan_tiers_stripe_price_id ON public.plan_tiers USING btree (stripe_price_id) WHERE (stripe_price_id IS NOT NULL);
    -- The default plan, seeded only into a table this run just created and found empty.
    INSERT INTO public.plan_tiers (code, name, description, is_default)
    SELECT 'discovery', 'Discovery', 'Free plan', true
    WHERE NOT EXISTS (SELECT 1 FROM public.plan_tiers);
  \endif

  \if :new_plan_entitlements
    CREATE TABLE public.plan_entitlements (
        plan_code text NOT NULL,
        metric_key text NOT NULL,
        window_key text NOT NULL,
        limit_value bigint NOT NULL,
        hard_cap boolean DEFAULT true NOT NULL,
        unit text,
        created_at timestamp with time zone DEFAULT now() NOT NULL,
        updated_at timestamp with time zone DEFAULT now() NOT NULL,
        CONSTRAINT plan_entitlements_pkey PRIMARY KEY (plan_code, metric_key, window_key),
        CONSTRAINT plan_entitlements_plan_code_fkey FOREIGN KEY (plan_code) REFERENCES public.plan_tiers(code) ON DELETE CASCADE
    );
  \endif

  \if :new_tenant_plans
    CREATE TABLE public.tenant_plans (
        tenant_id uuid NOT NULL,
        plan_code text NOT NULL,
        is_active boolean DEFAULT true NOT NULL,
        started_at timestamp with time zone DEFAULT now() NOT NULL,
        ended_at timestamp with time zone,
        CONSTRAINT tenant_plans_pkey PRIMARY KEY (tenant_id, plan_code, started_at)
    );
    CREATE INDEX idx_tenant_plans_active_lookup ON public.tenant_plans USING btree (tenant_id, started_at DESC) WHERE ((is_active = true) AND (ended_at IS NULL));
    CREATE UNIQUE INDEX idx_tenant_plans_one_active_per_tenant ON public.tenant_plans USING btree (tenant_id) WHERE ((is_active = true) AND (ended_at IS NULL));
    CREATE UNIQUE INDEX uniq_tenant_plans_active ON public.tenant_plans USING btree (tenant_id) WHERE (is_active = true);
  \endif

  \if :new_tenant_api_keys
    CREATE TABLE public.tenant_api_keys (
        id uuid DEFAULT gen_random_uuid() NOT NULL,
        tenant_id uuid NOT NULL,
        name text NOT NULL,
        key_hash text NOT NULL,
        created_at timestamp with time zone DEFAULT now() NOT NULL,
        last_used_at timestamp with time zone,
        revoked_at timestamp with time zone,
        is_active boolean DEFAULT true NOT NULL,
        CONSTRAINT tenant_api_keys_pkey PRIMARY KEY (id),
        CONSTRAINT tenant_api_keys_key_hash_key UNIQUE (key_hash),
        CONSTRAINT chk_key_hash_valid CHECK (length(TRIM(BOTH FROM key_hash)) >= 32),
        CONSTRAINT chk_name_valid CHECK ((length(TRIM(BOTH FROM name)) > 0) AND (length(name) <= 255)),
        CONSTRAINT chk_revocation_consistency CHECK (((is_active = true) AND (revoked_at IS NULL)) OR ((is_active = false) AND (revoked_at IS NOT NULL))),
        CONSTRAINT tenant_api_keys_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id) ON DELETE CASCADE
    );
    CREATE INDEX idx_tenant_api_keys_active ON public.tenant_api_keys USING btree (tenant_id, is_active) WHERE ((is_active = true) AND (revoked_at IS NULL));
    CREATE INDEX idx_tenant_api_keys_tenant_id ON public.tenant_api_keys USING btree (tenant_id);
    CREATE UNIQUE INDEX uq_tenant_api_keys_tenant_id_name_active ON public.tenant_api_keys USING btree (tenant_id, name) WHERE (is_active = true);
  \endif

  \if :new_tenant_event_outbox
    CREATE TABLE public.tenant_event_outbox (
        id bigserial PRIMARY KEY,
        topic text NOT NULL,
        envelope jsonb NOT NULL,
        created_at timestamp with time zone DEFAULT now() NOT NULL,
        published_at timestamp with time zone,
        attempts integer DEFAULT 0 NOT NULL,
        last_error text
    );
    CREATE INDEX idx_tenant_event_outbox_pending ON public.tenant_event_outbox USING btree (created_at) WHERE (published_at IS NULL);
  \endif

  \if :new_gateway_workflows
    CREATE TABLE public.gateway_workflows (
        workflow_id uuid NOT NULL,
        tenant_id uuid NOT NULL,
        principal_id text NOT NULL,
        authenticated_user_id text NOT NULL,
        workflow_type text NOT NULL,
        contract_id text NOT NULL,
        command_topic text NOT NULL,
        envelope_id uuid NOT NULL,
        correlation_id uuid NOT NULL,
        causation_id uuid NOT NULL,
        status text NOT NULL,
        submitted_at timestamp with time zone DEFAULT now() NOT NULL,
        updated_at timestamp with time zone DEFAULT now() NOT NULL,
        terminal_model_used text,
        terminal_total_tokens integer,
        terminal_latency_ms integer,
        completed_at timestamp with time zone,
        result_content text,
        terminal_failure_class text,
        terminal_failure_code text,
        terminal_failure_reason text,
        terminal_remediation text,
        terminal_route text,
        terminal_provider text,
        terminal_credential_source text,
        terminal_rule_evaluations jsonb,
        CONSTRAINT gateway_workflows_pkey PRIMARY KEY (workflow_id),
        CONSTRAINT gateway_workflows_status_check CHECK (status = ANY (ARRAY['accepted'::text, 'published'::text, 'failed_publish'::text, 'completed'::text, 'failed'::text])),
        CONSTRAINT gateway_workflows_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id) ON DELETE CASCADE
    );
    CREATE INDEX idx_gateway_workflows_correlation ON public.gateway_workflows USING btree (correlation_id);
    CREATE INDEX idx_gateway_workflows_tenant_submitted ON public.gateway_workflows USING btree (tenant_id, submitted_at DESC);
  \endif

  \if :new_gateway_rate_limit_events
    CREATE TABLE public.gateway_rate_limit_events (
        event_id uuid NOT NULL,
        tenant_id uuid NOT NULL,
        principal_id text NOT NULL,
        workflow_type text NOT NULL,
        correlation_id uuid NOT NULL,
        limit_per_minute integer NOT NULL,
        burst_capacity integer NOT NULL,
        retry_after_seconds integer NOT NULL,
        occurred_at timestamp with time zone DEFAULT now() NOT NULL,
        CONSTRAINT gateway_rate_limit_events_pkey PRIMARY KEY (event_id),
        CONSTRAINT gateway_rate_limit_events_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id) ON DELETE CASCADE
    );
    CREATE INDEX idx_gateway_rate_limit_events_tenant_time ON public.gateway_rate_limit_events USING btree (tenant_id, occurred_at DESC);
  \endif

  COMMIT;

  \echo [forward-migration]   ok    tenant identity schema created where absent (tables present are left as they are; the readback compares them)
\else
  \echo [forward-migration]   skip  tenant identity schema (database omninode_cloud absent on this lane)
\endif
