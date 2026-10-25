-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- OMN-21015. Apply ONLY through scripts/apply_application_database_acl.py dev-contractor-usage.
-- OPERATOR-CONSENT 2026-10-25T13:57:28Z authorizes SELECT on these dev tables.
-- Existing login, CONNECT and schema USAGE are prerequisites; no role creation,
-- default privileges, membership, RLS changes or credential changes belong here.
BEGIN;
DO $$
BEGIN
    IF current_database() <> 'omnidash_analytics'
       OR current_setting('onex.dev_contractor_grants', true) IS DISTINCT FROM 'dev' THEN
        RAISE EXCEPTION 'dev contractor grant runner and analytics database required';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = 'lakshman_ro'
        AND NOT rolsuper AND NOT rolbypassrls AND NOT rolcreaterole
        AND NOT rolcreatedb AND NOT rolreplication
    ) THEN
        RAISE EXCEPTION 'existing unprivileged lakshman_ro role required';
    END IF;
END
$$;
GRANT SELECT ON TABLE public.usage_by_model_day,
    public.usage_by_model_day_calls, public.llm_call_metrics TO lakshman_ro;
COMMIT;
