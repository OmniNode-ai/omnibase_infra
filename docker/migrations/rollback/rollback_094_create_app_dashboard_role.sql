-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Rollback: 094_create_app_dashboard_role (OMN-14899)
--
-- Revokes app_dashboard's grants in omnidash_analytics (added by the
-- OMN-14894 RLS migrations — DROP ROLE fails while any grant to the role
-- remains) and drops the role. Manual/operator-run, like all rollbacks.

\connect omnidash_analytics

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_dashboard') THEN
    EXECUTE 'REVOKE ALL ON ALL TABLES IN SCHEMA public FROM app_dashboard';
    EXECUTE 'REVOKE USAGE ON SCHEMA public FROM app_dashboard';
  END IF;
END;
$$;

\connect omnibase_infra

DROP ROLE IF EXISTS app_dashboard;
