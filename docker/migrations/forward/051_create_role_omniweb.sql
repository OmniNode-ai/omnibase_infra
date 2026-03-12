-- =============================================================================
-- MIGRATION: Create role_omniweb and grant table permissions
-- =============================================================================
-- Ticket: OMN-4700 (omniweb-migrate P0 connectivity fix)
-- Version: 1.0.0
--
-- PURPOSE:
--   omniweb-migrate currently uses the postgres superuser from
--   per-service-db-credentials. This migration creates a least-privilege
--   role for omniweb and grants it INSERT, SELECT, UPDATE, DELETE on
--   the tables omniweb needs.
--
--   After this migration lands:
--   1. Add role_omniweb to per-service-db-credentials k8s secret
--      (kubectl patch — see PR description for the operator command)
--   2. Update omniweb-migrate.yaml to use role_omniweb instead of postgres
--      (done in the companion omninode_infra PR)
--
-- ROLLBACK:
--   DROP ROLE role_omniweb;  (after revoking grants)
-- =============================================================================

-- Create the role if it doesn't already exist.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'role_omniweb') THEN
    CREATE ROLE role_omniweb WITH LOGIN;
  END IF;
END;
$$;

-- Grant DML on omniweb tables in the omnibase_infra DB.
-- These are the only tables omniweb migrations touch.
GRANT INSERT, SELECT, UPDATE, DELETE
  ON TABLE public.waitlist_signups
  TO role_omniweb;

GRANT INSERT, SELECT, UPDATE, DELETE
  ON TABLE public.admin_events_log
  TO role_omniweb;

-- Grant usage on the schema so the role can resolve objects.
GRANT USAGE ON SCHEMA public TO role_omniweb;
