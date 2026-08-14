-- =============================================================================
-- ROLLBACK: 097_grant_app_dashboard_connect_omnidash_analytics.sql
-- =============================================================================
-- Ticket: OMN-15297
--
-- SCOPE
--   Revokes exactly what 097 granted: CONNECT on omnidash_analytics, USAGE on
--   schema public, and SELECT on the tenant-isolated tables. It deliberately
--   does NOT drop app_dashboard — the role is created by 094 and dropping it
--   here would roll back a different migration (see
--   rollback_094_create_app_dashboard_role.sql for that).
--
--   It also does not touch PUBLIC's privileges. 097 never changed them, and
--   OMN-15355 owns that surface.
--
-- WARNING BEFORE RUNNING THIS
--   Any application pool connected AS app_dashboard loses its read path the
--   moment the CONNECT revoke lands — existing sessions survive (Postgres
--   checks CONNECT at session establishment), new ones fail with
--   `permission denied for database`, which is the exact symptom OMN-15297
--   exists to remove. Repoint the DSN FIRST.
--
--   Repointing that DSN back to a superuser role also restores a connection
--   that is exempt from row-level security, which makes every RLS/FORCE state
--   on this database inert again. This rollback is an emergency surface, not a
--   maintenance one.
--
-- IDEMPOTENCY
--   REVOKE on a privilege that was never granted is a no-op in Postgres, so
--   this file is safe to re-run and safe to run on a lane where 097 never
--   applied. The SELECT revoke is table-driven rather than a blanket
--   `ALL TABLES` so it cannot revoke a grant that some other migration owns.
-- =============================================================================

\connect omnidash_analytics

DO $$
DECLARE
  covered RECORD;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_dashboard') THEN
    RAISE NOTICE 'app_dashboard does not exist — nothing to revoke';
    RETURN;
  END IF;

  FOR covered IN
    SELECT c.relname
      FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
     WHERE n.nspname = 'public'
       AND c.relkind = 'r'
       AND c.relrowsecurity
       AND EXISTS (
         SELECT 1 FROM pg_policy p
          WHERE p.polrelid = c.oid AND p.polname = 'tenant_isolation'
       )
     ORDER BY c.relname
  LOOP
    EXECUTE format('REVOKE ALL ON public.%I FROM app_dashboard', covered.relname);
  END LOOP;

  EXECUTE 'REVOKE USAGE ON SCHEMA public FROM app_dashboard';
END;
$$;

\connect omnibase_infra

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_dashboard')
     AND EXISTS (SELECT 1 FROM pg_database WHERE datname = 'omnidash_analytics') THEN
    EXECUTE 'REVOKE CONNECT ON DATABASE omnidash_analytics FROM app_dashboard';
  END IF;
END;
$$;
