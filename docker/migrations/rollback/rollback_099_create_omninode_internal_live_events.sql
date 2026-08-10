-- =============================================================================
-- ROLLBACK: 099_create_omninode_internal_live_events.sql
-- =============================================================================
-- Ticket: OMN-15359
--
-- SCOPE
--   Drops omninode_internal.live_events. RESTRICT (the default DROP TABLE
--   behavior with no CASCADE): fails closed instead of silently taking a
--   dependent object down with it, if anything has come to depend on this
--   table since 099 applied. public.live_events is NEVER touched by this
--   rollback -- 099 only ever copies into omninode_internal.live_events, so
--   the source of truth survives regardless of whether this rollback runs.
--
--   Also reverts the grant-gap repair (099 step 4/5): the ALTER DEFAULT
--   PRIVILEGES and the schema USAGE grant, so a re-run of 099 starts from a
--   clean slate. Deliberately does NOT drop the omninode_runtime role or
--   revoke its CONNECT/DATABASE grant -- same rationale rollback_096 and
--   rollback_094 already state for role_omnidash/app_dashboard: the role may
--   predate 099 (out-of-band provisioning on a managed instance) or be
--   needed by another already-cutover table, so this rollback only undoes
--   what 099 itself added.
--
-- WARNING BEFORE RUNNING THIS
--   The live runtime write path (handler_wiring._resolve_projection_database_target)
--   issues `INSERT INTO omninode_internal.live_events` unconditionally --
--   rolling this back re-introduces the original UndefinedTable failure this
--   migration exists to close. Emergency surface, not a maintenance one.
--
-- IDEMPOTENCY
--   DROP TABLE IF EXISTS is safe to re-run and safe on a lane where 099 never
--   applied. The REVOKE/ALTER DEFAULT PRIVILEGES statements are guarded on
--   the role existing so this file is also safe on a lane where the role was
--   never created.
--
-- MANUAL EXECUTION ONLY
--   This file lives under rollback/ and is never auto-applied by the forward
--   runner or docker-entrypoint-initdb.d.
-- =============================================================================

\connect omnidash_analytics

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'omninode_runtime') THEN
    EXECUTE
      'ALTER DEFAULT PRIVILEGES IN SCHEMA omninode_internal '
      'REVOKE SELECT, INSERT, UPDATE ON TABLES FROM omninode_runtime';
    EXECUTE
      'REVOKE SELECT, INSERT, UPDATE ON omninode_internal.live_events '
      'FROM omninode_runtime';
    EXECUTE 'REVOKE USAGE ON SCHEMA omninode_internal FROM omninode_runtime';
  END IF;
END;
$$;

DROP TABLE IF EXISTS omninode_internal.live_events;
