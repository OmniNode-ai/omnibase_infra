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
-- IDEMPOTENCY
--   DROP TABLE IF EXISTS is safe to re-run and safe on a lane where 099 never
--   applied.
--
-- MANUAL EXECUTION ONLY
--   This file lives under rollback/ and is never auto-applied by the forward
--   runner or docker-entrypoint-initdb.d.
-- =============================================================================

\connect omnidash_analytics

DROP TABLE IF EXISTS omninode_internal.live_events;
