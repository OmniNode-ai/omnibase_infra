-- =============================================================================
-- ROLLBACK: 098_create_omninode_internal_schema.sql
-- =============================================================================
-- Ticket: OMN-15359
--
-- SCOPE
--   Drops the `omninode_internal` schema created by 098. RESTRICT, never
--   CASCADE: this rollback is safe ONLY while the schema remains empty (its
--   state as of this ticket — no table has been copied into it). If a later
--   migration has copied any relation into `omninode_internal`, RESTRICT
--   makes this statement fail closed instead of silently destroying that
--   data; do not change it to CASCADE to force the rollback through.
--
-- IDEMPOTENCY
--   DROP SCHEMA IF EXISTS is safe to re-run and safe on a lane where 098
--   never applied.
--
-- MANUAL EXECUTION ONLY
--   This file lives under rollback/ and is never auto-applied by the forward
--   runner or docker-entrypoint-initdb.d.
-- =============================================================================

\connect omnidash_analytics

DROP SCHEMA IF EXISTS omninode_internal RESTRICT;
