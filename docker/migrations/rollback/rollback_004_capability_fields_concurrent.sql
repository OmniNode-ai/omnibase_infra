-- Rollback: forward/004_capability_fields_concurrent.sql
-- Purpose: Drop concurrent capability indexes from registration_projections
--
-- =============================================================================
-- IMPORTANT: TRANSACTION REQUIREMENTS
-- =============================================================================
--
-- DROP INDEX CONCURRENTLY cannot run inside a transaction block.
-- This script MUST be run with autocommit enabled (same as the forward migration).
--
-- If using psql:
--   \set AUTOCOMMIT on
--   \i rollback_004_capability_fields_concurrent.sql
--
-- If using psql with -f flag:
--   psql -h <host> -d <database> -f rollback_004_capability_fields_concurrent.sql
--   (autocommit is enabled by default)
--
-- DO NOT USE:
--   BEGIN;
--   \i rollback_004_capability_fields_concurrent.sql
--   COMMIT;
--   -- This will ERROR: "DROP INDEX CONCURRENTLY cannot run inside a transaction block"
--
-- If running inside a transaction, use DROP INDEX (without CONCURRENTLY) instead.
--
-- =============================================================================
-- ROLLBACK DEPENDENCY NOTE
-- =============================================================================
--
-- This rollback drops the SAME indexes that forward/003_capability_fields.sql
-- creates (standard) and forward/004_capability_fields_concurrent.sql creates
-- (concurrent). After running this rollback:
--
--   - The capability COLUMNS remain (they are created by 003, not 004)
--   - The capability INDEXES are removed
--
-- If you intend to keep standard (non-concurrent) indexes after reverting 004,
-- you must re-run forward/003_capability_fields.sql to recreate them as standard
-- indexes (the column creation is idempotent via IF NOT EXISTS / IF EXISTS).
--
-- Typical rollback sequences:
--   - Rolling back ONLY 004: Run this, then re-run 003 to restore standard indexes
--   - Rolling back BOTH 003 + 004: Run this first, then rollback_003
-- =============================================================================

DROP INDEX CONCURRENTLY IF EXISTS idx_registration_capability_tags;
DROP INDEX CONCURRENTLY IF EXISTS idx_registration_intent_types;
DROP INDEX CONCURRENTLY IF EXISTS idx_registration_protocols;
DROP INDEX CONCURRENTLY IF EXISTS idx_registration_contract_type_state;
