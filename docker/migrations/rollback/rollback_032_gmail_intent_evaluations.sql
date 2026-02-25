-- Rollback: forward/032_create_gmail_intent_evaluations.sql
-- Description: Remove gmail_intent_evaluations table (OMN-2792)
-- Created: 2026-02-25
--
-- WARNING: THIS ROLLBACK WILL PERMANENTLY DELETE ALL EVALUATION RECORDS.
-- This operation is IRREVERSIBLE. All idempotency records for Gmail intent
-- evaluations will be lost — re-processing events after this rollback may
-- produce duplicate Slack posts.

-- ============================================================================
-- DROP INDEXES
-- ============================================================================

DROP INDEX IF EXISTS idx_gmail_intent_evaluations_created_at;

-- ============================================================================
-- DROP TABLE
-- ============================================================================

DROP TABLE IF EXISTS gmail_intent_evaluations;

-- ============================================================================
-- REVERT DB_METADATA
-- ============================================================================

UPDATE public.db_metadata
SET schema_version = '031', updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
