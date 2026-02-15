-- Migration: 030_add_schema_fingerprint
-- Description: Add schema fingerprint columns to db_metadata (OMN-2087)
-- Created: 2026-02-12
--
-- Purpose: Stores expected schema fingerprint for runtime validation.
-- At startup, compute live fingerprint and compare to expected.
-- Mismatch = immediate failure (prevents schema drift).
--
-- Post-migration: Run `uv run python -m omnibase_infra.runtime.util_schema_fingerprint stamp`
-- Rollback: See rollback/rollback_030_add_schema_fingerprint.sql

ALTER TABLE public.db_metadata
    ADD COLUMN IF NOT EXISTS expected_schema_fingerprint TEXT,
    ADD COLUMN IF NOT EXISTS expected_schema_fingerprint_algo VARCHAR(16) DEFAULT 'sha256',
    ADD COLUMN IF NOT EXISTS expected_schema_fingerprint_generated_at TIMESTAMPTZ;

COMMENT ON COLUMN public.db_metadata.expected_schema_fingerprint IS
    'SHA256 hex digest of canonical schema shape. Compared at startup by validate_schema_fingerprint() (OMN-2087).';

-- Update schema_version
UPDATE public.db_metadata
SET schema_version = '030', updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
