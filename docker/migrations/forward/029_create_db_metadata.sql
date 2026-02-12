-- Migration: 029_create_db_metadata
-- Description: Create db_metadata table for DB ownership markers (OMN-2085)
-- Created: 2026-02-12
--
-- Purpose: Each service database contains a singleton db_metadata row that
-- records which service owns the database. At startup, each service asserts
-- that the connected database's owner_service matches the expected service.
-- Mismatch = immediate failure (prevents cross-service data corruption).
--
-- Rollback: See rollback/rollback_029_create_db_metadata.sql

CREATE TABLE IF NOT EXISTS public.db_metadata (
    id BOOLEAN PRIMARY KEY DEFAULT TRUE,
    owner_service TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT db_metadata_single_row CHECK (id = TRUE)
);

COMMENT ON TABLE public.db_metadata IS
    'Singleton table tracking which service owns this database. '
    'Used by validate_db_ownership() at startup to prevent cross-service '
    'data corruption after the DB-per-repo split (OMN-2085).';

-- Seed the ownership row for omnibase_infra
INSERT INTO public.db_metadata (id, owner_service, schema_version)
VALUES (TRUE, 'omnibase_infra', '029')
ON CONFLICT (id) DO UPDATE SET
    schema_version = EXCLUDED.schema_version,
    updated_at = NOW()
WHERE db_metadata.owner_service = EXCLUDED.owner_service;

-- Fail hard if a different service already owns this database.
-- Preserves idempotency for same-owner re-runs while preventing
-- cross-service data corruption with an immediate migration failure.
DO $$
DECLARE
    current_owner TEXT;
BEGIN
    SELECT owner_service INTO current_owner FROM public.db_metadata WHERE id = TRUE;
    IF current_owner IS NOT NULL AND current_owner != 'omnibase_infra' THEN
        RAISE EXCEPTION 'db_metadata.owner_service is "%" but this migration expects "omnibase_infra". Possible misconfiguration.', current_owner;
    END IF;
END $$;
