-- Migration: 117_add_registration_runtime_entity_id.sql
-- Purpose: Add runtime_entity_id column to registration_projections to record
--          the runtime that vouches for a node (OMN-24495).
-- Author: ONEX Infrastructure Team
-- Date: 2026-11-14
-- Ticket: OMN-24495
--
-- Design Decisions:
--
--   1. UUID column records which runtime vouches for the node. Nothing writes
--      the column yet; it is carried through the projection model, reader,
--      and projector contract so every column listing stays in sync.
--
--   2. Nullable with no default and no backfill: every existing row has no
--      runtime parent, and NULL is the correct representation of that state.
--
--   3. Idempotent via ALTER TABLE ADD COLUMN IF NOT EXISTS /
--      CREATE INDEX IF NOT EXISTS so re-running the migration is safe.
--
--   4. Partial index (WHERE runtime_entity_id IS NOT NULL) keeps the index
--      empty and cost-free until the column is populated.

-- =============================================================================
-- ALTER TABLE: registration_projections — add runtime parent column
-- =============================================================================

ALTER TABLE public.registration_projections
    ADD COLUMN IF NOT EXISTS runtime_entity_id UUID;

-- =============================================================================
-- INDEXES: registration_projections runtime parent column
-- =============================================================================

-- Partial index for runtime parent lookups: only rows with a runtime parent.
CREATE INDEX IF NOT EXISTS idx_registration_runtime_entity_id
    ON public.registration_projections (runtime_entity_id)
    WHERE runtime_entity_id IS NOT NULL;
