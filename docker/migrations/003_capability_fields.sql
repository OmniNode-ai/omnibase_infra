-- =============================================================================
-- MIGRATION: Add Capability Fields for Fast Discovery Queries
-- =============================================================================
-- Ticket: OMN-1134 (Registry Projection Extensions for Capabilities)
-- Version: 1.0.0
--
-- This migration adds denormalized capability fields to the registration
-- projections table to enable fast capability-based queries with GIN indexes.
--
-- NEW COLUMNS:
--   - contract_type: Node contract type (effect, compute, reducer, orchestrator)
--   - intent_types: Array of intent types this node handles
--   - protocols: Array of protocols this node implements
--   - capability_tags: Array of capability tags for discovery
--   - contract_version: Contract version string
--
-- NEW INDEXES:
--   - idx_registration_capability_tags: GIN index for capability_tags array
--   - idx_registration_intent_types: GIN index for intent_types array
--   - idx_registration_protocols: GIN index for protocols array
--   - idx_registration_contract_type_state: B-tree for contract_type + state
--
-- DESIGN NOTES:
--   - Fields are denormalized from the capabilities JSONB column for fast queries
--   - GIN indexes enable efficient @> (contains) array queries
--   - Existing registrations will have NULL/empty values until backfill
--
-- QUERY PATTERNS:
--   -- Find all effect nodes with postgres storage capability
--   SELECT * FROM registration_projections
--   WHERE contract_type = 'effect'
--     AND capability_tags @> ARRAY['postgres.storage'];
--
--   -- Find all nodes that handle postgres.upsert intent
--   SELECT * FROM registration_projections
--   WHERE intent_types @> ARRAY['postgres.upsert'];
--
-- PRODUCTION DATABASE SAFETY:
--   1. BACKUP your database before running this migration
--   2. TEST this migration on a staging environment first
--   3. MONITOR the migration - adding columns and indexes may take time on large tables
--
-- CONCURRENT INDEX CREATION (for large production tables):
-- -------------------------------------------------------------------------------------
-- For large tables in production, consider using CREATE INDEX CONCURRENTLY:
--   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_registration_capability_tags
--       ON registration_projections USING GIN (capability_tags);
--
-- IMPORTANT: CONCURRENTLY indexes cannot be created inside a transaction block.
-- These must be run separately from the ALTER TABLE statements.
-- Standard indexes (as shown in this migration) will lock the table briefly during creation.
-- -------------------------------------------------------------------------------------
--
-- ROLLBACK:
--   ALTER TABLE registration_projections
--     DROP COLUMN IF EXISTS contract_type,
--     DROP COLUMN IF EXISTS intent_types,
--     DROP COLUMN IF EXISTS protocols,
--     DROP COLUMN IF EXISTS capability_tags,
--     DROP COLUMN IF EXISTS contract_version;
--
--   DROP INDEX IF EXISTS idx_registration_capability_tags;
--   DROP INDEX IF EXISTS idx_registration_intent_types;
--   DROP INDEX IF EXISTS idx_registration_protocols;
--   DROP INDEX IF EXISTS idx_registration_contract_type_state;
--
-- =============================================================================

-- Add new columns to registration_projections table
ALTER TABLE registration_projections
    ADD COLUMN IF NOT EXISTS contract_type TEXT,
    ADD COLUMN IF NOT EXISTS intent_types TEXT[] DEFAULT ARRAY[]::TEXT[],
    ADD COLUMN IF NOT EXISTS protocols TEXT[] DEFAULT ARRAY[]::TEXT[],
    ADD COLUMN IF NOT EXISTS capability_tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    ADD COLUMN IF NOT EXISTS contract_version TEXT;

-- =============================================================================
-- GIN INDEXES FOR CAPABILITY ARRAY QUERIES
-- =============================================================================

-- GIN index for capability_tags array queries
-- Query pattern: SELECT * FROM registration_projections
--                WHERE capability_tags @> ARRAY['postgres.storage']
CREATE INDEX IF NOT EXISTS idx_registration_capability_tags
    ON registration_projections USING GIN (capability_tags);

-- GIN index for intent_types array queries
-- Query pattern: SELECT * FROM registration_projections
--                WHERE intent_types @> ARRAY['postgres.upsert']
CREATE INDEX IF NOT EXISTS idx_registration_intent_types
    ON registration_projections USING GIN (intent_types);

-- GIN index for protocols array queries
-- Query pattern: SELECT * FROM registration_projections
--                WHERE protocols @> ARRAY['ProtocolDatabaseAdapter']
CREATE INDEX IF NOT EXISTS idx_registration_protocols
    ON registration_projections USING GIN (protocols);

-- B-tree index for contract_type + state queries
-- Query pattern: SELECT * FROM registration_projections
--                WHERE contract_type = 'effect' AND current_state = 'active'
CREATE INDEX IF NOT EXISTS idx_registration_contract_type_state
    ON registration_projections (contract_type, current_state);

-- =============================================================================
-- CHECK CONSTRAINT FOR CONTRACT_TYPE
-- =============================================================================
-- Ensure contract_type values are consistent with node_type valid values
-- Note: Using DO block to handle constraint already existing (idempotent)

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'valid_contract_type'
        AND conrelid = 'registration_projections'::regclass
    ) THEN
        ALTER TABLE registration_projections
            ADD CONSTRAINT valid_contract_type CHECK (
                contract_type IS NULL
                OR contract_type IN ('effect', 'compute', 'reducer', 'orchestrator')
            );
    END IF;
END$$;

-- =============================================================================
-- VERIFICATION QUERY
-- =============================================================================
-- After running this migration, verify with:
--   SELECT column_name, data_type
--   FROM information_schema.columns
--   WHERE table_name = 'registration_projections'
--     AND column_name IN ('contract_type', 'intent_types', 'protocols', 'capability_tags', 'contract_version');
--
--   SELECT indexname FROM pg_indexes
--   WHERE tablename = 'registration_projections'
--     AND indexname LIKE 'idx_registration_%';
--
--   SELECT conname FROM pg_constraint
--   WHERE conrelid = 'registration_projections'::regclass
--     AND conname = 'valid_contract_type';
-- =============================================================================
