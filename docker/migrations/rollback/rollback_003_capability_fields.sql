-- Rollback: 003_capability_fields
-- Description: Remove capability fields and indexes from registration_projections
-- Forward migration: forward/003_capability_fields.sql

-- Drop indexes first (before removing columns they reference)
DROP INDEX IF EXISTS idx_registration_capability_tags;
DROP INDEX IF EXISTS idx_registration_intent_types;
DROP INDEX IF EXISTS idx_registration_protocols;
DROP INDEX IF EXISTS idx_registration_contract_type_state;

-- Drop constraint
ALTER TABLE registration_projections
    DROP CONSTRAINT IF EXISTS valid_contract_type;

-- Drop columns
ALTER TABLE registration_projections
    DROP COLUMN IF EXISTS contract_type,
    DROP COLUMN IF EXISTS intent_types,
    DROP COLUMN IF EXISTS protocols,
    DROP COLUMN IF EXISTS capability_tags,
    DROP COLUMN IF EXISTS contract_version;
