-- Rollback: forward/002_updated_at_audit_index.sql
-- Purpose: Drop audit query indexes from registration_projections
--
-- Note: Dropping indexes is non-destructive (no data loss).

DROP INDEX IF EXISTS idx_registration_updated_at;
DROP INDEX IF EXISTS idx_registration_state_updated_at;
