-- Rollback: 004_capability_fields_concurrent.sql
-- Purpose: Drop concurrent capability indexes from registration_projections
--
-- Note: DROP INDEX CONCURRENTLY also cannot run inside a transaction block.
-- If running inside a transaction, use DROP INDEX (without CONCURRENTLY) instead.

DROP INDEX CONCURRENTLY IF EXISTS idx_registration_capability_tags;
DROP INDEX CONCURRENTLY IF EXISTS idx_registration_intent_types;
DROP INDEX CONCURRENTLY IF EXISTS idx_registration_protocols;
DROP INDEX CONCURRENTLY IF EXISTS idx_registration_contract_type_state;
