-- Rollback: rollback_005_drop_decision_store.sql
-- Purpose: Reverse migration 005 — drop decision_conflicts and decision_store tables
--          including all indexes and constraints.
-- Ticket: OMN-2764
--
-- NOTE: decision_conflicts must be dropped before decision_store because it holds
--       foreign key references to decision_store(decision_id).

-- =============================================================================
-- DROP: decision_conflicts (child table — references decision_store)
-- =============================================================================

DROP INDEX IF EXISTS idx_decision_conflicts_open;
DROP TABLE IF EXISTS decision_conflicts;

-- =============================================================================
-- DROP: decision_store (parent table)
-- =============================================================================

DROP INDEX IF EXISTS idx_decision_store_scope_services;
DROP INDEX IF EXISTS idx_decision_store_correlation;
DROP INDEX IF EXISTS idx_decision_store_epic_id;
DROP INDEX IF EXISTS idx_decision_store_created_at_id;
DROP INDEX IF EXISTS idx_decision_store_active_scope;
DROP TABLE IF EXISTS decision_store;
