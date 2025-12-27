-- ONEX Registration Projection - Audit Query Index Migration
-- Ticket: PR #101 Review Feedback (Add updated_at index for audit queries)
-- Version: 1.0.1
--
-- Design Notes:
--   - Adds index on updated_at for efficient audit trail queries
--   - Supports time-range filtering for compliance and debugging
--   - Standard B-tree indexes (not partial) since updated_at is NOT NULL
--   - Composite index with current_state enables efficient audit-by-state queries
--
-- Query Patterns Optimized:
--   1. Time-range audit: WHERE updated_at >= :start AND updated_at < :end
--   2. Recent activity: WHERE updated_at > :threshold ORDER BY updated_at DESC
--   3. State change audit: WHERE current_state = :state AND updated_at >= :since
--   4. Entity audit trail: WHERE entity_id = :id ORDER BY updated_at
--
-- Usage:
--   Execute this SQL file to add audit-optimized indexes.
--   The migration is idempotent (IF NOT EXISTS used throughout).
--
-- Related Tickets:
--   - OMN-944 (F1): Registration Projection Schema
--   - PR #101: Performance optimization feedback

-- =============================================================================
-- AUDIT QUERY INDEXES
-- =============================================================================

-- Index for time-range audit queries
-- Query pattern: SELECT * FROM registration_projections
--                WHERE updated_at >= :start AND updated_at < :end
--                ORDER BY updated_at DESC
-- This is the primary audit index for retrieving changes in a time window.
CREATE INDEX IF NOT EXISTS idx_registration_updated_at
    ON registration_projections (updated_at DESC);

-- Composite index for state-based audit queries
-- Query pattern: SELECT * FROM registration_projections
--                WHERE current_state = :state AND updated_at >= :since
--                ORDER BY updated_at DESC
-- Optimizes queries like "find all entities that became 'active' in the last hour"
CREATE INDEX IF NOT EXISTS idx_registration_state_updated_at
    ON registration_projections (current_state, updated_at DESC);

-- Composite index for entity audit trail
-- Query pattern: SELECT * FROM registration_projections
--                WHERE entity_id = :id
--                ORDER BY updated_at DESC
-- Note: Primary key already covers (entity_id, domain), but this index
-- optimizes single-entity audit queries with time ordering.
-- Commented out as PK may be sufficient - uncomment if needed:
-- CREATE INDEX IF NOT EXISTS idx_registration_entity_updated_at
--     ON registration_projections (entity_id, updated_at DESC);

-- =============================================================================
-- INDEX COMMENTS
-- =============================================================================

COMMENT ON INDEX idx_registration_updated_at IS
    'B-tree index on updated_at (DESC) for efficient time-range audit queries. '
    'Supports compliance reporting and debugging recent changes.';

COMMENT ON INDEX idx_registration_state_updated_at IS
    'Composite index on (current_state, updated_at DESC) for state-based audit queries. '
    'Enables efficient queries like "all entities entering state X since time Y".';

-- =============================================================================
-- AUDIT QUERY EXAMPLES
-- =============================================================================
-- These queries should use the new indexes efficiently:
--
-- 1. Recent changes (last hour):
--    SELECT entity_id, current_state, updated_at
--    FROM registration_projections
--    WHERE updated_at > NOW() - INTERVAL '1 hour'
--    ORDER BY updated_at DESC;
--
-- 2. Changes in time window:
--    SELECT entity_id, current_state, updated_at
--    FROM registration_projections
--    WHERE updated_at >= '2025-01-01 00:00:00'
--      AND updated_at < '2025-01-02 00:00:00'
--    ORDER BY updated_at DESC;
--
-- 3. State-filtered audit:
--    SELECT entity_id, updated_at
--    FROM registration_projections
--    WHERE current_state = 'active'
--      AND updated_at > NOW() - INTERVAL '1 day'
--    ORDER BY updated_at DESC;
--
-- 4. Count changes by state in time window:
--    SELECT current_state, COUNT(*) as changes
--    FROM registration_projections
--    WHERE updated_at >= NOW() - INTERVAL '24 hours'
--    GROUP BY current_state;

-- =============================================================================
-- ROLLBACK MIGRATION
-- =============================================================================
-- To rollback this migration, execute the following:
--
--   DROP INDEX IF EXISTS idx_registration_updated_at;
--   DROP INDEX IF EXISTS idx_registration_state_updated_at;
--
-- Note: Dropping indexes is non-destructive (no data loss).

-- =============================================================================
-- INDEX MAINTENANCE NOTES
-- =============================================================================
-- B-tree indexes on timestamp columns may become bloated with high UPDATE frequency.
-- Monitor index size with: SELECT pg_size_pretty(pg_relation_size('idx_registration_updated_at'));
-- Consider REINDEX CONCURRENTLY during maintenance windows if bloat exceeds 50%.
-- PostgreSQL 12+ supports REINDEX CONCURRENTLY for online index rebuild.
