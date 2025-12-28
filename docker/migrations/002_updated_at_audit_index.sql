-- ONEX Registration Projection - Audit Query Index Migration
-- Ticket: PR #101 Review Feedback (Add updated_at index for audit queries)
-- Version: 1.0.1
--
-- Design Notes:
--   - Adds index on updated_at for efficient audit trail queries
--   - Supports time-range filtering for compliance and debugging
--   - These are STANDARD B-tree indexes covering ALL rows (no WHERE clause)
--   - Unlike the partial indexes in 001_registration_projection.sql (which use
--     WHERE clauses to index only rows matching specific conditions), these
--     audit indexes cover ALL rows because audit queries need complete history
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
-- NOTE: This is a FULL TABLE index (no WHERE clause) - indexes ALL rows.
--       Audit queries need complete history, unlike the partial indexes in
--       001_registration_projection.sql which only index specific row subsets.
CREATE INDEX IF NOT EXISTS idx_registration_updated_at
    ON registration_projections (updated_at DESC);

-- Composite index for state-based audit queries
-- Query pattern: SELECT * FROM registration_projections
--                WHERE current_state = :state AND updated_at >= :since
--                ORDER BY updated_at DESC
-- Optimizes queries like "find all entities that became 'active' in the last hour"
-- NOTE: This is a FULL TABLE composite index (no WHERE clause) - indexes ALL rows.
--       Enables efficient filtering by state with time ordering for audit purposes.
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
    'FULL TABLE B-tree index on updated_at (DESC) for time-range audit queries. '
    'No WHERE clause - indexes ALL rows for complete audit history. '
    'Supports compliance reporting and debugging recent changes.';

COMMENT ON INDEX idx_registration_state_updated_at IS
    'FULL TABLE composite index on (current_state, updated_at DESC) for state-based audits. '
    'No WHERE clause - indexes ALL rows unlike partial indexes in 001_registration_projection.sql. '
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
-- PRODUCTION DATABASE SAFETY
-- =============================================================================
-- CAUTION: Index creation can impact production database performance.
--
-- 1. INDEX CREATION TIME: CREATE INDEX IF NOT EXISTS is safe but may take time
--    on large tables. PostgreSQL 11+ creates indexes without blocking writes,
--    but CPU/IO overhead may impact query performance.
--
-- 2. CONCURRENT OPTION: For zero-downtime index creation on large tables, consider:
--    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_registration_updated_at ...
--    Note: CONCURRENTLY cannot be used in a transaction block.
--
-- 3. DISK SPACE: Indexes consume disk space. Monitor disk usage after creation:
--    SELECT pg_size_pretty(pg_relation_size('idx_registration_updated_at'));
--
-- 4. VERIFY: After migration, verify indexes exist and are being used:
--    \di idx_registration_updated_at
--    EXPLAIN ANALYZE SELECT * FROM registration_projections WHERE updated_at > NOW() - INTERVAL '1 hour';

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
-- These are FULL TABLE B-tree indexes (no WHERE clause), NOT partial indexes.
-- They index ALL rows in registration_projections for complete audit coverage.
--
-- Maintenance considerations:
-- 1. BLOAT: B-tree indexes on timestamp columns may become bloated with high
--    UPDATE frequency. Monitor size with:
--    SELECT pg_size_pretty(pg_relation_size('idx_registration_updated_at'));
-- 2. REINDEX: Consider REINDEX CONCURRENTLY during maintenance windows if
--    bloat exceeds 50%. PostgreSQL 12+ supports online rebuild.
-- 3. STORAGE: Full table indexes consume more space than partial indexes.
--    For a table with N rows, these indexes will have N entries each.
--    Compare with partial indexes in 001_registration_projection.sql which
--    only index row subsets matching their WHERE clauses.
