-- =============================================================================
-- PRODUCTION MIGRATION: Create GIN Indexes CONCURRENTLY
-- =============================================================================
-- Ticket: OMN-1134 (Registry Projection Extensions for Capabilities)
-- Version: 1.0.0
-- Companion to: forward/003_capability_fields.sql
--
-- PURPOSE:
--   This script creates the same indexes as 003_capability_fields.sql but uses
--   CREATE INDEX CONCURRENTLY to avoid blocking table writes in production.
--
-- =============================================================================
-- WHEN TO USE THIS SCRIPT vs 003_capability_fields.sql:
-- =============================================================================
--
--   USE 003_capability_fields.sql (standard indexes) WHEN:
--   - Running in development or staging environments
--   - Table has < 100,000 rows
--   - Brief table locks (seconds) are acceptable
--   - Running inside a transaction block (e.g., CI/CD pipeline migrations)
--
--   USE THIS SCRIPT (004_capability_fields_concurrent.sql) WHEN:
--   - Running in production with live traffic
--   - Table has > 100,000 rows (recommended for > 1M rows)
--   - Zero-downtime deployments are required
--   - Table write availability is critical
--
-- =============================================================================
-- IMPORTANT: TRANSACTION REQUIREMENTS
-- =============================================================================
--
-- CONCURRENTLY indexes CANNOT be created inside a transaction block.
-- This script MUST be run with autocommit enabled.
--
-- If using psql:
--   \set AUTOCOMMIT on
--   \i 004_capability_fields_concurrent.sql
--
-- If using psql with -f flag:
--   psql -h <host> -d <database> -f 004_capability_fields_concurrent.sql
--   (autocommit is enabled by default)
--
-- DO NOT USE:
--   BEGIN;
--   \i 004_capability_fields_concurrent.sql
--   COMMIT;
--   -- This will ERROR with: "CREATE INDEX CONCURRENTLY cannot run inside a transaction block"
--
-- =============================================================================
-- PREREQUISITES
-- =============================================================================
--
-- 1. Run 003_capability_fields.sql FIRST (for column creation)
--    This script creates ONLY the indexes, not the columns.
--    The columns must already exist from the base migration.
--
-- 2. If using 003_capability_fields.sql's standard indexes already:
--    Drop the existing indexes first (see CLEANUP section below)
--
-- =============================================================================
-- CLEANUP: Drop Standard Indexes (if already created)
-- =============================================================================
-- If you already ran 003_capability_fields.sql and want to replace with
-- concurrent indexes, uncomment and run these DROP statements first:
--
-- DROP INDEX IF EXISTS idx_registration_capability_tags;
-- DROP INDEX IF EXISTS idx_registration_intent_types;
-- DROP INDEX IF EXISTS idx_registration_protocols;
-- DROP INDEX IF EXISTS idx_registration_contract_type_state;
--
-- Note: DROP INDEX is quick and does not require CONCURRENTLY
--
-- =============================================================================
-- ESTIMATED EXECUTION TIME
-- =============================================================================
--
-- GIN indexes on array columns scale with:
--   - Number of rows
--   - Average array length
--   - I/O performance (SSD vs HDD)
--
-- Rough estimates for idx_registration_capability_tags (avg 5 tags per row):
--
--   | Row Count    | Estimated Time | Notes                          |
--   |--------------|----------------|--------------------------------|
--   | < 10,000     | < 1 second     | Trivial                        |
--   | 100,000      | 2-5 seconds    | Minor impact                   |
--   | 1,000,000    | 30-90 seconds  | Plan maintenance window        |
--   | 10,000,000   | 5-15 minutes   | Monitor closely                |
--   | 100,000,000  | 30-120 minutes | Consider off-peak hours        |
--
-- B-tree index (idx_registration_contract_type_state) is ~2-3x faster than GIN.
--
-- =============================================================================
-- MONITORING DURING INDEX CREATION
-- =============================================================================
--
-- PostgreSQL 12+ provides progress monitoring for CREATE INDEX:
--
-- Monitor index creation progress (run in another session):
--
--   SELECT
--       p.phase,
--       p.blocks_total,
--       p.blocks_done,
--       ROUND(100.0 * p.blocks_done / NULLIF(p.blocks_total, 0), 1) AS pct_complete,
--       p.tuples_total,
--       p.tuples_done,
--       i.relname AS index_name
--   FROM pg_stat_progress_create_index p
--   JOIN pg_class i ON i.oid = p.index_relid
--   WHERE p.datname = current_database();
--
-- Check for blocking queries (run in another session):
--
--   SELECT
--       pid,
--       now() - pg_stat_activity.query_start AS duration,
--       query,
--       state,
--       wait_event_type,
--       wait_event
--   FROM pg_stat_activity
--   WHERE state != 'idle'
--     AND query ILIKE '%create index%';
--
-- Check current locks:
--
--   SELECT
--       l.relation::regclass AS table_name,
--       l.mode,
--       l.granted,
--       a.query
--   FROM pg_locks l
--   JOIN pg_stat_activity a ON l.pid = a.pid
--   WHERE l.relation = 'registration_projections'::regclass;
--
-- =============================================================================
-- HANDLING FAILURES
-- =============================================================================
--
-- If a CONCURRENTLY index creation fails (e.g., due to duplicate values,
-- deadlock, or cancellation), the index will be left in an INVALID state.
--
-- Check for invalid indexes:
--
--   SELECT indexrelid::regclass AS index_name, indisvalid
--   FROM pg_index
--   WHERE NOT indisvalid
--     AND indrelid = 'registration_projections'::regclass;
--
-- To recover from an invalid index:
--
--   DROP INDEX IF EXISTS <invalid_index_name>;
--   -- Then re-run the CREATE INDEX CONCURRENTLY statement
--
-- =============================================================================
-- BEGIN INDEX CREATION
-- =============================================================================

-- Verify all required columns exist before creating indexes
DO $$
DECLARE
    missing_columns TEXT[] := ARRAY[]::TEXT[];
    required_columns TEXT[] := ARRAY['capability_tags', 'intent_types', 'protocols', 'contract_type'];
    col TEXT;
BEGIN
    FOREACH col IN ARRAY required_columns
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'registration_projections'
              AND column_name = col
        ) THEN
            missing_columns := array_append(missing_columns, col);
        END IF;
    END LOOP;

    IF array_length(missing_columns, 1) > 0 THEN
        RAISE EXCEPTION 'Missing columns: %. Run 003_capability_fields.sql first.', array_to_string(missing_columns, ', ');
    END IF;
END$$;

-- -----------------------------------------------------------------------------
-- GIN INDEX: capability_tags
-- -----------------------------------------------------------------------------
-- WHY GIN: GIN (Generalized Inverted Index) indexes individual array elements,
-- enabling efficient @> (contains) queries. B-tree cannot search within arrays.
--
-- Query pattern: WHERE capability_tags @> ARRAY['postgres.storage']
-- Supports: Array containment queries for capability-based discovery
-- Use case: Find nodes with specific capabilities (storage, messaging, etc.)

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_registration_capability_tags
    ON registration_projections USING GIN (capability_tags);

-- -----------------------------------------------------------------------------
-- GIN INDEX: intent_types
-- -----------------------------------------------------------------------------
-- WHY GIN: Same as above - enables efficient array element searches.
--
-- Query pattern: WHERE intent_types @> ARRAY['postgres.upsert']
-- Supports: Finding nodes that handle specific intent types for routing
-- Use case: Route intents to nodes capable of handling them

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_registration_intent_types
    ON registration_projections USING GIN (intent_types);

-- -----------------------------------------------------------------------------
-- GIN INDEX: protocols
-- -----------------------------------------------------------------------------
-- WHY GIN: Same as above - enables efficient array element searches.
--
-- Query pattern: WHERE protocols @> ARRAY['ProtocolDatabaseAdapter']
-- Supports: Protocol-based service discovery
-- Use case: Find nodes implementing specific protocols

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_registration_protocols
    ON registration_projections USING GIN (protocols);

-- -----------------------------------------------------------------------------
-- B-TREE INDEX: contract_type + current_state
-- -----------------------------------------------------------------------------
-- WHY B-TREE: contract_type is a scalar value (not array), so B-tree is optimal.
-- Composite index on (contract_type, current_state) supports both:
--   - WHERE contract_type = 'effect'
--   - WHERE contract_type = 'effect' AND current_state = 'active'
--
-- Query pattern: WHERE contract_type = 'effect' AND current_state = 'active'
-- Supports: Filtering by node type and state
--
-- NOTE on contract_type values:
--   - 'effect', 'compute', 'reducer', 'orchestrator': Valid ONEX node archetypes
--   - 'unknown': Backfill idempotency marker for legacy records where the type
--     could not be determined from capabilities JSONB (NOT an error state)
--   - NULL: Record not yet processed/backfilled
--   See 003_capability_fields.sql CHECK constraint for full documentation.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_registration_contract_type_state
    ON registration_projections (contract_type, current_state);

-- =============================================================================
-- VERIFICATION
-- =============================================================================
--
-- After successful completion, verify indexes are valid:
--
--   SELECT
--       i.relname AS index_name,
--       pg_size_pretty(pg_relation_size(i.oid)) AS index_size,
--       idx.indisvalid AS is_valid
--   FROM pg_class t
--   JOIN pg_index idx ON t.oid = idx.indrelid
--   JOIN pg_class i ON i.oid = idx.indexrelid
--   WHERE t.relname = 'registration_projections'
--     AND i.relname LIKE 'idx_registration_%'
--   ORDER BY i.relname;
--
-- Expected output (all indexes should have is_valid = true):
--
--   | index_name                           | index_size | is_valid |
--   |--------------------------------------|------------|----------|
--   | idx_registration_capability_tags     | 16 kB      | t        |
--   | idx_registration_contract_type_state | 16 kB      | t        |
--   | idx_registration_intent_types        | 16 kB      | t        |
--   | idx_registration_protocols           | 16 kB      | t        |
--
-- =============================================================================
-- ROLLBACK (if needed)
-- =============================================================================
--
-- To remove these indexes:
--
--   DROP INDEX CONCURRENTLY IF EXISTS idx_registration_capability_tags;
--   DROP INDEX CONCURRENTLY IF EXISTS idx_registration_intent_types;
--   DROP INDEX CONCURRENTLY IF EXISTS idx_registration_protocols;
--   DROP INDEX CONCURRENTLY IF EXISTS idx_registration_contract_type_state;
--
-- Note: DROP INDEX CONCURRENTLY also cannot run inside a transaction block.
--
-- =============================================================================
