-- =============================================================================
-- MIGRATION STATE VALIDATION SCRIPT
-- =============================================================================
-- Run this script to detect migration gaps and verify database state.
--
-- Usage:
--   psql -h $HOST -d $DB -f validate_migration_state.sql
--
-- Exit behavior:
--   The script will display a summary table showing the status of each
--   migration. Review the output to determine if any migrations are missing.
--
-- For the 003a to 004 renaming:
--   If your deployment history shows "003a" was applied, the database content
--   is identical to "004". This script validates the actual database state,
--   not the naming convention.
--
-- See MIGRATION_UPGRADE_003a_to_004.md for full documentation.
-- =============================================================================

\echo '=============================================='
\echo 'MIGRATION STATE VALIDATION'
\echo '=============================================='
\echo ''

-- =============================================================================
-- PART 1: Migration Status Summary
-- =============================================================================
\echo 'Migration Status:'
\echo '-----------------'

WITH table_check AS (
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'registration_projections'
          AND table_schema = 'public'
    ) AS table_exists
),
column_check AS (
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'registration_projections'
      AND column_name IN ('contract_type', 'intent_types', 'protocols', 'capability_tags', 'contract_version')
),
index_check AS (
    SELECT i.indexname, idx.indisvalid
    FROM pg_indexes i
    JOIN pg_class c ON c.relname = i.indexname
    JOIN pg_index idx ON idx.indexrelid = c.oid
    WHERE i.tablename = 'registration_projections'
      AND i.indexname IN (
        'idx_registration_projections_updated_at',
        'idx_registration_capability_tags',
        'idx_registration_intent_types',
        'idx_registration_protocols',
        'idx_registration_contract_type_state'
      )
),
outbox_table_check AS (
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'transition_notification_outbox'
          AND table_schema = 'public'
    ) AS table_exists
),
outbox_column_check AS (
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'transition_notification_outbox'
      AND column_name IN ('id', 'notification_data', 'created_at', 'processed_at', 'retry_count', 'last_error', 'aggregate_type', 'aggregate_id')
),
outbox_index_check AS (
    SELECT i.indexname, idx.indisvalid
    FROM pg_indexes i
    JOIN pg_class c ON c.relname = i.indexname
    JOIN pg_index idx ON idx.indexrelid = c.oid
    WHERE i.tablename = 'transition_notification_outbox'
      AND i.indexname IN (
        'idx_outbox_pending',
        'idx_outbox_aggregate',
        'idx_outbox_retry_pending',
        'idx_outbox_cleanup',
        'idx_outbox_aggregate_type_pending'
      )
)
SELECT
    migration,
    CASE status
        WHEN 'APPLIED' THEN '[OK] ' || status
        WHEN 'MISSING' THEN '[!!] ' || status
        WHEN 'PARTIAL' THEN '[??] ' || status
        WHEN 'INVALID_INDEXES' THEN '[XX] ' || status
        ELSE status
    END AS status,
    description
FROM (
    SELECT
        '001' AS migration,
        CASE WHEN (SELECT table_exists FROM table_check) THEN 'APPLIED' ELSE 'MISSING' END AS status,
        'registration_projections table' AS description
    UNION ALL
    SELECT
        '002' AS migration,
        CASE WHEN EXISTS (SELECT 1 FROM index_check WHERE indexname = 'idx_registration_projections_updated_at')
             THEN 'APPLIED' ELSE 'MISSING' END AS status,
        'updated_at audit index' AS description
    UNION ALL
    SELECT
        '003' AS migration,
        CASE WHEN (SELECT COUNT(*) FROM column_check) = 5 THEN 'APPLIED'
             WHEN (SELECT COUNT(*) FROM column_check) > 0 THEN 'PARTIAL'
             ELSE 'MISSING' END AS status,
        'capability columns (5 columns)' AS description
    UNION ALL
    SELECT
        '004' AS migration,
        CASE
            WHEN (SELECT COUNT(*) FROM index_check
                  WHERE indexname IN ('idx_registration_capability_tags', 'idx_registration_intent_types',
                                      'idx_registration_protocols', 'idx_registration_contract_type_state')
                    AND indisvalid = true) = 4 THEN 'APPLIED'
            WHEN (SELECT COUNT(*) FROM index_check
                  WHERE indexname IN ('idx_registration_capability_tags', 'idx_registration_intent_types',
                                      'idx_registration_protocols', 'idx_registration_contract_type_state')
                    AND indisvalid = false) > 0 THEN 'INVALID_INDEXES'
            WHEN (SELECT COUNT(*) FROM index_check
                  WHERE indexname IN ('idx_registration_capability_tags', 'idx_registration_intent_types',
                                      'idx_registration_protocols', 'idx_registration_contract_type_state')) > 0 THEN 'PARTIAL'
            ELSE 'MISSING'
        END AS status,
        'capability GIN indexes (4 indexes)' AS description
    UNION ALL
    SELECT
        '005' AS migration,
        CASE
            WHEN NOT (SELECT table_exists FROM outbox_table_check) THEN 'MISSING'
            WHEN (SELECT COUNT(*) FROM outbox_column_check) < 8 THEN 'PARTIAL'
            WHEN (SELECT COUNT(*) FROM outbox_index_check WHERE indisvalid = false) > 0 THEN 'INVALID_INDEXES'
            WHEN (SELECT COUNT(*) FROM outbox_index_check WHERE indisvalid = true) = 5 THEN 'APPLIED'
            WHEN (SELECT COUNT(*) FROM outbox_index_check) > 0 THEN 'PARTIAL'
            ELSE 'PARTIAL'
        END AS status,
        'transition_notification_outbox table (8 columns, 5 indexes)' AS description
) results
ORDER BY migration;

\echo ''

-- =============================================================================
-- PART 2: Detailed Column Check (for 003 validation)
-- =============================================================================
\echo 'Migration 003 - Capability Columns Detail:'
\echo '-------------------------------------------'

SELECT
    column_name,
    data_type,
    CASE WHEN is_nullable = 'YES' THEN 'nullable' ELSE 'not null' END AS nullable
FROM information_schema.columns
WHERE table_name = 'registration_projections'
  AND column_name IN ('contract_type', 'intent_types', 'protocols', 'capability_tags', 'contract_version')
ORDER BY column_name;

\echo ''

-- =============================================================================
-- PART 3: Detailed Index Check (for 004 validation)
-- =============================================================================
\echo 'Migration 004 - GIN Index Detail:'
\echo '----------------------------------'

SELECT
    i.indexname AS index_name,
    CASE WHEN idx.indisvalid THEN 'valid' ELSE 'INVALID' END AS status,
    pg_size_pretty(pg_relation_size(c.oid)) AS size,
    am.amname AS index_type
FROM pg_indexes i
JOIN pg_class c ON c.relname = i.indexname
JOIN pg_index idx ON idx.indexrelid = c.oid
JOIN pg_am am ON c.relam = am.oid
WHERE i.tablename = 'registration_projections'
  AND i.indexname IN (
    'idx_registration_capability_tags',
    'idx_registration_intent_types',
    'idx_registration_protocols',
    'idx_registration_contract_type_state'
  )
ORDER BY i.indexname;

\echo ''

-- =============================================================================
-- PART 4: Detailed Outbox Table Check (for 005 validation)
-- =============================================================================
\echo 'Migration 005 - Transition Notification Outbox Detail:'
\echo '-------------------------------------------------------'

SELECT
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'transition_notification_outbox'
          AND table_schema = 'public'
    ) THEN 'Table exists' ELSE 'TABLE MISSING' END AS table_status;

\echo ''
\echo 'Outbox Columns:'

SELECT
    column_name,
    data_type,
    CASE WHEN is_nullable = 'YES' THEN 'nullable' ELSE 'not null' END AS nullable
FROM information_schema.columns
WHERE table_name = 'transition_notification_outbox'
  AND column_name IN ('id', 'notification_data', 'created_at', 'processed_at', 'retry_count', 'last_error', 'aggregate_type', 'aggregate_id')
ORDER BY column_name;

\echo ''
\echo 'Outbox Indexes:'

SELECT
    i.indexname AS index_name,
    CASE WHEN idx.indisvalid THEN 'valid' ELSE 'INVALID' END AS status,
    pg_size_pretty(pg_relation_size(c.oid)) AS size,
    am.amname AS index_type
FROM pg_indexes i
JOIN pg_class c ON c.relname = i.indexname
JOIN pg_index idx ON idx.indexrelid = c.oid
JOIN pg_am am ON c.relam = am.oid
WHERE i.tablename = 'transition_notification_outbox'
  AND i.indexname IN (
    'idx_outbox_pending',
    'idx_outbox_aggregate',
    'idx_outbox_retry_pending',
    'idx_outbox_cleanup',
    'idx_outbox_aggregate_type_pending'
  )
ORDER BY i.indexname;

\echo ''

-- =============================================================================
-- PART 5: Invalid Index Detection
-- =============================================================================
\echo 'Invalid Index Check:'
\echo '--------------------'

SELECT
    indexrelid::regclass AS index_name,
    'INVALID - needs recreation' AS status
FROM pg_index
WHERE NOT indisvalid
  AND (indrelid = 'registration_projections'::regclass
       OR (to_regclass('transition_notification_outbox') IS NOT NULL
           AND indrelid = to_regclass('transition_notification_outbox')));

-- If no rows, show message
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_index
        WHERE NOT indisvalid
          AND (indrelid = 'registration_projections'::regclass
               OR (to_regclass('transition_notification_outbox') IS NOT NULL
                   AND indrelid = to_regclass('transition_notification_outbox')))
    ) THEN
        RAISE NOTICE 'No invalid indexes found (good!)';
    END IF;
END$$;

\echo ''

-- =============================================================================
-- PART 6: 003a vs 004 Disambiguation
-- =============================================================================
\echo '003a/004 Disambiguation (for legacy deployments):'
\echo '--------------------------------------------------'

SELECT
    CASE
        WHEN COUNT(*) = 4 THEN 'GIN indexes present - if tracking shows "003a", update to "004"'
        WHEN COUNT(*) > 0 THEN 'Partial indexes (' || COUNT(*) || ' of 4) - migration interrupted'
        ELSE 'No GIN indexes - neither 003a nor 004 applied'
    END AS guidance
FROM pg_indexes
WHERE tablename = 'registration_projections'
  AND indexname IN (
    'idx_registration_capability_tags',
    'idx_registration_intent_types',
    'idx_registration_protocols',
    'idx_registration_contract_type_state'
  );

\echo ''
\echo '=============================================='
\echo 'VALIDATION COMPLETE'
\echo '=============================================='
\echo ''
\echo 'Legend:'
\echo '  [OK] APPLIED        - Migration successfully applied'
\echo '  [!!] MISSING        - Migration needs to be applied'
\echo '  [??] PARTIAL        - Migration partially applied (review needed)'
\echo '  [XX] INVALID_INDEXES - Concurrent index creation interrupted'
\echo ''
\echo 'See MIGRATION_UPGRADE_003a_to_004.md for remediation steps.'
