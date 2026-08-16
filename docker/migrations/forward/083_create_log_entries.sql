-- =============================================================================
-- TOMBSTONE (OMN-15846, 2026-08-10): this file is UNDELIVERABLE via the k8s
-- Job that applies docker/migrations/forward/*.sql
-- (omninode_infra/k8s/migrations/omnibase-infra-migrate.yaml). That Job owns
-- only the omnibase_infra database; its flat loop's `psql -f` apply is
-- gated on `directive_db == $DB_NAME` and is UNREACHABLE for this file's
-- `\connect omnidash_analytics` below, in that loop or any other in the
-- runner. It has never executed anywhere -- live-confirmed 2026-08-10
-- (role_omnidash / omninode_runtime, omninode-dev-postgres RDS):
-- `to_regclass('public.log_entries')` is NULL in BOTH omnibase_infra and
-- omnidash_analytics. omnibase_infra's own schema_migrations carried a
-- false "applied" row for this file (applied_at 2026-07-28T22:53:47Z,
-- checksum byte-identical to this file's live content) -- the same
-- OMN-15819 masking class OMN-15846's classification-ordering fix
-- unmasked for 098/099. Kept in place, byte-unchanged below this header,
-- as ledgered history (migration files are append-only) -- do NOT delete
-- it and do NOT try to make it deliverable in place; the fix is the
-- node-owned migration under
-- docker/migrations/forward/nodes/node_log_persistence_effect/, which
-- CREATEs (this file's original intent, unchanged content) the table via
-- the node-owned loop's role_omnidash connection to omnidash_analytics --
-- the one code path in the runner that can actually reach this database.
-- Static pre-merge enforcement in THIS repo:
-- tests/ci/test_flat_migration_no_foreign_connect_gate.py /
-- docker/migrations/forward/cross-database-flat-migrations.yaml.
-- =============================================================================
-- MIGRATION: Create log_entries table in omnidash_analytics
-- =============================================================================
-- Ticket: OMN-12131 (Event Bus Observability — log_entries table)
-- Version: 1.0.0
--
-- PURPOSE:
--   Creates the log_entries table in the omnidash_analytics database to store
--   structured log events emitted by ONEX nodes. Feeds the event bus
--   observability pipeline consumed by node_log_persistence_effect.
--
--   Retention: 30-day rolling window. Rows older than 30 days should be
--   purged by a scheduled maintenance job (e.g., pg_cron or a nightly
--   retention node). The ingested_at column is the retention anchor.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   See rollback/rollback_083_create_log_entries.sql
-- =============================================================================

\connect omnidash_analytics

-- =============================================================================
-- TABLE: log_entries
-- =============================================================================

CREATE TABLE IF NOT EXISTS log_entries (
    entry_id        UUID            PRIMARY KEY,
    timestamp       TIMESTAMPTZ     NOT NULL,
    node_name       TEXT            NOT NULL,
    function_name   TEXT            NOT NULL DEFAULT '',
    level           TEXT            NOT NULL DEFAULT 'info',
    message         TEXT            NOT NULL,
    correlation_id  TEXT,
    duration_ms     DOUBLE PRECISION,
    metadata        JSONB           NOT NULL DEFAULT '{}',
    ingested_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE log_entries IS
    'Structured log events from ONEX nodes (OMN-12131). '
    '30-day retention window anchored on ingested_at. '
    'Consumed by node_log_persistence_effect via onex.evt.*.log-emitted.v1.';

COMMENT ON COLUMN log_entries.entry_id IS
    'Client-supplied UUID — idempotency key for exactly-once ingestion.';

COMMENT ON COLUMN log_entries.ingested_at IS
    '30-day retention anchor. Rows with ingested_at < NOW() - INTERVAL ''30 days'' '
    'are eligible for pruning by the nightly retention job.';

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Partial index: correlation lookups (skip NULL rows to reduce index size)
CREATE INDEX IF NOT EXISTS idx_log_entries_correlation
    ON log_entries (correlation_id)
    WHERE correlation_id IS NOT NULL;

-- Composite: node-scoped time-range queries (dashboard primary query path)
CREATE INDEX IF NOT EXISTS idx_log_entries_node_ts
    ON log_entries (node_name, timestamp DESC);

-- Composite: level-filtered time-range queries (error/warn filtering)
CREATE INDEX IF NOT EXISTS idx_log_entries_level_ts
    ON log_entries (level, timestamp DESC);

-- Standalone timestamp: full time-range scans and retention sweeps
CREATE INDEX IF NOT EXISTS idx_log_entries_ts
    ON log_entries (timestamp DESC);
