-- ONEX Topic Migration Projection Schema
-- Ticket: OMN-12623 (Section 6b: migration executor + status projection)
-- Version: 1.0.0
--
-- Design Notes:
--   - migration_ticket = primary key (one projection row per migration contract)
--   - current_state is the migration FSM phase (matches EnumMigrationPhase in core)
--   - Idempotency columns (last_applied_event_id/offset/sequence/partition) mirror
--     registration_projections so the projection is replay-safe and rejects stale
--     or duplicate lifecycle events.
--   - Schema is idempotent (IF NOT EXISTS used throughout); safe to re-run.
--
-- Modeled on: schema_registration_projection.sql (OMN-944).
--
-- Related Tickets:
--   - OMN-12621 (T3): ModelTopicMigrationContract + EnumMigrationPhase (core)
--   - OMN-12623 (T4): migration executor + projection (infra)

-- =============================================================================
-- ENUM TYPES
-- =============================================================================

-- Create enum type for migration phases (matches EnumMigrationPhase in core Python).
-- Note: PostgreSQL enums are immutable - to add values, use ALTER TYPE ... ADD VALUE.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'topic_migration_phase') THEN
        CREATE TYPE topic_migration_phase AS ENUM (
            'planned',
            'dual_write',
            'dual_read',
            'cutover',
            'complete'
        );
    END IF;
END$$;

-- =============================================================================
-- MAIN TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS topic_migration_projection (
    -- Identity (one row per migration contract)
    migration_ticket VARCHAR(64) NOT NULL,

    -- Migration descriptors
    old_topic VARCHAR(255) NOT NULL,
    new_topic VARCHAR(255) NOT NULL,
    old_consumer_group VARCHAR(255) NOT NULL,
    new_consumer_group VARCHAR(255) NOT NULL,

    -- FSM State
    current_state topic_migration_phase NOT NULL,

    -- Observed migration facts
    new_topic_provisioned BOOLEAN NOT NULL DEFAULT FALSE,
    retirement_allowed BOOLEAN NOT NULL DEFAULT FALSE,
    residual_lag BIGINT NOT NULL DEFAULT 0,

    -- Idempotency and Ordering (mirror registration_projections)
    -- These fields enable replay correctness and stale update rejection.
    last_applied_event_id UUID NOT NULL,
    last_applied_sequence BIGINT NOT NULL,
    last_applied_offset BIGINT NOT NULL DEFAULT 0,
    last_applied_partition VARCHAR(128),

    -- Timestamps
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    PRIMARY KEY (migration_ticket),
    CONSTRAINT valid_migration_residual_lag CHECK (residual_lag >= 0),
    CONSTRAINT valid_migration_offset CHECK (last_applied_offset >= 0),
    CONSTRAINT valid_migration_sequence CHECK (last_applied_sequence >= 0)
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- State filtering (operator queries for in-flight migrations)
CREATE INDEX IF NOT EXISTS idx_topic_migration_current_state
    ON topic_migration_projection (current_state);

-- Idempotency lookup by last applied event id
CREATE INDEX IF NOT EXISTS idx_topic_migration_last_event_id
    ON topic_migration_projection (last_applied_event_id);

-- Old-group retirement scan (migrations awaiting drain proof)
CREATE INDEX IF NOT EXISTS idx_topic_migration_retirement
    ON topic_migration_projection (retirement_allowed, current_state);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE topic_migration_projection IS
    'Materialized topic-migration FSM state (OMN-12623). One row per migration '
    'contract; idempotency columns make projection replay-safe.';

COMMENT ON COLUMN topic_migration_projection.migration_ticket IS
    'Migration contract ticket id (OMN-XXXX) - primary key.';

COMMENT ON COLUMN topic_migration_projection.current_state IS
    'Current migration FSM phase. Matches EnumMigrationPhase in omnibase_core.';

COMMENT ON COLUMN topic_migration_projection.retirement_allowed IS
    'Whether the drain-proof gate has permitted retiring the old consumer group.';

COMMENT ON COLUMN topic_migration_projection.residual_lag IS
    'Residual un-consumed lag on the old topic for the old group at last transition.';

COMMENT ON COLUMN topic_migration_projection.last_applied_event_id IS
    'event_id of last applied lifecycle event - used for idempotency checks.';

COMMENT ON COLUMN topic_migration_projection.last_applied_sequence IS
    'Sequence number of last applied lifecycle event - canonical ordering for '
    'stale-update rejection during replay.';

COMMENT ON COLUMN topic_migration_projection.last_applied_offset IS
    'Kafka offset of last applied lifecycle event.';

COMMENT ON COLUMN topic_migration_projection.last_applied_partition IS
    'Kafka partition of last applied lifecycle event.';
