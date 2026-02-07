-- Migration: 002_create_validation_event_ledger.sql
-- Purpose: Create the validation_event_ledger table for cross-repo validation event persistence and replay
-- Author: ONEX Infrastructure Team
-- Date: 2026-02-07
-- Ticket: OMN-1908
--
-- Design Decisions:
--   1. Domain-specific table: Keeps validation events separate from the generic event_ledger.
--      First-class run_id, repo_id, event_type columns enable efficient domain queries
--      without polluting the generic ledger schema.
--
--   2. BYTEA for envelope_bytes: Raw binary preservation of the complete Kafka envelope.
--      Enables deterministic replay by storing the exact bytes as received.
--      Base64 encoding happens at the application layer (model transport), not in the DB.
--
--   3. TEXT for envelope_hash: SHA-256 hex digest of envelope_bytes for integrity verification.
--      Consumers can verify replay fidelity by comparing hash of decoded bytes.
--
--   4. Idempotency via (kafka_topic, kafka_partition, kafka_offset): Ensures exactly-once
--      semantics for event capture even with consumer restarts or rebalancing.
--      Uses ON CONFLICT DO NOTHING for safe duplicate handling.
--
--   5. Replay ordering by (kafka_partition, kafka_offset): Within a single topic+partition,
--      Kafka offsets provide total ordering. occurred_at is stored for display but NOT used
--      for ordering (clock drift unsafe).
--
--   6. Retention via cleanup_expired(): Time-based + minimum run count per repo.
--      No pg_cron dependency - cleanup is programmatic via repository method.

-- =============================================================================
-- TABLE: validation_event_ledger
-- =============================================================================
-- Domain-specific ledger for cross-repo validation events. Stores validation
-- run lifecycle events (started, violations batch, completed) for:
--   - Replay capability for historical validation runs
--   - Dashboard queries by run_id and repo_id
--   - Deterministic replay via raw envelope bytes
--   - Idempotent Kafka consumer writes
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_event_ledger (
    -- Primary key: UUID for ledger entry identification
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- =========================================================================
    -- Domain Fields (first-class for efficient querying)
    -- =========================================================================
    run_id              UUID            NOT NULL,       -- Links started -> violations -> completed events
    repo_id             TEXT            NOT NULL,       -- Human-readable repository identifier
    event_type          TEXT            NOT NULL,       -- Event type discriminator (e.g., "onex.validation.cross_repo.run.started.v1")
    event_version       TEXT            NOT NULL,       -- Event schema version

    -- =========================================================================
    -- Event Timestamp
    -- =========================================================================
    occurred_at         TIMESTAMPTZ     NOT NULL,       -- When the validation event occurred (from event payload)

    -- =========================================================================
    -- Kafka Position (Idempotency Key)
    -- =========================================================================
    -- These three fields together form the unique idempotency key.
    -- Any consumer restart will attempt to re-insert, but the constraint prevents duplicates.
    kafka_topic         TEXT            NOT NULL,       -- Kafka topic name
    kafka_partition     INTEGER         NOT NULL,       -- Kafka partition number
    kafka_offset        BIGINT          NOT NULL,       -- Kafka offset within partition

    -- =========================================================================
    -- Raw Envelope Data
    -- =========================================================================
    -- Preserved exactly as received from Kafka for deterministic replay.
    envelope_bytes      BYTEA           NOT NULL,       -- Complete Kafka envelope as raw bytes
    envelope_hash       TEXT            NOT NULL,       -- SHA-256 hex digest of envelope_bytes

    -- =========================================================================
    -- Timestamps
    -- =========================================================================
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),  -- When this ledger entry was written

    -- =========================================================================
    -- Constraints
    -- =========================================================================
    -- Idempotency constraint: Ensures each Kafka message is recorded exactly once.
    CONSTRAINT uk_validation_ledger_kafka_position UNIQUE (kafka_topic, kafka_partition, kafka_offset)
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Index 1: Run ID lookups
-- Use case: Replay all events for a specific validation run
CREATE INDEX IF NOT EXISTS idx_validation_ledger_run_id
    ON validation_event_ledger (run_id);

-- Index 2: Repo + time range queries (descending for recent-first)
-- Use case: Dashboard showing recent validation runs for a repository
CREATE INDEX IF NOT EXISTS idx_validation_ledger_repo_occurred_at
    ON validation_event_ledger (repo_id, occurred_at DESC);

-- Index 3: Repo + run composite
-- Use case: Find all events for a specific repo's validation run
CREATE INDEX IF NOT EXISTS idx_validation_ledger_repo_run
    ON validation_event_ledger (repo_id, run_id);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE validation_event_ledger IS
    'Domain-specific ledger for cross-repo validation events. Enables deterministic replay and dashboard queries for validation runs.';

COMMENT ON COLUMN validation_event_ledger.id IS
    'Auto-generated UUID primary key for this ledger entry';

COMMENT ON COLUMN validation_event_ledger.run_id IS
    'Validation run identifier linking started, violations, and completed events';

COMMENT ON COLUMN validation_event_ledger.repo_id IS
    'Human-readable repository identifier (e.g., "omnibase_core", "omnibase_infra")';

COMMENT ON COLUMN validation_event_ledger.event_type IS
    'Event type discriminator (e.g., "onex.validation.cross_repo.run.started.v1")';

COMMENT ON COLUMN validation_event_ledger.event_version IS
    'Event schema version for forward compatibility';

COMMENT ON COLUMN validation_event_ledger.occurred_at IS
    'When the validation event occurred (from event payload timestamp)';

COMMENT ON COLUMN validation_event_ledger.kafka_topic IS
    'Kafka topic from which the event was consumed';

COMMENT ON COLUMN validation_event_ledger.kafka_partition IS
    'Kafka partition number (idempotency key component)';

COMMENT ON COLUMN validation_event_ledger.kafka_offset IS
    'Kafka offset within partition (idempotency key component)';

COMMENT ON COLUMN validation_event_ledger.envelope_bytes IS
    'Complete Kafka envelope as raw BYTEA for deterministic replay';

COMMENT ON COLUMN validation_event_ledger.envelope_hash IS
    'SHA-256 hex digest of envelope_bytes for integrity verification';

COMMENT ON COLUMN validation_event_ledger.created_at IS
    'Timestamp when this entry was written to the ledger';

COMMENT ON CONSTRAINT uk_validation_ledger_kafka_position ON validation_event_ledger IS
    'Idempotency constraint: ensures each Kafka message is recorded exactly once';
