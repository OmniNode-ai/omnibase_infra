-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 OmniNode Team
--
-- Migration: 002_create_validation_event_ledger.sql
-- Purpose: Create the validation_event_ledger table for cross-repo validation event persistence and replay
-- Author: ONEX Infrastructure Team
-- Date: 2026-02-06
--
-- Design Decisions:
--   1. Domain-specific first-class columns: run_id, repo_id, event_type, and event_version
--      are all NOT NULL because validation events have a well-defined schema. Unlike the
--      generic event_ledger (which must tolerate malformed events), validation events are
--      structurally guaranteed by the producing pipeline.
--
--   2. BYTEA for envelope_bytes: Raw byte-level preservation for deterministic replay.
--      The validation replay subsystem requires bit-for-bit identical event reconstruction,
--      making binary storage essential.
--
--   3. TEXT for envelope_hash: SHA-256 hex digest of envelope_bytes for integrity
--      verification during replay. Consumers can verify hash(envelope_bytes) == envelope_hash
--      before processing to detect storage corruption.
--
--   4. Idempotency via (kafka_topic, kafka_partition, kafka_offset): Same pattern as
--      the generic event_ledger. ON CONFLICT DO NOTHING ensures exactly-once semantics
--      for validation event capture even with consumer restarts or rebalancing.
--
--   5. occurred_at from event payload vs created_at from database: occurred_at reflects
--      when the validation event actually happened; created_at records when we persisted it.
--      Both are NOT NULL for audit completeness.

-- =============================================================================
-- TABLE: validation_event_ledger
-- =============================================================================
-- The validation_event_ledger provides durable, append-only storage of all
-- cross-repo validation events consumed from Kafka. It serves as:
--   - Source of truth for validation run replay and reprocessing
--   - Audit trail for cross-repository validation outcomes
--   - Integrity-verified store via envelope_hash for deterministic replay
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_event_ledger (
    -- Primary key: Auto-generated UUID for ledger entry identification
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- =========================================================================
    -- Validation Event Identity
    -- =========================================================================
    -- These fields identify the validation context. All NOT NULL because
    -- validation events are structurally well-defined.
    run_id              UUID            NOT NULL,       -- Validation run correlation ID
    repo_id             TEXT            NOT NULL,       -- Repository being validated
    event_type          TEXT            NOT NULL,       -- Event type (e.g., onex.evt.validation.cross-repo-run-started.v1)
    event_version       TEXT            NOT NULL,       -- Event schema version

    -- =========================================================================
    -- Temporal
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
    -- Raw Event Data
    -- =========================================================================
    -- Preserved as raw bytes for deterministic replay. The hash provides
    -- integrity verification without needing to deserialize.
    envelope_bytes      BYTEA           NOT NULL,       -- Raw envelope bytes for deterministic replay
    envelope_hash       TEXT            NOT NULL,       -- SHA-256 hex digest of envelope_bytes

    -- =========================================================================
    -- Ledger Metadata
    -- =========================================================================
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),  -- When this ledger entry was persisted

    -- =========================================================================
    -- Constraints
    -- =========================================================================
    -- Idempotency constraint: Ensures each Kafka message is recorded exactly once.
    -- On consumer restart, duplicate inserts will fail gracefully (ON CONFLICT DO NOTHING).
    CONSTRAINT uk_validation_ledger_kafka_position UNIQUE (kafka_topic, kafka_partition, kafka_offset)
);

-- =============================================================================
-- INDEXES
-- =============================================================================
-- Optimized for common query patterns: run lookups, repo+time scans,
-- and repo+run composite queries.

-- Index 1: Run ID lookups
-- Use case: Retrieving all events for a specific validation run
CREATE INDEX IF NOT EXISTS idx_validation_ledger_run_id
    ON validation_event_ledger (run_id);

-- Index 2: Repository + occurred_at descending
-- Use case: Finding recent validation events for a specific repository
CREATE INDEX IF NOT EXISTS idx_validation_ledger_repo_occurred
    ON validation_event_ledger (repo_id, occurred_at DESC);

-- Index 3: Repository + Run ID composite
-- Use case: Scoping a validation run to a specific repository
CREATE INDEX IF NOT EXISTS idx_validation_ledger_repo_run
    ON validation_event_ledger (repo_id, run_id);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE validation_event_ledger IS
    'Durable, append-only ledger of cross-repo validation events consumed from Kafka. Provides replay capability, integrity verification via envelope_hash, and idempotency guarantees.';

COMMENT ON COLUMN validation_event_ledger.id IS
    'Auto-generated UUID primary key for this ledger entry';

COMMENT ON COLUMN validation_event_ledger.run_id IS
    'Validation run correlation ID linking all events in a single validation run';

COMMENT ON COLUMN validation_event_ledger.repo_id IS
    'Identifier of the repository being validated';

COMMENT ON COLUMN validation_event_ledger.event_type IS
    'Fully qualified event type (e.g., onex.evt.validation.cross-repo-run-started.v1)';

COMMENT ON COLUMN validation_event_ledger.event_version IS
    'Schema version of the event type for forward/backward compatibility';

COMMENT ON COLUMN validation_event_ledger.occurred_at IS
    'Timestamp from the event payload indicating when the validation event occurred';

COMMENT ON COLUMN validation_event_ledger.kafka_topic IS
    'Kafka topic from which the event was consumed';

COMMENT ON COLUMN validation_event_ledger.kafka_partition IS
    'Kafka partition number (idempotency key component)';

COMMENT ON COLUMN validation_event_ledger.kafka_offset IS
    'Kafka offset within the partition (idempotency key component)';

COMMENT ON COLUMN validation_event_ledger.envelope_bytes IS
    'Raw envelope bytes stored as BYTEA for bit-level deterministic replay';

COMMENT ON COLUMN validation_event_ledger.envelope_hash IS
    'SHA-256 hex digest of envelope_bytes for integrity verification during replay';

COMMENT ON COLUMN validation_event_ledger.created_at IS
    'Timestamp when this entry was persisted to the ledger (database server time)';

COMMENT ON CONSTRAINT uk_validation_ledger_kafka_position ON validation_event_ledger IS
    'Idempotency constraint: ensures each Kafka message is recorded exactly once';
