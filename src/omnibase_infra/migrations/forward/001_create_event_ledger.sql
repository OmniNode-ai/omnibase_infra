-- Migration: 001_create_event_ledger.sql
-- Purpose: Create the event_ledger table for durable event capture and replay
-- Author: ONEX Infrastructure Team
-- Date: 2026-01-29
--
-- Design Decisions:
--   1. TEXT vs VARCHAR(255) for topic: Kafka topic names can exceed 255 characters
--      with namespaced conventions (e.g., "dev.archon-intelligence.intelligence.code-analysis-requested.v1")
--
--   2. BYTEA for event_key/event_value: Raw binary preservation without encoding assumptions.
--      Events may contain non-UTF8 data (Protobuf, Avro, MessagePack). TEXT would require
--      encoding validation and could corrupt binary payloads.
--
--   3. All metadata fields NULLABLE: The audit ledger must NEVER drop events due to
--      malformed metadata. Missing envelope_id, correlation_id, event_type, or source
--      must not block event capture. Schema enforcement happens downstream.
--
--   4. JSONB for onex_headers: Structured storage of ONEX-specific headers with indexing
--      capability. Avoids column explosion as header schema evolves.
--
--   5. Dual timestamps: event_timestamp (from event) may be NULL if source doesn't provide it.
--      ledger_written_at provides guaranteed ordering for replay scenarios.
--
--   6. Idempotency via (topic, partition, kafka_offset): Ensures exactly-once semantics
--      for event capture even with consumer restarts or rebalancing.

-- =============================================================================
-- TABLE: event_ledger
-- =============================================================================
-- The event_ledger provides durable, append-only storage of all events consumed
-- from Kafka. It serves as:
--   - Audit trail for compliance and debugging
--   - Source for event replay and reprocessing
--   - Idempotency guard against duplicate processing
-- =============================================================================

CREATE TABLE IF NOT EXISTS event_ledger (
    -- Primary key: Auto-generated UUID for ledger entry identification
    ledger_entry_id     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- =========================================================================
    -- Kafka Position (Idempotency Key)
    -- =========================================================================
    -- These three fields together form the unique idempotency key.
    -- Any consumer restart will attempt to re-insert, but the constraint prevents duplicates.
    topic               TEXT            NOT NULL,       -- Kafka topic name (TEXT for long namespaced topics)
    partition           INTEGER         NOT NULL,       -- Kafka partition number
    kafka_offset        BIGINT          NOT NULL,       -- Kafka offset within partition

    -- =========================================================================
    -- Raw Event Data
    -- =========================================================================
    -- Preserved exactly as received from Kafka, no transformation applied.
    event_key           BYTEA,                          -- Kafka message key (nullable - not all events have keys)
    event_value         BYTEA           NOT NULL,       -- Kafka message value (required - the actual event payload)
    onex_headers        JSONB           NOT NULL DEFAULT '{}',  -- ONEX-specific headers extracted from Kafka headers

    -- =========================================================================
    -- Extracted Metadata (ALL NULLABLE)
    -- =========================================================================
    -- These fields are extracted from the event for query optimization.
    -- ALL are nullable because:
    --   1. Malformed events must still be captured (audit requirement)
    --   2. Legacy events may not have all metadata fields
    --   3. Schema evolution may introduce new optional fields
    -- Missing metadata must NEVER block event capture.
    envelope_id         UUID,                           -- Event envelope identifier (if present)
    correlation_id      UUID,                           -- Request correlation ID for distributed tracing
    event_type          TEXT,                           -- Event type discriminator (e.g., "NodeRegistered")
    source              TEXT,                           -- Event source identifier (e.g., "node-registration-orchestrator")

    -- =========================================================================
    -- Timestamps
    -- =========================================================================
    event_timestamp     TIMESTAMPTZ,                    -- Timestamp from event payload (nullable - not all events have timestamps)
    ledger_written_at   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),  -- When this ledger entry was written (guaranteed)

    -- =========================================================================
    -- Constraints
    -- =========================================================================
    -- Idempotency constraint: Ensures each Kafka message is recorded exactly once.
    -- On consumer restart, duplicate inserts will fail gracefully (ON CONFLICT DO NOTHING).
    CONSTRAINT uk_event_ledger_kafka_position UNIQUE (topic, partition, kafka_offset)
);

-- =============================================================================
-- INDEXES
-- =============================================================================
-- Optimized for common query patterns: correlation lookups, event type filtering,
-- and time-range scans.

-- Index 1: Correlation ID lookups (partial - only indexed when NOT NULL)
-- Use case: Distributed tracing, finding all events for a request
-- Partial index reduces storage overhead for events without correlation_id
CREATE INDEX IF NOT EXISTS idx_event_ledger_correlation_id
    ON event_ledger (correlation_id)
    WHERE correlation_id IS NOT NULL;

-- Index 2: Event type filtering (partial - only indexed when NOT NULL)
-- Use case: Finding all events of a specific type for replay or analysis
-- Partial index excludes malformed events without event_type
CREATE INDEX IF NOT EXISTS idx_event_ledger_event_type
    ON event_ledger (event_type)
    WHERE event_type IS NOT NULL;

-- Index 3: Timestamp ordering with fallback
-- Use case: Time-range queries for replay, audit, and debugging
-- COALESCE ensures consistent ordering even when event_timestamp is NULL
-- Falls back to ledger_written_at which is always populated
CREATE INDEX IF NOT EXISTS idx_event_ledger_event_timestamp
    ON event_ledger (COALESCE(event_timestamp, ledger_written_at));

-- Index 4: Topic + Timestamp composite (for topic-scoped time-range queries)
-- Use case: Replay events from a specific topic within a time window
-- Common pattern: "replay all registration events from the last hour"
CREATE INDEX IF NOT EXISTS idx_event_ledger_topic_timestamp
    ON event_ledger (topic, COALESCE(event_timestamp, ledger_written_at));

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE event_ledger IS
    'Durable, append-only ledger of all events consumed from Kafka. Provides audit trail, replay capability, and idempotency guarantees.';

COMMENT ON COLUMN event_ledger.ledger_entry_id IS
    'Auto-generated UUID primary key for this ledger entry';

COMMENT ON COLUMN event_ledger.topic IS
    'Kafka topic name (TEXT to support long namespaced topics)';

COMMENT ON COLUMN event_ledger.partition IS
    'Kafka partition number';

COMMENT ON COLUMN event_ledger.kafka_offset IS
    'Kafka offset within the partition (idempotency key component)';

COMMENT ON COLUMN event_ledger.event_key IS
    'Raw Kafka message key as BYTEA (nullable - not all events have keys)';

COMMENT ON COLUMN event_ledger.event_value IS
    'Raw Kafka message value as BYTEA (required - the actual event payload)';

COMMENT ON COLUMN event_ledger.onex_headers IS
    'ONEX-specific headers extracted from Kafka headers as JSONB';

COMMENT ON COLUMN event_ledger.envelope_id IS
    'Event envelope identifier extracted from payload (nullable for malformed events)';

COMMENT ON COLUMN event_ledger.correlation_id IS
    'Request correlation ID for distributed tracing (nullable for legacy events)';

COMMENT ON COLUMN event_ledger.event_type IS
    'Event type discriminator extracted from payload (nullable for malformed events)';

COMMENT ON COLUMN event_ledger.source IS
    'Event source identifier (nullable for events without source metadata)';

COMMENT ON COLUMN event_ledger.event_timestamp IS
    'Timestamp from event payload (nullable - falls back to ledger_written_at)';

COMMENT ON COLUMN event_ledger.ledger_written_at IS
    'Timestamp when this entry was written to the ledger (guaranteed, used for ordering fallback)';

COMMENT ON CONSTRAINT uk_event_ledger_kafka_position ON event_ledger IS
    'Idempotency constraint: ensures each Kafka message is recorded exactly once';
