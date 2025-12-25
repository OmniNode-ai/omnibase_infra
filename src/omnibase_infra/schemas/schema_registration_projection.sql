-- ONEX Registration Projection Schema
-- Ticket: OMN-944 (F1: Implement Registration Projection Schema)
-- Version: 1.0.0
--
-- Design Notes:
--   - (entity_id, domain) = composite primary key for multi-domain support
--   - Indexes optimized for orchestrator query patterns:
--     1. Deadline scanning (ack_deadline, liveness_deadline)
--     2. State filtering (current_state)
--     3. Domain + state queries
--     4. Idempotency checks (last_applied_event_id)
--   - JSONB for capabilities (flexible schema, GIN indexable)
--   - All timestamps are timezone-aware (TIMESTAMPTZ)
--   - This schema is idempotent (IF NOT EXISTS used throughout)
--
-- Usage:
--   Execute this SQL file to create or update the projection schema.
--   The schema is designed to be re-run safely (idempotent).
--
-- Related Tickets:
--   - OMN-940 (F0): Projector Execution Model
--   - OMN-932 (C2): Durable Timeout Handling
--   - OMN-888 (C1): Registration Orchestrator

-- =============================================================================
-- ENUM TYPES
-- =============================================================================

-- Create enum type for registration states (matches EnumRegistrationState in Python)
-- Note: PostgreSQL enums are immutable - to add values, use ALTER TYPE ... ADD VALUE
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'registration_state') THEN
        CREATE TYPE registration_state AS ENUM (
            'pending_registration',
            'accepted',
            'awaiting_ack',
            'rejected',
            'ack_timed_out',
            'ack_received',
            'active',
            'liveness_expired'
        );
    END IF;
END$$;

-- =============================================================================
-- MAIN TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS registration_projections (
    -- Identity (composite primary key)
    entity_id UUID NOT NULL,
    domain VARCHAR(128) NOT NULL DEFAULT 'registration',

    -- FSM State
    current_state registration_state NOT NULL,

    -- Node Information (snapshot at registration time)
    node_type VARCHAR(64) NOT NULL,
    node_version VARCHAR(32) NOT NULL DEFAULT '1.0.0',
    capabilities JSONB NOT NULL DEFAULT '{}',

    -- Timeout Deadlines (for durable timeout handling per C2)
    ack_deadline TIMESTAMPTZ,
    liveness_deadline TIMESTAMPTZ,
    last_heartbeat_at TIMESTAMPTZ,

    -- Timeout Emission Markers (for deduplication per C2)
    -- These prevent emitting duplicate timeout events during replay
    ack_timeout_emitted_at TIMESTAMPTZ,
    liveness_timeout_emitted_at TIMESTAMPTZ,

    -- Idempotency and Ordering (per F1 requirements)
    -- These fields enable replay correctness and stale update rejection
    last_applied_event_id UUID NOT NULL,
    last_applied_offset BIGINT NOT NULL DEFAULT 0,
    last_applied_sequence BIGINT,
    last_applied_partition VARCHAR(128),

    -- Timestamps
    registered_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,

    -- Tracing
    correlation_id UUID,

    -- Constraints
    PRIMARY KEY (entity_id, domain),
    CONSTRAINT valid_offset CHECK (last_applied_offset >= 0),
    CONSTRAINT valid_sequence CHECK (last_applied_sequence IS NULL OR last_applied_sequence >= 0),
    CONSTRAINT valid_node_type CHECK (node_type IN ('effect', 'compute', 'reducer', 'orchestrator'))
);

-- =============================================================================
-- INDEXES FOR ORCHESTRATION QUERIES
-- =============================================================================

-- Index for ack deadline scanning (C2: orchestrator queries for overdue ack)
-- Partial index: only index rows with non-null ack_deadline
-- Query pattern: SELECT * FROM registration_projections
--                WHERE ack_deadline < :now
--                AND ack_timeout_emitted_at IS NULL
--                AND current_state = 'awaiting_ack'
CREATE INDEX IF NOT EXISTS idx_registration_ack_deadline
    ON registration_projections (ack_deadline)
    WHERE ack_deadline IS NOT NULL;

-- Index for liveness deadline scanning (C2: orchestrator queries for overdue liveness)
-- Partial index: only index active nodes with non-null liveness_deadline
-- Query pattern: SELECT * FROM registration_projections
--                WHERE liveness_deadline < :now
--                AND liveness_timeout_emitted_at IS NULL
--                AND current_state = 'active'
CREATE INDEX IF NOT EXISTS idx_registration_liveness_deadline
    ON registration_projections (liveness_deadline)
    WHERE liveness_deadline IS NOT NULL;

-- Index for state filtering (common orchestrator query pattern)
-- Query pattern: SELECT * FROM registration_projections WHERE current_state = 'active'
CREATE INDEX IF NOT EXISTS idx_registration_current_state
    ON registration_projections (current_state);

-- Index for state + domain filtering (multi-domain queries)
-- Query pattern: SELECT * FROM registration_projections
--                WHERE domain = :domain AND current_state = :state
CREATE INDEX IF NOT EXISTS idx_registration_domain_state
    ON registration_projections (domain, current_state);

-- Index for idempotency checks (fast lookup by event_id)
-- Query pattern: SELECT 1 FROM registration_projections
--                WHERE last_applied_event_id = :event_id
CREATE INDEX IF NOT EXISTS idx_registration_last_event_id
    ON registration_projections (last_applied_event_id);

-- Index for capability queries (GIN index on JSONB)
-- Query pattern: SELECT * FROM registration_projections
--                WHERE capabilities @> '{"postgres": true}'
CREATE INDEX IF NOT EXISTS idx_registration_capabilities
    ON registration_projections USING GIN (capabilities);

-- Composite index for deadline + emission marker scans
-- Optimizes the most common timeout check query
CREATE INDEX IF NOT EXISTS idx_registration_ack_timeout_scan
    ON registration_projections (ack_deadline, ack_timeout_emitted_at)
    WHERE ack_deadline IS NOT NULL AND ack_timeout_emitted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_registration_liveness_timeout_scan
    ON registration_projections (current_state, liveness_deadline, liveness_timeout_emitted_at)
    WHERE current_state = 'active' AND liveness_deadline IS NOT NULL AND liveness_timeout_emitted_at IS NULL;

-- =============================================================================
-- TABLE AND COLUMN COMMENTS
-- =============================================================================

COMMENT ON TABLE registration_projections IS
    'Materialized registration state for orchestrator queries (OMN-944/F1). '
    'Stores current FSM state, timeout deadlines, and idempotency tracking fields.';

COMMENT ON COLUMN registration_projections.entity_id IS
    'Node UUID - partition key for per-entity ordering. Combined with domain forms primary key.';

COMMENT ON COLUMN registration_projections.domain IS
    'Domain namespace for multi-domain support. Default: "registration".';

COMMENT ON COLUMN registration_projections.current_state IS
    'Current FSM state. Used by orchestrators to determine workflow progression.';

COMMENT ON COLUMN registration_projections.ack_deadline IS
    'Deadline for node acknowledgment. Orchestrator emits timeout event when passed.';

COMMENT ON COLUMN registration_projections.liveness_deadline IS
    'Deadline for next heartbeat. Orchestrator emits liveness expired event when passed.';

COMMENT ON COLUMN registration_projections.last_heartbeat_at IS
    'Timestamp of last received heartbeat from the node. Updated on each HeartbeatReceived event.';

COMMENT ON COLUMN registration_projections.ack_timeout_emitted_at IS
    'Marker indicating ack timeout event was already emitted. Prevents duplicates during replay.';

COMMENT ON COLUMN registration_projections.liveness_timeout_emitted_at IS
    'Marker indicating liveness timeout event was already emitted. Prevents duplicates during replay.';

COMMENT ON COLUMN registration_projections.last_applied_event_id IS
    'message_id of last applied event - used for idempotency checks.';

COMMENT ON COLUMN registration_projections.last_applied_offset IS
    'Kafka offset of last applied event - canonical ordering for replay correctness.';

COMMENT ON COLUMN registration_projections.last_applied_sequence IS
    'Sequence number for non-Kafka transports (optional). Used when offset is not applicable.';

COMMENT ON COLUMN registration_projections.last_applied_partition IS
    'Kafka partition of last applied event. Used with offset for global ordering.';

-- =============================================================================
-- INDEX STRATEGY DOCUMENTATION
-- =============================================================================
--
-- This schema defines multiple indexes for different query patterns. Some indexes
-- may appear redundant but serve distinct purposes due to partial WHERE clauses.
--
-- ACK DEADLINE INDEXES:
-- ---------------------
-- 1. idx_registration_ack_deadline (line 104-106):
--    - Single-column index on ack_deadline
--    - WHERE: ack_deadline IS NOT NULL
--    - Use case: General queries filtering by ack_deadline regardless of emission status
--    - Example: SELECT * FROM registration_projections WHERE ack_deadline < :now
--
-- 2. idx_registration_ack_timeout_scan (line 143-145):
--    - Composite index on (ack_deadline, ack_timeout_emitted_at)
--    - WHERE: ack_deadline IS NOT NULL AND ack_timeout_emitted_at IS NULL
--    - Use case: Timeout scanner queries that need un-emitted timeouts only
--    - Example: SELECT * FROM registration_projections
--               WHERE ack_deadline < :now AND ack_timeout_emitted_at IS NULL
--
-- WHY BOTH INDEXES EXIST:
-- The partial WHERE clauses differ significantly:
--   - idx_registration_ack_deadline includes ALL rows with non-null deadlines
--   - idx_registration_ack_timeout_scan ONLY includes rows awaiting timeout emission
--
-- PostgreSQL's query planner selects the most selective index. For timeout scans
-- (the common orchestrator query), idx_registration_ack_timeout_scan is optimal
-- because it pre-filters to only candidate rows. For other deadline queries that
-- don't filter on emission status, idx_registration_ack_deadline is used.
--
-- LIVENESS DEADLINE INDEXES:
-- --------------------------
-- Similar rationale applies to liveness indexes:
--   - idx_registration_liveness_deadline: All rows with liveness deadlines
--   - idx_registration_liveness_timeout_scan: Only active nodes awaiting emission
--
-- =============================================================================
-- UPSERT ORDERING ENFORCEMENT PATTERN
-- =============================================================================
--
-- The projector (projector_registration.py) uses an atomic INSERT-or-UPDATE with
-- sequence validation to enforce event ordering and reject stale updates.
--
-- KEY CONCEPT: Stale Update Rejection
-- ------------------------------------
-- "Stale" means the incoming event is older than what we've already processed.
-- The upsert's WHERE clause compares sequence/offset numbers to ensure we never
-- regress projection state. If the incoming event is older, PostgreSQL skips
-- the UPDATE entirely - the existing row remains unchanged.
--
-- ORDERING MODES (mutually exclusive):
-- ------------------------------------
-- 1. KAFKA-BASED ORDERING (partition IS NOT NULL):
--    Uses last_applied_offset for ordering within a partition.
--    Kafka guarantees ordering within a partition, so (partition, offset)
--    provides a total order for events affecting the same entity.
--
-- 2. SEQUENCE-BASED ORDERING (partition IS NULL):
--    Falls back to last_applied_sequence for non-Kafka transports (e.g., HTTP).
--    Used when the event source doesn't provide Kafka-style partition/offset.
--
-- IDEMPOTENCY FIELDS:
-- -------------------
-- - last_applied_event_id: Message UUID for exact duplicate detection
-- - last_applied_offset: Kafka offset for Kafka-based ordering
-- - last_applied_sequence: Generic sequence for non-Kafka ordering
-- - last_applied_partition: Kafka partition for partitioned ordering context
--
-- These fields together enable:
-- 1. Replay correctness: Replaying events produces identical projection state
-- 2. Stale rejection: Out-of-order events don't regress state
-- 3. Idempotency: Duplicate events are handled gracefully
--
-- See projector_registration.py for the complete upsert SQL implementation.
