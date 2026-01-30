-- Migration: 005_create_contracts_topics
-- Description: Create tables for contract registry state materialization (OMN-1653)
-- Created: 2026-01-29
--
-- Purpose: NodeContractRegistryReducer materializes contract registry state from Kafka events.
-- These tables store registered contracts and their associated topic suffixes for routing.
--
-- Rollback: DROP TABLE IF EXISTS topics; DROP TABLE IF EXISTS contracts;

-- ============================================================================
-- CONTRACTS TABLE
-- ============================================================================
-- Stores registered ONEX contracts with full YAML for replay capability.
-- Uses Kafka position tracking for exactly-once semantics during event replay.

CREATE TABLE IF NOT EXISTS contracts (
    -- Identity (derived natural key: node_name:major.minor.patch)
    contract_id TEXT PRIMARY KEY,
    node_name TEXT NOT NULL,
    version_major INT NOT NULL,
    version_minor INT NOT NULL,
    version_patch INT NOT NULL,

    -- Contract content
    contract_hash TEXT NOT NULL,
    contract_yaml TEXT NOT NULL,

    -- Lifecycle
    registered_at TIMESTAMPTZ NOT NULL,
    deregistered_at TIMESTAMPTZ,
    last_seen_at TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Dedupe tracking (Kafka position for exactly-once replay)
    last_event_topic TEXT,
    last_event_partition INT,
    last_event_offset BIGINT,

    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Unique constraint for upsert by node identity (also creates implicit index for version lookups)
    CONSTRAINT unique_contract_version UNIQUE (node_name, version_major, version_minor, version_patch)
);

-- Indexes for staleness and active contract queries
CREATE INDEX IF NOT EXISTS idx_contracts_last_seen
    ON contracts(last_seen_at)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_contracts_active
    ON contracts(is_active)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_contracts_node_name
    ON contracts(node_name);

CREATE INDEX IF NOT EXISTS idx_contracts_hash
    ON contracts(contract_hash);

-- ============================================================================
-- TOPICS TABLE
-- ============================================================================
-- Stores topic suffixes referenced by contracts for routing discovery.
-- Uses 5-segment naming: onex.evt.platform.contract-registered.v1
-- Stores SUFFIXES only - environment prefix ({env}.) applied at runtime.

CREATE TABLE IF NOT EXISTS topics (
    -- Identity (composite key: same topic can have both publish and subscribe directions)
    topic_suffix TEXT NOT NULL,
    direction TEXT NOT NULL,

    -- Which contracts reference this topic (JSON array of contract_id strings)
    contract_ids JSONB NOT NULL DEFAULT '[]',

    -- Lifecycle
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    PRIMARY KEY (topic_suffix, direction),
    CONSTRAINT valid_direction CHECK (direction IN ('publish', 'subscribe')),
    CONSTRAINT contract_ids_is_array CHECK (jsonb_typeof(contract_ids) = 'array')
);

-- Indexes for topic queries
CREATE INDEX IF NOT EXISTS idx_topics_direction
    ON topics(direction);

CREATE INDEX IF NOT EXISTS idx_topics_active
    ON topics(is_active)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_topics_last_seen
    ON topics(last_seen_at);

-- GIN index for contract_ids JSONB array queries
CREATE INDEX IF NOT EXISTS idx_topics_contract_ids
    ON topics USING GIN (contract_ids);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger to update updated_at timestamp on contracts
CREATE OR REPLACE FUNCTION update_contracts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_contracts_updated_at ON contracts;
CREATE TRIGGER trigger_contracts_updated_at
    BEFORE UPDATE ON contracts
    FOR EACH ROW
    EXECUTE FUNCTION update_contracts_updated_at();

-- Trigger to update updated_at timestamp on topics
CREATE OR REPLACE FUNCTION update_topics_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_topics_updated_at ON topics;
CREATE TRIGGER trigger_topics_updated_at
    BEFORE UPDATE ON topics
    FOR EACH ROW
    EXECUTE FUNCTION update_topics_updated_at();

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE contracts IS 'ONEX contract registry for NodeContractRegistryReducer (OMN-1653). Stores registered contracts with full YAML for replay.';
COMMENT ON COLUMN contracts.contract_id IS 'Derived natural key: node_name:major.minor.patch (e.g., my-node:1.0.0)';
COMMENT ON COLUMN contracts.node_name IS 'ONEX node name from contract metadata';
COMMENT ON COLUMN contracts.version_major IS 'Semantic version major component';
COMMENT ON COLUMN contracts.version_minor IS 'Semantic version minor component';
COMMENT ON COLUMN contracts.version_patch IS 'Semantic version patch component';
COMMENT ON COLUMN contracts.contract_hash IS 'SHA-256 hash of contract YAML for change detection';
COMMENT ON COLUMN contracts.contract_yaml IS 'Full contract YAML content for replay capability';
COMMENT ON COLUMN contracts.registered_at IS 'Timestamp when contract was first registered';
COMMENT ON COLUMN contracts.deregistered_at IS 'Timestamp when contract was deregistered (NULL if active)';
COMMENT ON COLUMN contracts.last_seen_at IS 'Timestamp of most recent heartbeat or registration event';
COMMENT ON COLUMN contracts.is_active IS 'Whether contract is currently active (soft delete)';
COMMENT ON COLUMN contracts.last_event_topic IS 'Kafka topic of last processed event (for dedupe)';
COMMENT ON COLUMN contracts.last_event_partition IS 'Kafka partition of last processed event (for dedupe)';
COMMENT ON COLUMN contracts.last_event_offset IS 'Kafka offset of last processed event (for dedupe)';

COMMENT ON TABLE topics IS 'Topic suffixes referenced by contracts for routing discovery (OMN-1653). Uses 5-segment naming. Composite key (topic_suffix, direction) allows same topic with both publish and subscribe directions.';
COMMENT ON COLUMN topics.topic_suffix IS 'Topic suffix without environment prefix, e.g., onex.evt.platform.contract-registered.v1';
COMMENT ON COLUMN topics.direction IS 'Whether contract publishes to or subscribes from this topic';
COMMENT ON COLUMN topics.contract_ids IS 'JSON array of contract_id strings that reference this topic';
COMMENT ON COLUMN topics.first_seen_at IS 'Timestamp when topic was first seen in any contract';
COMMENT ON COLUMN topics.last_seen_at IS 'Timestamp when topic was last seen in any contract';
COMMENT ON COLUMN topics.is_active IS 'Whether topic is currently referenced by any active contract';
