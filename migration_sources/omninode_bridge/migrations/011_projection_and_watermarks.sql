-- Migration: 011_projection_and_watermarks
-- Description: Create projection store and watermark tracking for eventual consistency
-- Dependencies: None (new tables)
-- Created: 2025-10-21
-- Workstream: Pure Reducer Refactor - Wave 1, Workstream 1B

-- ============================================================================
-- Workflow Projection Table
-- ============================================================================
-- Purpose: Fast read-optimized projection of workflow state
-- Read pattern: High frequency, low latency requirements (<10ms)
-- Write pattern: Async materialization from StateCommitted events
-- Consistency: Eventual (with fallback to canonical store)

CREATE TABLE IF NOT EXISTS workflow_projection (
    -- Primary key
    workflow_key TEXT PRIMARY KEY,

    -- Version tracking (for eventual consistency gating)
    version BIGINT NOT NULL,

    -- Workflow state tag (FSM state)
    tag TEXT NOT NULL,

    -- Last action applied (for debugging and tracing)
    last_action TEXT,

    -- Multi-tenant isolation
    namespace TEXT NOT NULL,

    -- Temporal tracking
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Custom query indexes (future optimization)
    -- Stores denormalized indexes for fast filtering
    -- Example: {"priority": "high", "team": "platform"}
    indices JSONB,

    -- Additional metadata
    -- Stores projection-specific data not in canonical state
    -- Example: {"last_error": "...", "retry_count": 3}
    extras JSONB
);

-- ============================================================================
-- Indexes for workflow_projection
-- ============================================================================

-- Fast namespace-based queries (multi-tenant isolation)
CREATE INDEX IF NOT EXISTS idx_projection_namespace
    ON workflow_projection(namespace);

-- Fast tag-based queries (FSM state filtering)
-- Example: Find all FAILED workflows
CREATE INDEX IF NOT EXISTS idx_projection_tag
    ON workflow_projection(tag);

-- Combined index for common query patterns
-- Example: Find PROCESSING workflows in "production" namespace
CREATE INDEX IF NOT EXISTS idx_projection_namespace_tag
    ON workflow_projection(namespace, tag);

-- Version-based queries (for lag detection)
CREATE INDEX IF NOT EXISTS idx_projection_version
    ON workflow_projection(version DESC);

-- GIN index for custom indices JSONB field (for future optimization)
CREATE INDEX IF NOT EXISTS idx_projection_indices_gin
    ON workflow_projection USING GIN(indices)
    WHERE indices IS NOT NULL AND indices != '{}'::jsonb;

-- ============================================================================
-- Projection Watermarks Table
-- ============================================================================
-- Purpose: Track materialization progress per partition
-- Pattern: One row per Kafka partition or shard
-- Usage: Detect projection lag, prevent reprocessing

CREATE TABLE IF NOT EXISTS projection_watermarks (
    -- Partition identifier (Kafka partition ID or shard ID)
    partition_id TEXT PRIMARY KEY,

    -- Last successfully processed offset (quoted because 'offset' is reserved keyword)
    "offset" BIGINT NOT NULL DEFAULT 0,

    -- Timestamp of last update (for lag detection)
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraint: offset must be non-negative
    CONSTRAINT watermark_offset_non_negative CHECK ("offset" >= 0)
);

-- ============================================================================
-- Comments for documentation
-- ============================================================================

COMMENT ON TABLE workflow_projection IS 'Read-optimized projection of workflow state (eventual consistency)';
COMMENT ON COLUMN workflow_projection.workflow_key IS 'Unique workflow identifier (matches canonical store)';
COMMENT ON COLUMN workflow_projection.version IS 'Version number for eventual consistency gating';
COMMENT ON COLUMN workflow_projection.tag IS 'Workflow FSM state (PENDING, PROCESSING, COMPLETED, FAILED)';
COMMENT ON COLUMN workflow_projection.last_action IS 'Last action type applied (for debugging)';
COMMENT ON COLUMN workflow_projection.namespace IS 'Namespace for multi-tenant isolation';
COMMENT ON COLUMN workflow_projection.updated_at IS 'Last update timestamp (auto-managed)';
COMMENT ON COLUMN workflow_projection.indices IS 'Custom query indexes (JSONB, for future optimization)';
COMMENT ON COLUMN workflow_projection.extras IS 'Additional projection-specific metadata (JSONB)';

COMMENT ON TABLE projection_watermarks IS 'Tracks materialization progress per partition for projection consistency';
COMMENT ON COLUMN projection_watermarks.partition_id IS 'Kafka partition ID or shard identifier';
COMMENT ON COLUMN projection_watermarks."offset" IS 'Last successfully processed event offset';
COMMENT ON COLUMN projection_watermarks.updated_at IS 'Timestamp of last watermark update (for lag detection)';
