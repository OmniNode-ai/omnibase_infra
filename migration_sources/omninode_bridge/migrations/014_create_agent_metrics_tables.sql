-- ================================================================
-- Migration 014: Create Agent Performance Metrics Tables
-- ================================================================
-- Description: Create partitioned tables for agent performance metrics
-- Author: System
-- Date: 2025-11-06
-- Dependencies: uuid-ossp extension (created in earlier migrations)

-- ================================================================
-- Agent Routing Metrics Table
-- ================================================================
CREATE TABLE IF NOT EXISTS agent_routing_metrics (
    id BIGSERIAL,
    metric_id VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(20) NOT NULL CHECK (metric_type IN ('timing', 'counter', 'gauge', 'rate')),
    value NUMERIC(10, 4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    tags JSONB DEFAULT '{}',
    agent_id VARCHAR(100),
    correlation_id VARCHAR(100),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create initial partitions (3 months)
CREATE TABLE IF NOT EXISTS agent_routing_metrics_2025_11 PARTITION OF agent_routing_metrics
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE IF NOT EXISTS agent_routing_metrics_2025_12 PARTITION OF agent_routing_metrics
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS agent_routing_metrics_2026_01 PARTITION OF agent_routing_metrics
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_routing_metrics_timestamp ON agent_routing_metrics (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_routing_metrics_metric_name ON agent_routing_metrics (metric_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_routing_metrics_tags ON agent_routing_metrics USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_routing_metrics_correlation ON agent_routing_metrics (correlation_id) WHERE correlation_id IS NOT NULL;

-- ================================================================
-- Agent State Metrics Table
-- ================================================================
CREATE TABLE IF NOT EXISTS agent_state_metrics (
    id BIGSERIAL,
    metric_id VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(20) NOT NULL CHECK (metric_type IN ('timing', 'counter', 'gauge', 'rate')),
    value NUMERIC(10, 4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    tags JSONB DEFAULT '{}',
    agent_id VARCHAR(100),
    correlation_id VARCHAR(100),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create initial partitions (3 months)
CREATE TABLE IF NOT EXISTS agent_state_metrics_2025_11 PARTITION OF agent_state_metrics
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE IF NOT EXISTS agent_state_metrics_2025_12 PARTITION OF agent_state_metrics
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS agent_state_metrics_2026_01 PARTITION OF agent_state_metrics
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_state_metrics_timestamp ON agent_state_metrics (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_state_metrics_metric_name ON agent_state_metrics (metric_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_state_metrics_tags ON agent_state_metrics USING GIN (tags);

-- ================================================================
-- Agent Coordination Metrics Table
-- ================================================================
CREATE TABLE IF NOT EXISTS agent_coordination_metrics (
    id BIGSERIAL,
    metric_id VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(20) NOT NULL CHECK (metric_type IN ('timing', 'counter', 'gauge', 'rate')),
    value NUMERIC(10, 4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    tags JSONB DEFAULT '{}',
    agent_id VARCHAR(100),
    correlation_id VARCHAR(100),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create initial partitions (3 months)
CREATE TABLE IF NOT EXISTS agent_coordination_metrics_2025_11 PARTITION OF agent_coordination_metrics
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE IF NOT EXISTS agent_coordination_metrics_2025_12 PARTITION OF agent_coordination_metrics
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS agent_coordination_metrics_2026_01 PARTITION OF agent_coordination_metrics
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_coord_metrics_timestamp ON agent_coordination_metrics (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_coord_metrics_metric_name ON agent_coordination_metrics (metric_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_coord_metrics_tags ON agent_coordination_metrics USING GIN (tags);

-- ================================================================
-- Agent Workflow Metrics Table
-- ================================================================
CREATE TABLE IF NOT EXISTS agent_workflow_metrics (
    id BIGSERIAL,
    metric_id VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(20) NOT NULL CHECK (metric_type IN ('timing', 'counter', 'gauge', 'rate')),
    value NUMERIC(10, 4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    tags JSONB DEFAULT '{}',
    agent_id VARCHAR(100),
    correlation_id VARCHAR(100),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create initial partitions (3 months)
CREATE TABLE IF NOT EXISTS agent_workflow_metrics_2025_11 PARTITION OF agent_workflow_metrics
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE IF NOT EXISTS agent_workflow_metrics_2025_12 PARTITION OF agent_workflow_metrics
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS agent_workflow_metrics_2026_01 PARTITION OF agent_workflow_metrics
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_workflow_metrics_timestamp ON agent_workflow_metrics (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_metrics_metric_name ON agent_workflow_metrics (metric_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_metrics_tags ON agent_workflow_metrics USING GIN (tags);

-- ================================================================
-- Agent Quorum Metrics Table
-- ================================================================
CREATE TABLE IF NOT EXISTS agent_quorum_metrics (
    id BIGSERIAL,
    metric_id VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(20) NOT NULL CHECK (metric_type IN ('timing', 'counter', 'gauge', 'rate')),
    value NUMERIC(10, 4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    tags JSONB DEFAULT '{}',
    agent_id VARCHAR(100),
    correlation_id VARCHAR(100),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create initial partitions (3 months)
CREATE TABLE IF NOT EXISTS agent_quorum_metrics_2025_11 PARTITION OF agent_quorum_metrics
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE IF NOT EXISTS agent_quorum_metrics_2025_12 PARTITION OF agent_quorum_metrics
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS agent_quorum_metrics_2026_01 PARTITION OF agent_quorum_metrics
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_quorum_metrics_timestamp ON agent_quorum_metrics (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_quorum_metrics_metric_name ON agent_quorum_metrics (metric_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_quorum_metrics_tags ON agent_quorum_metrics USING GIN (tags);

-- ================================================================
-- Comments
-- ================================================================
COMMENT ON TABLE agent_routing_metrics IS 'Agent routing performance metrics (partitioned by month)';
COMMENT ON TABLE agent_state_metrics IS 'Agent state operation metrics (partitioned by month)';
COMMENT ON TABLE agent_coordination_metrics IS 'Agent coordination metrics (partitioned by month)';
COMMENT ON TABLE agent_workflow_metrics IS 'Agent workflow execution metrics (partitioned by month)';
COMMENT ON TABLE agent_quorum_metrics IS 'AI quorum validation metrics (partitioned by month)';

-- ================================================================
-- Notes
-- ================================================================
-- 1. Partitions are created for 3 months initially
-- 2. Add new partitions monthly via cron job or manual process
-- 3. Retention policy: 90 days (drop partitions older than 90 days)
-- 4. All tables use same schema for consistency
-- 5. GIN indexes on JSONB tags for fast filtering
