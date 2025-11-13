-- Migration: 010_enhance_bridge_states
-- Description: Enhance bridge_states table for reducer-specific aggregation tracking
-- Dependencies: 004_create_bridge_states.sql
-- Created: 2025-10-15

-- Add reducer-specific columns to bridge_states table
-- These fields track detailed aggregation statistics and performance

-- Aggregation statistics
ALTER TABLE bridge_states
ADD COLUMN IF NOT EXISTS total_size_bytes BIGINT DEFAULT 0 CHECK (total_size_bytes >= 0),
ADD COLUMN IF NOT EXISTS unique_file_types TEXT[] DEFAULT '{}',
ADD COLUMN IF NOT EXISTS unique_workflows UUID[] DEFAULT '{}';

-- Aggregation configuration and windowing
ALTER TABLE bridge_states
ADD COLUMN IF NOT EXISTS aggregation_type VARCHAR(50) DEFAULT 'namespace_grouping',
ADD COLUMN IF NOT EXISTS window_start TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS window_end TIMESTAMP WITH TIME ZONE;

-- Performance metrics
ALTER TABLE bridge_states
ADD COLUMN IF NOT EXISTS aggregation_duration_ms INTEGER CHECK (aggregation_duration_ms >= 0),
ADD COLUMN IF NOT EXISTS items_per_second NUMERIC(10,2) CHECK (items_per_second >= 0);

-- Versioning and configuration
ALTER TABLE bridge_states
ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1 CHECK (version >= 1),
ADD COLUMN IF NOT EXISTS configuration JSONB DEFAULT '{}';

-- Performance-optimized indexes
CREATE INDEX IF NOT EXISTS idx_bridge_states_aggregation_type
    ON bridge_states(aggregation_type);

CREATE INDEX IF NOT EXISTS idx_bridge_states_window_start
    ON bridge_states(window_start DESC)
    WHERE window_start IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_bridge_states_window_end
    ON bridge_states(window_end DESC)
    WHERE window_end IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_bridge_states_total_size_bytes
    ON bridge_states(total_size_bytes DESC);

-- GIN index for array columns (fast array containment queries)
CREATE INDEX IF NOT EXISTS idx_bridge_states_unique_file_types_gin
    ON bridge_states USING GIN(unique_file_types)
    WHERE unique_file_types IS NOT NULL AND unique_file_types != '{}';

CREATE INDEX IF NOT EXISTS idx_bridge_states_unique_workflows_gin
    ON bridge_states USING GIN(unique_workflows)
    WHERE unique_workflows IS NOT NULL AND unique_workflows != '{}';

-- GIN index for configuration JSONB
CREATE INDEX IF NOT EXISTS idx_bridge_states_configuration_gin
    ON bridge_states USING GIN(configuration)
    WHERE configuration IS NOT NULL AND configuration != '{}'::jsonb;

-- Compound indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_bridge_states_namespace_aggregation_type
    ON bridge_states(namespace, aggregation_type);

CREATE INDEX IF NOT EXISTS idx_bridge_states_namespace_window
    ON bridge_states(namespace, window_start DESC, window_end DESC)
    WHERE window_start IS NOT NULL;

-- Comments for new columns
COMMENT ON COLUMN bridge_states.total_size_bytes IS 'Total size of all stamped files in bytes';
COMMENT ON COLUMN bridge_states.unique_file_types IS 'Array of unique MIME content types encountered';
COMMENT ON COLUMN bridge_states.unique_workflows IS 'Array of unique workflow UUIDs processed';
COMMENT ON COLUMN bridge_states.aggregation_type IS 'Type of aggregation strategy (namespace_grouping, time_window, file_type_grouping, etc.)';
COMMENT ON COLUMN bridge_states.window_start IS 'Start timestamp of aggregation window (for windowed aggregations)';
COMMENT ON COLUMN bridge_states.window_end IS 'End timestamp of aggregation window (for windowed aggregations)';
COMMENT ON COLUMN bridge_states.aggregation_duration_ms IS 'Duration of aggregation operation in milliseconds';
COMMENT ON COLUMN bridge_states.items_per_second IS 'Aggregation throughput (items processed per second)';
COMMENT ON COLUMN bridge_states.version IS 'Version number for optimistic locking and conflict resolution';
COMMENT ON COLUMN bridge_states.configuration IS 'Aggregation configuration (batch_size, window_size_ms, etc.)';
