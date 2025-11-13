-- Rollback Migration: 010_enhance_bridge_states
-- Description: Remove reducer-specific enhancements from bridge_states
-- Created: 2025-10-15

-- Drop compound indexes first
DROP INDEX IF EXISTS idx_bridge_states_namespace_window;
DROP INDEX IF EXISTS idx_bridge_states_namespace_aggregation_type;

-- Drop GIN indexes
DROP INDEX IF EXISTS idx_bridge_states_configuration_gin;
DROP INDEX IF EXISTS idx_bridge_states_unique_workflows_gin;
DROP INDEX IF EXISTS idx_bridge_states_unique_file_types_gin;

-- Drop single-column indexes
DROP INDEX IF EXISTS idx_bridge_states_total_size_bytes;
DROP INDEX IF EXISTS idx_bridge_states_window_end;
DROP INDEX IF EXISTS idx_bridge_states_window_start;
DROP INDEX IF EXISTS idx_bridge_states_aggregation_type;

-- Remove reducer-specific columns
ALTER TABLE bridge_states
DROP COLUMN IF EXISTS configuration,
DROP COLUMN IF EXISTS version,
DROP COLUMN IF EXISTS items_per_second,
DROP COLUMN IF EXISTS aggregation_duration_ms,
DROP COLUMN IF EXISTS window_end,
DROP COLUMN IF EXISTS window_start,
DROP COLUMN IF EXISTS aggregation_type,
DROP COLUMN IF EXISTS unique_workflows,
DROP COLUMN IF EXISTS unique_file_types,
DROP COLUMN IF EXISTS total_size_bytes;
