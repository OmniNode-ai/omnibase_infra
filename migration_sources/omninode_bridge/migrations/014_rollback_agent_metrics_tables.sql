-- ================================================================
-- Migration 014 Rollback: Drop Agent Performance Metrics Tables
-- ================================================================

-- Drop indexes first
DROP INDEX IF EXISTS idx_routing_metrics_timestamp;
DROP INDEX IF EXISTS idx_routing_metrics_metric_name;
DROP INDEX IF EXISTS idx_routing_metrics_tags;
DROP INDEX IF EXISTS idx_routing_metrics_correlation;

DROP INDEX IF EXISTS idx_state_metrics_timestamp;
DROP INDEX IF EXISTS idx_state_metrics_metric_name;
DROP INDEX IF EXISTS idx_state_metrics_tags;

DROP INDEX IF EXISTS idx_coord_metrics_timestamp;
DROP INDEX IF EXISTS idx_coord_metrics_metric_name;
DROP INDEX IF EXISTS idx_coord_metrics_tags;

DROP INDEX IF EXISTS idx_workflow_metrics_timestamp;
DROP INDEX IF EXISTS idx_workflow_metrics_metric_name;
DROP INDEX IF EXISTS idx_workflow_metrics_tags;

DROP INDEX IF EXISTS idx_quorum_metrics_timestamp;
DROP INDEX IF EXISTS idx_quorum_metrics_metric_name;
DROP INDEX IF EXISTS idx_quorum_metrics_tags;

-- Drop partitions (will cascade to parent table)
DROP TABLE IF EXISTS agent_routing_metrics_2025_11;
DROP TABLE IF EXISTS agent_routing_metrics_2025_12;
DROP TABLE IF EXISTS agent_routing_metrics_2026_01;

DROP TABLE IF EXISTS agent_state_metrics_2025_11;
DROP TABLE IF EXISTS agent_state_metrics_2025_12;
DROP TABLE IF EXISTS agent_state_metrics_2026_01;

DROP TABLE IF EXISTS agent_coordination_metrics_2025_11;
DROP TABLE IF EXISTS agent_coordination_metrics_2025_12;
DROP TABLE IF EXISTS agent_coordination_metrics_2026_01;

DROP TABLE IF EXISTS agent_workflow_metrics_2025_11;
DROP TABLE IF EXISTS agent_workflow_metrics_2025_12;
DROP TABLE IF EXISTS agent_workflow_metrics_2026_01;

DROP TABLE IF EXISTS agent_quorum_metrics_2025_11;
DROP TABLE IF EXISTS agent_quorum_metrics_2025_12;
DROP TABLE IF EXISTS agent_quorum_metrics_2026_01;

-- Drop parent tables
DROP TABLE IF EXISTS agent_routing_metrics CASCADE;
DROP TABLE IF EXISTS agent_state_metrics CASCADE;
DROP TABLE IF EXISTS agent_coordination_metrics CASCADE;
DROP TABLE IF EXISTS agent_workflow_metrics CASCADE;
DROP TABLE IF EXISTS agent_quorum_metrics CASCADE;
