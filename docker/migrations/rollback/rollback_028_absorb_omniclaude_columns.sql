-- Rollback: forward/028_absorb_omniclaude_columns.sql
-- Description: Remove absorbed omniclaude columns and agent_activity_realtime view (OMN-2057)
-- Created: 2026-02-11

-- Drop the view first
DROP VIEW IF EXISTS agent_activity_realtime;

-- agent_actions
ALTER TABLE agent_actions DROP COLUMN IF EXISTS project_path;
ALTER TABLE agent_actions DROP COLUMN IF EXISTS project_name;
ALTER TABLE agent_actions DROP COLUMN IF EXISTS working_directory;

-- agent_routing_decisions
ALTER TABLE agent_routing_decisions DROP COLUMN IF EXISTS project_path;
ALTER TABLE agent_routing_decisions DROP COLUMN IF EXISTS project_name;
ALTER TABLE agent_routing_decisions DROP COLUMN IF EXISTS claude_session_id;

-- agent_transformation_events
ALTER TABLE agent_transformation_events DROP COLUMN IF EXISTS project_path;
ALTER TABLE agent_transformation_events DROP COLUMN IF EXISTS project_name;
ALTER TABLE agent_transformation_events DROP COLUMN IF EXISTS claude_session_id;

-- agent_detection_failures
ALTER TABLE agent_detection_failures DROP COLUMN IF EXISTS project_path;
ALTER TABLE agent_detection_failures DROP COLUMN IF EXISTS project_name;
ALTER TABLE agent_detection_failures DROP COLUMN IF EXISTS claude_session_id;

-- agent_execution_logs
ALTER TABLE agent_execution_logs DROP COLUMN IF EXISTS project_path;
ALTER TABLE agent_execution_logs DROP COLUMN IF EXISTS project_name;
ALTER TABLE agent_execution_logs DROP COLUMN IF EXISTS claude_session_id;
ALTER TABLE agent_execution_logs DROP COLUMN IF EXISTS terminal_id;
