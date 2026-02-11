-- Rollback: forward/028_absorb_omniclaude_columns.sql
-- Description: Remove absorbed omniclaude columns and agent_activity_realtime view (OMN-2057)
-- Created: 2026-02-11
--
-- NOTE ON claude_session_* TABLES (INTENTIONALLY NOT RESTORED)
-- =============================================================
-- The forward migration drops claude_session_snapshots, claude_session_prompts,
-- claude_session_tools, and claude_session_event_idempotency. This rollback does
-- NOT recreate them. This is intentional for three reasons:
--
--   1. DB-SPLIT ownership transfer: These tables belong to the omniclaude service
--      database now (DB-SPLIT-07, OMN-2057). Their schema and lifecycle are owned
--      by omniclaude4/sql/migrations/, not by omnibase_infra.
--
--   2. No omnibase_infra data: The tables held omniclaude-specific session data
--      (Claude Code session snapshots, prompts, tool invocations). No omnibase_infra
--      service ever read from or wrote to them. They existed here only because
--      omniclaude historically shared a single database with omnibase_infra.
--
--   3. Restoring would be incorrect: Recreating the tables here would reintroduce
--      orphaned schema that conflicts with omniclaude's authoritative copies in its
--      own database. The correct restore path is through omniclaude's migrations.
--

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
