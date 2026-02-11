-- Migration: 028_absorb_omniclaude_columns
-- Description: Absorb columns that omniclaude added via cross-repo ALTER TABLE (OMN-2057)
-- Created: 2026-02-11
--
-- Purpose: For existing environments that already ran migrations 020-025, this adds
-- the project context columns that omniclaude previously injected via ALTER TABLE in
-- omniclaude4/sql/migrations/001_add_project_context_to_observability_tables.sql.
--
-- New installs already have these columns from the updated CREATE TABLE statements.
-- This migration uses ADD COLUMN IF NOT EXISTS to be safe for both paths.
--
-- Also creates the agent_activity_realtime view (previously owned by omniclaude).
--
-- Rollback: See rollback/rollback_028_absorb_omniclaude_columns.sql

-- ============================================================================
-- AGENT_ACTIONS: Add project_path, project_name, working_directory
-- ============================================================================

ALTER TABLE agent_actions ADD COLUMN IF NOT EXISTS project_path TEXT;
ALTER TABLE agent_actions ADD COLUMN IF NOT EXISTS project_name VARCHAR(255);
ALTER TABLE agent_actions ADD COLUMN IF NOT EXISTS working_directory TEXT;

-- ============================================================================
-- AGENT_ROUTING_DECISIONS: Add project_path, project_name, claude_session_id
-- ============================================================================

ALTER TABLE agent_routing_decisions ADD COLUMN IF NOT EXISTS project_path TEXT;
ALTER TABLE agent_routing_decisions ADD COLUMN IF NOT EXISTS project_name VARCHAR(255);
ALTER TABLE agent_routing_decisions ADD COLUMN IF NOT EXISTS claude_session_id VARCHAR(255);

-- ============================================================================
-- AGENT_TRANSFORMATION_EVENTS: Add project_path, project_name, claude_session_id
-- ============================================================================

ALTER TABLE agent_transformation_events ADD COLUMN IF NOT EXISTS project_path TEXT;
ALTER TABLE agent_transformation_events ADD COLUMN IF NOT EXISTS project_name VARCHAR(255);
ALTER TABLE agent_transformation_events ADD COLUMN IF NOT EXISTS claude_session_id VARCHAR(255);

-- ============================================================================
-- AGENT_DETECTION_FAILURES: Add project_path, project_name, claude_session_id
-- ============================================================================

ALTER TABLE agent_detection_failures ADD COLUMN IF NOT EXISTS project_path TEXT;
ALTER TABLE agent_detection_failures ADD COLUMN IF NOT EXISTS project_name VARCHAR(255);
ALTER TABLE agent_detection_failures ADD COLUMN IF NOT EXISTS claude_session_id VARCHAR(255);

-- ============================================================================
-- AGENT_EXECUTION_LOGS: Add project_path, project_name, claude_session_id, terminal_id
-- ============================================================================

ALTER TABLE agent_execution_logs ADD COLUMN IF NOT EXISTS project_path TEXT;
ALTER TABLE agent_execution_logs ADD COLUMN IF NOT EXISTS project_name VARCHAR(255);
ALTER TABLE agent_execution_logs ADD COLUMN IF NOT EXISTS claude_session_id VARCHAR(255);
ALTER TABLE agent_execution_logs ADD COLUMN IF NOT EXISTS terminal_id VARCHAR(255);

-- ============================================================================
-- AGENT_ACTIVITY_REALTIME VIEW
-- ============================================================================
-- Unified real-time view across all agent observability tables.
-- Previously owned by omniclaude, now absorbed into omnibase_infra.

CREATE OR REPLACE VIEW agent_activity_realtime AS

SELECT
    'routing_decision' AS event_type,
    id AS event_id,
    correlation_id,
    selected_agent AS agent_name,
    NULL::VARCHAR(50) AS action_type,
    NULL::VARCHAR(255) AS action_name,
    created_at,
    project_path,
    project_name,
    claude_session_id,
    NULL::VARCHAR(255) AS terminal_id,
    NULL::TEXT AS working_directory,
    metadata
FROM agent_routing_decisions
WHERE created_at > NOW() - INTERVAL '24 hours'

UNION ALL

SELECT
    'execution_log' AS event_type,
    execution_id AS event_id,
    correlation_id,
    agent_name,
    NULL::VARCHAR(50) AS action_type,
    NULL::VARCHAR(255) AS action_name,
    created_at,
    project_path,
    project_name,
    claude_session_id,
    terminal_id,
    NULL::TEXT AS working_directory,
    metadata
FROM agent_execution_logs
WHERE updated_at > NOW() - INTERVAL '24 hours'

UNION ALL

SELECT
    'action' AS event_type,
    id AS event_id,
    correlation_id,
    agent_name,
    action_type,
    action_name,
    created_at,
    project_path,
    project_name,
    NULL::VARCHAR(255) AS claude_session_id,
    NULL::VARCHAR(255) AS terminal_id,
    working_directory,
    metadata
FROM agent_actions
WHERE created_at > NOW() - INTERVAL '24 hours'

UNION ALL

SELECT
    'transformation' AS event_type,
    id AS event_id,
    correlation_id,
    source_agent AS agent_name,
    NULL::VARCHAR(50) AS action_type,
    NULL::VARCHAR(255) AS action_name,
    created_at,
    project_path,
    project_name,
    claude_session_id,
    NULL::VARCHAR(255) AS terminal_id,
    NULL::TEXT AS working_directory,
    metadata
FROM agent_transformation_events
WHERE created_at > NOW() - INTERVAL '24 hours'

UNION ALL

SELECT
    'detection_failure' AS event_type,
    correlation_id AS event_id,
    correlation_id,
    fallback_used AS agent_name,
    NULL::VARCHAR(50) AS action_type,
    NULL::VARCHAR(255) AS action_name,
    created_at,
    project_path,
    project_name,
    claude_session_id,
    NULL::VARCHAR(255) AS terminal_id,
    NULL::TEXT AS working_directory,
    metadata
FROM agent_detection_failures
WHERE created_at > NOW() - INTERVAL '24 hours';

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON VIEW agent_activity_realtime IS 'Unified real-time agent activity view across all observability tables (OMN-2057). Shows last 24 hours of activity.';
