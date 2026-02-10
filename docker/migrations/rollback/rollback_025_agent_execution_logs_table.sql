-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright 2026 OmniNode Team
--
-- Rollback Migration: rollback_025_agent_execution_logs_table
-- Rolls Back: forward/025_create_agent_execution_logs_table.sql
-- Created: 2026-01-31
--
-- WARNING: THIS ROLLBACK WILL PERMANENTLY DELETE ALL DATA IN THE agent_execution_logs TABLE.
-- This operation is IRREVERSIBLE. Ensure you have backups before proceeding.
-- All agent execution log observability data will be lost.
--
-- ============================================================================

-- Drop trigger first (depends on both table and function)
DROP TRIGGER IF EXISTS trigger_agent_execution_logs_updated_at ON agent_execution_logs;

-- Drop function (no longer needed after trigger is gone)
DROP FUNCTION IF EXISTS update_agent_execution_logs_updated_at();

-- Drop indexes
DROP INDEX IF EXISTS idx_agent_execution_logs_updated_at;

-- Drop the table
DROP TABLE IF EXISTS agent_execution_logs;
