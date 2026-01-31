-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright 2026 OmniNode Team
--
-- Rollback Migration: 020_rollback_agent_actions_table
-- Rolls Back: 020_create_agent_actions_table.sql
-- Created: 2026-01-31
--
-- WARNING: THIS ROLLBACK WILL PERMANENTLY DELETE ALL DATA IN THE agent_actions TABLE.
-- This operation is IRREVERSIBLE. Ensure you have backups before proceeding.
-- All agent action observability data will be lost.
--
-- ============================================================================

-- Drop indexes first
DROP INDEX IF EXISTS idx_agent_actions_created_at;

-- Drop the table
DROP TABLE IF EXISTS agent_actions;
