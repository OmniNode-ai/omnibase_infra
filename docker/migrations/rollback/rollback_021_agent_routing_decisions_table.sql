-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright 2026 OmniNode Team
--
-- Rollback Migration: rollback_021_agent_routing_decisions_table
-- Rolls Back: forward/021_create_agent_routing_decisions_table.sql
-- Created: 2026-01-31
--
-- WARNING: THIS ROLLBACK WILL PERMANENTLY DELETE ALL DATA IN THE agent_routing_decisions TABLE.
-- This operation is IRREVERSIBLE. Ensure you have backups before proceeding.
-- All agent routing decision observability data will be lost.
--
-- ============================================================================

-- Drop indexes first
DROP INDEX IF EXISTS idx_agent_routing_decisions_created_at;

-- Drop the table
DROP TABLE IF EXISTS agent_routing_decisions;
