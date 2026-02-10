-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright 2026 OmniNode Team
--
-- Rollback Migration: rollback_027_agent_status_events_table
-- Rolls Back: forward/027_create_agent_status_events_table.sql
-- Created: 2026-02-07
--
-- WARNING: THIS ROLLBACK WILL PERMANENTLY DELETE ALL DATA IN THE agent_status_events TABLE.
-- This operation is IRREVERSIBLE. Ensure you have backups before proceeding.
-- All agent status observability data will be lost.
--
-- ============================================================================

-- Drop indexes first
DROP INDEX IF EXISTS idx_agent_status_correlation;
DROP INDEX IF EXISTS idx_agent_status_agent_session;
DROP INDEX IF EXISTS idx_agent_status_state;
DROP INDEX IF EXISTS idx_agent_status_created;
DROP INDEX IF EXISTS idx_agent_status_ordering;

-- Drop the table
DROP TABLE IF EXISTS agent_status_events;
