-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright 2026 OmniNode Team
--
-- Rollback Migration: rollback_022_agent_transformation_events_table
-- Rolls Back: forward/022_create_agent_transformation_events_table.sql
-- Created: 2026-01-31
--
-- WARNING: THIS ROLLBACK WILL PERMANENTLY DELETE ALL DATA IN THE agent_transformation_events TABLE.
-- This operation is IRREVERSIBLE. Ensure you have backups before proceeding.
-- All agent transformation event observability data will be lost.
--
-- ============================================================================

-- Drop indexes first
DROP INDEX IF EXISTS idx_agent_transformation_events_created_at;

-- Drop the table
DROP TABLE IF EXISTS agent_transformation_events;
