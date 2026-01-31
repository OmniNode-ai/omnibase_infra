-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright 2026 OmniNode Team
--
-- Rollback Migration: 024_rollback_agent_detection_failures_table
-- Rolls Back: 024_create_agent_detection_failures_table.sql
-- Created: 2026-01-31
--
-- WARNING: THIS ROLLBACK WILL PERMANENTLY DELETE ALL DATA IN THE agent_detection_failures TABLE.
-- This operation is IRREVERSIBLE. Ensure you have backups before proceeding.
-- All agent detection failure observability data will be lost.
--
-- ============================================================================

-- Drop indexes first
DROP INDEX IF EXISTS idx_agent_detection_failures_created_at;

-- Drop the table
DROP TABLE IF EXISTS agent_detection_failures;
