-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright 2026 OmniNode Team
--
-- Rollback Migration: rollback_023_router_performance_metrics_table
-- Rolls Back: forward/023_create_router_performance_metrics_table.sql
-- Created: 2026-01-31
--
-- WARNING: THIS ROLLBACK WILL PERMANENTLY DELETE ALL DATA IN THE router_performance_metrics TABLE.
-- This operation is IRREVERSIBLE. Ensure you have backups before proceeding.
-- All router performance metrics observability data will be lost.
--
-- ============================================================================

-- Drop indexes first
DROP INDEX IF EXISTS idx_router_performance_metrics_created_at;

-- Drop the table
DROP TABLE IF EXISTS router_performance_metrics;
