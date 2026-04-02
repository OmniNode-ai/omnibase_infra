-- Rollback: rollback_003_drop_baselines_tables.sql
-- Purpose: Drop baselines comparison tables created by migration 003
-- Author: ONEX Infrastructure Team
-- Date: 2026-02-18
-- Ticket: OMN-2305

DROP TABLE IF EXISTS baselines_breakdown CASCADE;
DROP TABLE IF EXISTS baselines_trend CASCADE;
DROP TABLE IF EXISTS baselines_comparisons CASCADE;
