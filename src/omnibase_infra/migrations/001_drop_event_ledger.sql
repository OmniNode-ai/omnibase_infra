-- Migration: 001_drop_event_ledger.sql
-- Purpose: Rollback for 001_create_event_ledger.sql
-- Author: ONEX Infrastructure Team
-- Date: 2026-01-29
--
-- WARNING: This migration is DESTRUCTIVE. All data in event_ledger will be lost.
-- Use only in development or when intentionally resetting the ledger.
--
-- This rollback drops:
--   - Table: event_ledger
--   - Constraint: uk_event_ledger_kafka_position (dropped with table)
--   - Index: idx_event_ledger_correlation_id (dropped with table)
--   - Index: idx_event_ledger_event_type (dropped with table)
--   - Index: idx_event_ledger_event_timestamp (dropped with table)
--   - Index: idx_event_ledger_topic_timestamp (dropped with table)
--   - All table and column comments (dropped with table)

DROP TABLE IF EXISTS event_ledger CASCADE;
