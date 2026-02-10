-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright 2026 OmniNode Team
--
-- Rollback Migration: 005_rollback_create_contracts_topics
-- Rolls Back: 005_create_contracts_topics.sql
-- Created: 2026-01-29
--
-- WARNING: THIS ROLLBACK WILL PERMANENTLY DELETE ALL DATA IN THE contracts AND topics TABLES.
-- This operation is IRREVERSIBLE. Ensure you have backups before proceeding.
-- All contract registry and topic routing data will be lost.
--
-- ============================================================================

-- Drop triggers first
DROP TRIGGER IF EXISTS trigger_topics_updated_at ON topics;
DROP TRIGGER IF EXISTS trigger_contracts_updated_at ON contracts;

-- Drop trigger functions
DROP FUNCTION IF EXISTS update_topics_updated_at();
DROP FUNCTION IF EXISTS update_contracts_updated_at();

-- Drop indexes
DROP INDEX IF EXISTS idx_topics_contract_ids;
DROP INDEX IF EXISTS idx_topics_last_seen;
DROP INDEX IF EXISTS idx_topics_active;
DROP INDEX IF EXISTS idx_topics_direction;
DROP INDEX IF EXISTS idx_contracts_hash;
DROP INDEX IF EXISTS idx_contracts_node_name;
DROP INDEX IF EXISTS idx_contracts_active;
DROP INDEX IF EXISTS idx_contracts_last_seen;

-- Drop tables (topics first due to no FK dependency, but order is safe either way)
DROP TABLE IF EXISTS topics;
DROP TABLE IF EXISTS contracts;
