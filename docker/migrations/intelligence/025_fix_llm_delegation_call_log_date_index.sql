-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration: 025_fix_llm_delegation_call_log_date_index
-- Description: Corrective migration for idx_llm_delegation_call_log_date.
--              Migration 024 used date(created_at) which is STABLE not IMMUTABLE
--              in PostgreSQL, causing CREATE INDEX to fail on some deployments.
--              This migration drops the broken index (if it exists) and
--              recreates it using the IMMUTABLE (created_at::date) cast form.
-- Ticket: OMN-11966
--
-- Idempotency:
--   DROP INDEX IF EXISTS + CREATE INDEX IF NOT EXISTS — safe to re-apply.

-- Drop the STABLE-function-based index if it was successfully created
DROP INDEX IF EXISTS idx_llm_delegation_call_log_date;

-- Recreate with IMMUTABLE ::date cast form
CREATE INDEX IF NOT EXISTS idx_llm_delegation_call_log_date
    ON llm_delegation_call_log ((created_at::date));
