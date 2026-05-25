-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration: 025_fix_llm_delegation_call_log_date_index
-- Description: Remove idx_llm_delegation_call_log_date introduced in migration 024.
--              (created_at::date) is not IMMUTABLE in PostgreSQL 16, so the functional
--              index fails to build. The plain created_at index (also in 024) already
--              serves range and equality queries on the timestamp column; the date cast
--              index provides no additional query coverage.
-- Ticket: OMN-11997
--
-- Idempotency:
--   DROP INDEX IF EXISTS — safe to re-apply.

DROP INDEX IF EXISTS idx_llm_delegation_call_log_date;
