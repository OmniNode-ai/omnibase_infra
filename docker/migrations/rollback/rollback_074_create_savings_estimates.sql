-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Rollback migration 074: remove savings_estimates projection table.

DROP INDEX IF EXISTS idx_savings_estimates_model_local;
DROP INDEX IF EXISTS idx_savings_estimates_session;
DROP INDEX IF EXISTS idx_savings_estimates_event_ts;
DROP TABLE IF EXISTS savings_estimates;
