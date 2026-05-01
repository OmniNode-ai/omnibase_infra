-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Rollback migration 076: remove savings_estimates provenance columns.

ALTER TABLE savings_estimates
    DROP CONSTRAINT IF EXISTS chk_savings_estimates_source_payload_hash,
    DROP CONSTRAINT IF EXISTS chk_savings_estimates_estimation_method,
    DROP CONSTRAINT IF EXISTS chk_savings_estimates_usage_source;

ALTER TABLE savings_estimates
    DROP COLUMN IF EXISTS source_payload_hash,
    DROP COLUMN IF EXISTS estimation_method,
    DROP COLUMN IF EXISTS usage_source;
