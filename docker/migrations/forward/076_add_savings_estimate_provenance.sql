-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration 076: add provenance columns for dispatch-derived savings estimates (OMN-10388).

ALTER TABLE savings_estimates
    ADD COLUMN IF NOT EXISTS usage_source TEXT NOT NULL DEFAULT 'UNKNOWN',
    ADD COLUMN IF NOT EXISTS estimation_method TEXT,
    ADD COLUMN IF NOT EXISTS source_payload_hash TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_savings_estimates_usage_source'
    ) THEN
        ALTER TABLE savings_estimates
            ADD CONSTRAINT chk_savings_estimates_usage_source
            CHECK (usage_source IN ('MEASURED', 'ESTIMATED', 'UNKNOWN'));
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_savings_estimates_estimation_method'
    ) THEN
        ALTER TABLE savings_estimates
            ADD CONSTRAINT chk_savings_estimates_estimation_method
            CHECK (
                (usage_source = 'ESTIMATED' AND estimation_method IS NOT NULL)
                OR usage_source <> 'ESTIMATED'
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_savings_estimates_source_payload_hash'
    ) THEN
        ALTER TABLE savings_estimates
            ADD CONSTRAINT chk_savings_estimates_source_payload_hash
            CHECK (
                (usage_source = 'MEASURED' AND source_payload_hash IS NOT NULL)
                OR usage_source <> 'MEASURED'
            );
    END IF;
END $$;

COMMENT ON COLUMN savings_estimates.usage_source IS
    'Cost provenance source for dispatch-derived savings estimates. OMN-10388.';
COMMENT ON COLUMN savings_estimates.estimation_method IS
    'Estimator identifier when savings are derived from estimated dispatch usage. OMN-10388.';
COMMENT ON COLUMN savings_estimates.source_payload_hash IS
    'Stable source payload hash when savings are derived from measured dispatch usage. OMN-10388.';
