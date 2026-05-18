-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration 078: add data_provenance column to projection tables (OMN-11201).
--
-- Tracks the origin of projected data so consumers can reason about data quality.
-- Valid values: demo_seeded, demo_projected_shortcut, measured, estimated, unknown

ALTER TABLE contracts ADD COLUMN IF NOT EXISTS data_provenance TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE topics ADD COLUMN IF NOT EXISTS data_provenance TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE registration_projections ADD COLUMN IF NOT EXISTS data_provenance TEXT NOT NULL DEFAULT 'unknown';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_contracts_data_provenance'
    ) THEN
        ALTER TABLE contracts
            ADD CONSTRAINT chk_contracts_data_provenance
            CHECK (data_provenance IN ('demo_seeded', 'demo_projected_shortcut', 'measured', 'estimated', 'unknown'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_topics_data_provenance'
    ) THEN
        ALTER TABLE topics
            ADD CONSTRAINT chk_topics_data_provenance
            CHECK (data_provenance IN ('demo_seeded', 'demo_projected_shortcut', 'measured', 'estimated', 'unknown'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_registration_projections_data_provenance'
    ) THEN
        ALTER TABLE registration_projections
            ADD CONSTRAINT chk_registration_projections_data_provenance
            CHECK (data_provenance IN ('demo_seeded', 'demo_projected_shortcut', 'measured', 'estimated', 'unknown'));
    END IF;
END $$;

COMMENT ON COLUMN contracts.data_provenance IS
    'Origin of projected data. Values: demo_seeded, demo_projected_shortcut, measured, estimated, unknown. OMN-11201.';
COMMENT ON COLUMN topics.data_provenance IS
    'Origin of projected data. Values: demo_seeded, demo_projected_shortcut, measured, estimated, unknown. OMN-11201.';
COMMENT ON COLUMN registration_projections.data_provenance IS
    'Origin of projected data. Values: demo_seeded, demo_projected_shortcut, measured, estimated, unknown. OMN-11201.';
