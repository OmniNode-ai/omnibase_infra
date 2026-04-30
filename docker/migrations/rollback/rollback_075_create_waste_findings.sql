-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT

-- Rollback OMN-10341 Task 10 waste detection findings projection.

DROP INDEX IF EXISTS idx_waste_findings_evidence_hash;
DROP INDEX IF EXISTS idx_waste_findings_detected_at;
DROP INDEX IF EXISTS idx_waste_findings_session;
DROP INDEX IF EXISTS idx_waste_findings_rule_id;

DROP TABLE IF EXISTS waste_findings;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'severity_type')
        AND NOT EXISTS (
            SELECT 1
            FROM pg_attribute a
            JOIN pg_type t ON a.atttypid = t.oid
            WHERE t.typname = 'severity_type'
              AND a.attnum > 0
              AND NOT a.attisdropped
        ) THEN
        DROP TYPE severity_type;
    END IF;
END$$;
