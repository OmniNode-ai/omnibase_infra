-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT

-- OMN-10341 Task 10: waste detection findings projection.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'severity_type') THEN
        CREATE TYPE severity_type AS ENUM ('LOW', 'MEDIUM', 'HIGH');
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS waste_findings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255),
    rule_id VARCHAR(64) NOT NULL,
    severity severity_type NOT NULL,
    waste_tokens BIGINT NOT NULL,
    waste_cost_usd NUMERIC(14, 6) NOT NULL,
    evidence JSONB,
    evidence_hash CHAR(64) NOT NULL,
    dedup_key VARCHAR(512) NOT NULL,
    recommendation TEXT,
    repo_name VARCHAR(255),
    machine_id VARCHAR(64),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (dedup_key)
);

CREATE INDEX IF NOT EXISTS idx_waste_findings_rule_id
    ON waste_findings (rule_id);
CREATE INDEX IF NOT EXISTS idx_waste_findings_session
    ON waste_findings (session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_waste_findings_detected_at
    ON waste_findings (detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_waste_findings_evidence_hash
    ON waste_findings (evidence_hash);
