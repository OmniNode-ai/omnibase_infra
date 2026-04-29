-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration 074: create savings_estimates projection table (OMN-10340).

CREATE TABLE IF NOT EXISTS savings_estimates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_timestamp TIMESTAMPTZ NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    model_local VARCHAR(255) NOT NULL,
    model_cloud_baseline VARCHAR(255) NOT NULL,
    local_cost_usd NUMERIC(14, 6) NOT NULL,
    cloud_cost_usd NUMERIC(14, 6) NOT NULL,
    savings_usd NUMERIC(14, 6) NOT NULL,
    repo_name VARCHAR(255),
    machine_id VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT non_negative_local CHECK (local_cost_usd >= 0),
    CONSTRAINT non_negative_cloud CHECK (cloud_cost_usd >= 0),
    CONSTRAINT savings_consistency CHECK (savings_usd = cloud_cost_usd - local_cost_usd),
    CONSTRAINT unique_savings_estimate_event UNIQUE (
        session_id,
        event_timestamp,
        model_local,
        model_cloud_baseline
    )
);

CREATE INDEX IF NOT EXISTS idx_savings_estimates_event_ts
    ON savings_estimates (event_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_savings_estimates_session
    ON savings_estimates (session_id);

CREATE INDEX IF NOT EXISTS idx_savings_estimates_model_local
    ON savings_estimates (model_local);
