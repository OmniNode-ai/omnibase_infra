-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration 078: create runtime_manifests append-only projection table (OMN-11197).
--
-- Append-only: handler INSERTS only, never UPDATEs.
-- The unique index on (runtime_profile, topology_hash, started_at) deduplicates
-- repeated startup events from the same process boot.

CREATE TABLE IF NOT EXISTS runtime_manifests (
    id SERIAL PRIMARY KEY,
    runtime_profile TEXT NOT NULL,
    contract_hash TEXT NOT NULL,
    topology_hash TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    contracts JSONB NOT NULL DEFAULT '[]',
    owned_command_topics JSONB NOT NULL DEFAULT '[]',
    subscribed_event_topics JSONB NOT NULL DEFAULT '[]',
    handlers JSONB NOT NULL DEFAULT '[]',
    skipped_contracts JSONB NOT NULL DEFAULT '[]',
    failed_contracts JSONB NOT NULL DEFAULT '[]',
    ownership_violations JSONB NOT NULL DEFAULT '[]',
    image_digest TEXT,
    started_at TIMESTAMPTZ NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_runtime_manifests_dedup
    ON runtime_manifests(runtime_profile, topology_hash, started_at);

CREATE INDEX IF NOT EXISTS idx_runtime_manifests_profile
    ON runtime_manifests(runtime_profile);

CREATE INDEX IF NOT EXISTS idx_runtime_manifests_latest
    ON runtime_manifests(runtime_profile, recorded_at DESC);
