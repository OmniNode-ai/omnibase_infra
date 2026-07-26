-- =============================================================================
-- MIGRATION: 001_create_delivery_replay_canary_projection (managed-staging canary)
-- =============================================================================
-- Ticket: OMN-14737 (Lane B / task B12 — provision the canary Postgres surface)
-- Version: 1.0.0
-- Epoch:   mstg1 (managed-staging canary epoch — see B7 catalog namespace)
--
-- PURPOSE:
--   Landing/projection table for the one-tenant managed-staging delegation
--   canary (B11). It is the durable surface the canary's *correlation-linked
--   readback* writes to — "without this there is nowhere for the projected row
--   to land" (OMN-14737).
--
--   The column set is DERIVED, field-by-field, from the deterministic output of
--   the B6 delivery/replay projection compute node landed in OMN-14726
--   (omnimarket node `node_delivery_replay_projection_compute`, merged 2026-07-18):
--       output model  ModelReplayProjection
--       module        omnimarket.nodes.node_delivery_replay_projection_compute
--                     .models.model_replay_projection
--   Each column below cites the exact model field it materializes so the seam is
--   matched, not approximated (CLAUDE.md "define and match seams").
--
-- TARGET DATABASE (hand-created — RDS omninode-dev-postgres has DBName=null):
--   A DEDICATED canary logical DB, default name `omninode_canary_mstg1`
--   (epoch-scoped so a re-run — mstg2, mstg3 — provisions a provably disjoint
--   surface, mirroring the fresh-topic-namespace doctrine of the B7 epoch).
--   This migration is DB-agnostic: it creates the table in whichever DB the
--   operator targets with `psql -d <db>`. It is NOT part of the flat
--   docker/migrations/forward/ set and is NEVER auto-applied via
--   docker-entrypoint-initdb.d — it is applied MANUALLY, one operator-gated step
--   at a time, per the provisioning runbook:
--       docs/runbooks/managed-staging-canary-postgres-provisioning.md
--
-- PROVISIONING SAFETY:
--   This file is HELD FOR OPERATOR. Authoring or reading it authorizes no live
--   DB/table/cred mutation. No AWS/RDS mutation occurs by committing it.
--
-- ROLLBACK:
--   See docker/migrations/canary/rollback/rollback_001_delivery_replay_canary_projection.sql
--   (drops the table; DB-level teardown is B13 — managed-staging-canary-teardown-rollback.md).
--
-- Idempotent: all statements use IF NOT EXISTS / OR REPLACE; safe to re-apply.
-- =============================================================================

CREATE TABLE IF NOT EXISTS delivery_replay_canary_projection (
    -- ModelReplayProjection.correlation_id (UUID | None in the pure-compute
    -- model). The canary readback is *correlation-linked*, so for the landing
    -- contract it is REQUIRED and is the natural key: it is the ONLY key the
    -- delegation legs carry on the wire (same rationale as delegation_workflow_
    -- state, migration 090). PRIMARY KEY on correlation_id alone — it is already
    -- a globally-unique UUID; a composite key would permit an ambiguous read.
    correlation_id          UUID            NOT NULL,

    -- ModelReplayProjection.projection_checksum (str, min_length=1). B6 always
    -- emits a lowercase sha256 hex digest (64 chars) over the ordered fold of
    -- the sequence; content- and order-sensitive. Stored verbatim for equality
    -- against the replay expectation. CHECK matches the model contract
    -- (non-empty) rather than pinning 64 chars, so a future checksum-algorithm
    -- change in B6 does not silently reject a contract-valid value.
    projection_checksum     TEXT            NOT NULL,

    -- ModelReplayCursor.token (str, min_length=1) — the canonical, stable string
    -- form of the terminal cursor used for equality/divergence comparison.
    -- Order-insensitive; orthogonal to projection_checksum.
    cursor_token            TEXT            NOT NULL,

    -- ModelReplayCursor.positions (tuple[ModelDeliveryPosition, ...]) — the
    -- per-(topic, partition) terminal max offsets, sorted deterministically.
    -- Materialized as a JSON array of {topic, partition, offset} objects; the
    -- cursor is a nested structure, so it is stored opaquely rather than
    -- flattened into a side table (single-tenant canary; no join required).
    cursor_positions        JSONB           NOT NULL DEFAULT '[]'::jsonb,

    -- ModelReplayCursor.event_count (int, ge=0) — events consumed to reach the
    -- cursor. Expected identical to event_count below (B6 derives both from the
    -- same replayed sequence length); both are persisted to mirror the emitted
    -- model shape exactly.
    cursor_event_count      BIGINT          NOT NULL DEFAULT 0,

    -- ModelReplayProjection.event_count (int, ge=0) — number of events in the
    -- replayed sequence.
    event_count             BIGINT          NOT NULL DEFAULT 0,

    -- ModelReplayProjection.compared (bool) — whether an expected result was
    -- supplied and compared against.
    compared                BOOLEAN         NOT NULL DEFAULT FALSE,

    -- ModelReplayProjection.diverged (bool) — whether the computed projection
    -- diverged from the expected result. Only meaningful when compared = TRUE.
    diverged                BOOLEAN         NOT NULL DEFAULT FALSE,

    -- ModelReplayProjection.divergence_reasons (tuple[str, ...]) — the signals
    -- that diverged, a subset of {"projection_checksum", "cursor"}. Empty when
    -- not diverged. Materialized as a JSON string array.
    divergence_reasons      JSONB           NOT NULL DEFAULT '[]'::jsonb,

    -- Operational landing metadata (not part of ModelReplayProjection).
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- tenant-scoping: pending decision #5 (Adil), out of one-tenant canary scope.
    -- The one-tenant canary has exactly one synthetic tenant, so this table
    -- carries NO tenant_id column and NO multi-tenant partitioning. If this
    -- projection is later promoted to a tenant-scoped surface, that is a
    -- multi-tenant data-model call (decision #5) — do not add it here.

    CONSTRAINT pk_delivery_replay_canary_projection
        PRIMARY KEY (correlation_id),
    CONSTRAINT valid_projection_checksum
        CHECK (char_length(projection_checksum) > 0),
    CONSTRAINT valid_cursor_token
        CHECK (char_length(cursor_token) > 0),
    CONSTRAINT valid_cursor_event_count
        CHECK (cursor_event_count >= 0),
    CONSTRAINT valid_event_count
        CHECK (event_count >= 0),
    -- divergence_reasons must be a JSON array (subset of the two signal labels).
    CONSTRAINT valid_divergence_reasons
        CHECK (jsonb_typeof(divergence_reasons) = 'array'),
    -- cursor_positions must be a JSON array of position objects.
    CONSTRAINT valid_cursor_positions
        CHECK (jsonb_typeof(cursor_positions) = 'array')
);

-- Soak/staleness scans and the B10 monitoring window read by recency.
CREATE INDEX IF NOT EXISTS ix_delivery_replay_canary_projection_updated_at
    ON delivery_replay_canary_projection (updated_at);

-- Divergence monitoring during the canary soak: a partial index over the rows
-- that failed the replay-determinism check keeps the B10 divergence signal cheap.
CREATE INDEX IF NOT EXISTS ix_delivery_replay_canary_projection_diverged
    ON delivery_replay_canary_projection (correlation_id)
    WHERE diverged;

-- Keep updated_at fresh on UPSERT (the readback is an idempotent upsert keyed on
-- correlation_id). Mirrors the trigger pattern used by delegation_workflow_state
-- (migration 090).
CREATE OR REPLACE FUNCTION refresh_delivery_replay_canary_projection_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_delivery_replay_canary_projection_updated_at
    ON delivery_replay_canary_projection;
CREATE TRIGGER trg_delivery_replay_canary_projection_updated_at
    BEFORE UPDATE ON delivery_replay_canary_projection
    FOR EACH ROW
    EXECUTE FUNCTION refresh_delivery_replay_canary_projection_updated_at();
