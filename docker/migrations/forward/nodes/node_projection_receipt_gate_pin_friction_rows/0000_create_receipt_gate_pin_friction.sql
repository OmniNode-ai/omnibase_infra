-- =============================================================================
-- MIGRATION: receipt-gate pin friction rows
-- =============================================================================
-- Ticket:  OMN-20134 (ruling 2026-10-31T00:13:02Z: "we should measure friction overtime")
-- Owner:   omnimarket.nodes.node_projection_receipt_gate_pin_friction
-- Version: 1.0.0
--
-- One row per pin pull request the receipt-gate pin updater opens, or fails to
-- open, keyed by (repository, branch). The branch name carries the core tag, so
-- the key is one caller and one tag. The row keeps the caller's pin kind (sha or
-- tag), the time from the core tag to the pull request, the time to merge, the
-- fix pushes and CI re-runs that followed the updater, and any refusal, so the
-- two pin forms can be compared over a window.
--
-- Only the updater's outcome event creates a row; every later observation (merge,
-- head change, workflow run, red check) updates a row that exists.

CREATE TABLE IF NOT EXISTS omninode_internal.receipt_gate_pin_friction (
    repository                TEXT             NOT NULL,
    branch                    TEXT             NOT NULL,
    pin_kind                  TEXT             NOT NULL,
    core_tag                  TEXT             NOT NULL,
    state                     TEXT             NOT NULL,
    first_seen_at             TIMESTAMPTZ      NOT NULL,
    pr_number                 INTEGER,
    tag_pushed_at             TIMESTAMPTZ,
    pr_opened_at              TIMESTAMPTZ,
    merged_at                 TIMESTAMPTZ,
    closed_at                 TIMESTAMPTZ,
    seconds_tag_to_pr_opened  DOUBLE PRECISION,
    seconds_pr_open_to_merge  DOUBLE PRECISION,
    seconds_tag_to_merge      DOUBLE PRECISION,
    published_head_sha        TEXT,
    head_shas                 TEXT[]           NOT NULL DEFAULT '{}',
    rerun_keys                TEXT[]           NOT NULL DEFAULT '{}',
    refusals                  JSONB            NOT NULL DEFAULT '[]'::jsonb,
    friction_reasons          TEXT[]           NOT NULL DEFAULT '{}',
    last_event_at             TIMESTAMPTZ      NOT NULL,
    updated_at                TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    projection_cursor         BIGSERIAL        NOT NULL,

    CONSTRAINT pk_receipt_gate_pin_friction PRIMARY KEY (repository, branch),
    CONSTRAINT ck_receipt_gate_pin_friction_pin_kind
        CHECK (pin_kind IN ('sha', 'tag')),
    CONSTRAINT ck_receipt_gate_pin_friction_state
        CHECK (state IN ('opened', 'merged', 'closed', 'failed', 'refused'))
);

-- ---- BEGIN OMN-15376 shape reconciliation: receipt_gate_pin_friction ----
-- CREATE TABLE IF NOT EXISTS silently no-ops against a table of this name with a
-- different shape; the guarded adds below converge it onto the shape declared
-- above. On the fresh-create path every one is a no-op.

ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS repository               TEXT;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS branch                   TEXT;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS pin_kind                 TEXT;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS core_tag                 TEXT;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS state                    TEXT;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS first_seen_at            TIMESTAMPTZ;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS pr_number                INTEGER;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS tag_pushed_at            TIMESTAMPTZ;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS pr_opened_at             TIMESTAMPTZ;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS merged_at                TIMESTAMPTZ;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS closed_at                TIMESTAMPTZ;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS seconds_tag_to_pr_opened DOUBLE PRECISION;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS seconds_pr_open_to_merge DOUBLE PRECISION;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS seconds_tag_to_merge     DOUBLE PRECISION;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS published_head_sha       TEXT;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS head_shas                TEXT[] DEFAULT '{}';
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS rerun_keys               TEXT[] DEFAULT '{}';
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS refusals                 JSONB DEFAULT '[]'::jsonb;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS friction_reasons         TEXT[] DEFAULT '{}';
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS last_event_at            TIMESTAMPTZ;
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS updated_at               TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE omninode_internal.receipt_gate_pin_friction
    ADD COLUMN IF NOT EXISTS projection_cursor        BIGSERIAL;

-- ---- END OMN-15376 shape reconciliation: receipt_gate_pin_friction ----

CREATE UNIQUE INDEX IF NOT EXISTS idx_receipt_gate_pin_friction_projection_cursor
    ON omninode_internal.receipt_gate_pin_friction (projection_cursor);

CREATE INDEX IF NOT EXISTS idx_receipt_gate_pin_friction_kind_seen
    ON omninode_internal.receipt_gate_pin_friction (pin_kind, first_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_receipt_gate_pin_friction_pr
    ON omninode_internal.receipt_gate_pin_friction (repository, pr_number);

COMMENT ON TABLE omninode_internal.receipt_gate_pin_friction IS
    'OMN-20134: one row per receipt-gate pin pull request, keyed by caller repository and '
    'pin branch. Written only by node_projection_receipt_gate_pin_friction. touches are '
    'fix pushes (head shas beyond the first) plus CI re-runs (run attempts above 1); the '
    'observations carry no actor, so a touch is a lane, a person or the landing controller.';
