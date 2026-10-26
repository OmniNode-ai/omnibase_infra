-- =============================================================================
-- MIGRATION: per-session compaction current context
-- =============================================================================
-- Owner:   omnimarket.nodes.node_projection_compaction_current_context
-- Version: 1.0.0
--
-- One row per session: the joinable fold state (maximal boundary chains and
-- distinct boundary observations) and its resolved read view (the current
-- boundary, what it installed, the activations retired at later boundaries,
-- or explicit conflicts). The fold joins in any order, so the writer replaces
-- the row with the joined state; state is current, no_boundary or conflict.
-- The schema is provisioned by infrastructure, as in the alert-channel template.
-- =============================================================================

CREATE TABLE IF NOT EXISTS omninode_internal.compaction_current_context (
    session_id                 TEXT NOT NULL,
    state                      TEXT NOT NULL,
    boundary_uuid              TEXT,
    boundary_kind              TEXT,
    activation_id              UUID,
    chains                     JSONB NOT NULL DEFAULT '[]'::jsonb,
    observations               JSONB NOT NULL DEFAULT '[]'::jsonb,
    chain_boundary_uuids       JSONB NOT NULL DEFAULT '[]'::jsonb,
    retirements                JSONB NOT NULL DEFAULT '[]'::jsonb,
    conflicts                  JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_topic               TEXT NOT NULL DEFAULT '',
    projection_cursor          BIGSERIAL NOT NULL,
    projected_at               TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (session_id)
);

-- Shape reconciliation: guarded ADDs precede indexes, as in the template.
-- Existing rows cannot satisfy new NOT NULL columns without defaults.
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS session_id                 TEXT;
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS state                      TEXT;
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS boundary_uuid              TEXT;
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS boundary_kind              TEXT;
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS activation_id              UUID;
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS chains                     JSONB DEFAULT '[]'::jsonb;
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS observations               JSONB DEFAULT '[]'::jsonb;
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS chain_boundary_uuids       JSONB DEFAULT '[]'::jsonb;
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS retirements                JSONB DEFAULT '[]'::jsonb;
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS conflicts                  JSONB DEFAULT '[]'::jsonb;
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS source_topic               TEXT DEFAULT '';
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS projection_cursor          BIGSERIAL;
ALTER TABLE omninode_internal.compaction_current_context
    ADD COLUMN IF NOT EXISTS projected_at               TIMESTAMPTZ;

CREATE UNIQUE INDEX IF NOT EXISTS idx_compaction_current_context_cursor
    ON omninode_internal.compaction_current_context (projection_cursor);

CREATE INDEX IF NOT EXISTS idx_compaction_current_context_state
    ON omninode_internal.compaction_current_context (state);
