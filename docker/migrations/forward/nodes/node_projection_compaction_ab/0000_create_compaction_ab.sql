-- =============================================================================
-- MIGRATION: compaction A/B live projection and per-build scoreboard series
-- =============================================================================
-- Owner:   omnimarket.nodes.node_projection_compaction_ab
-- One row per (experiment_id, session_id, boundary_uuid). Each column has one
-- writer (see data/live_measurement.yaml); a writer's columns only advance when
-- its (event_time, event_id) is newer than the stored one.
-- =============================================================================

CREATE TABLE IF NOT EXISTS omninode_internal.compaction_ab_rows (
    experiment_id               TEXT NOT NULL,
    session_id                  TEXT NOT NULL,
    boundary_uuid               TEXT NOT NULL,
    arm                         TEXT,
    block_index                 INTEGER,
    carryover                   BOOLEAN,
    assignment_event_time       TIMESTAMPTZ,
    assignment_event_id         TEXT,
    pre_compaction_tokens       BIGINT,
    trigger                     TEXT,
    rulings_in_window           INTEGER,
    open_loops_at_boundary      INTEGER,
    covariates_event_time       TIMESTAMPTZ,
    covariates_event_id         TEXT,
    hl1_events                  INTEGER,
    hl2_acted_items             INTEGER,
    hl2_open_loop_items         INTEGER,
    incident_count              INTEGER,
    hold_scope_violations       INTEGER,
    s5_cost_tokens              BIGINT,
    s3_defects                  INTEGER,
    s3_merged_prs               INTEGER,
    scores_event_time           TIMESTAMPTZ,
    scores_event_id             TEXT,
    PRIMARY KEY (experiment_id, session_id, boundary_uuid)
);

ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS experiment_id TEXT;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS boundary_uuid TEXT;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS arm TEXT;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS block_index INTEGER;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS carryover BOOLEAN;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS assignment_event_time TIMESTAMPTZ;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS assignment_event_id TEXT;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS pre_compaction_tokens BIGINT;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS trigger TEXT;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS rulings_in_window INTEGER;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS open_loops_at_boundary INTEGER;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS covariates_event_time TIMESTAMPTZ;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS covariates_event_id TEXT;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS hl1_events INTEGER;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS hl2_acted_items INTEGER;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS hl2_open_loop_items INTEGER;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS incident_count INTEGER;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS hold_scope_violations INTEGER;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS s5_cost_tokens BIGINT;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS s3_defects INTEGER;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS s3_merged_prs INTEGER;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS scores_event_time TIMESTAMPTZ;
ALTER TABLE omninode_internal.compaction_ab_rows
    ADD COLUMN IF NOT EXISTS scores_event_id TEXT;

-- One row per (candidate_id, node_commit); a build with no candidate_id is refused.
CREATE TABLE IF NOT EXISTS omninode_internal.compaction_ab_build_series (
    candidate_id                    TEXT NOT NULL CHECK (candidate_id <> ''),
    node_commit                     TEXT NOT NULL,
    split                           TEXT NOT NULL CHECK (split = 'calibration'),
    h1_recall                       DOUBLE PRECISION,
    acted_share                     DOUBLE PRECISION,
    planted_recall                  DOUBLE PRECISION,
    contradiction_rate              DOUBLE PRECISION,
    false_memory_rate               DOUBLE PRECISION,
    hold_and_scope_violation_rate   DOUBLE PRECISION,
    cost_ratio                      DOUBLE PRECISION,
    recorded_at                     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (candidate_id, node_commit)
);

ALTER TABLE omninode_internal.compaction_ab_build_series
    ADD COLUMN IF NOT EXISTS candidate_id TEXT;
ALTER TABLE omninode_internal.compaction_ab_build_series
    ADD COLUMN IF NOT EXISTS node_commit TEXT;
ALTER TABLE omninode_internal.compaction_ab_build_series
    ADD COLUMN IF NOT EXISTS split TEXT;
ALTER TABLE omninode_internal.compaction_ab_build_series
    ADD COLUMN IF NOT EXISTS h1_recall DOUBLE PRECISION;
ALTER TABLE omninode_internal.compaction_ab_build_series
    ADD COLUMN IF NOT EXISTS acted_share DOUBLE PRECISION;
ALTER TABLE omninode_internal.compaction_ab_build_series
    ADD COLUMN IF NOT EXISTS planted_recall DOUBLE PRECISION;
ALTER TABLE omninode_internal.compaction_ab_build_series
    ADD COLUMN IF NOT EXISTS contradiction_rate DOUBLE PRECISION;
ALTER TABLE omninode_internal.compaction_ab_build_series
    ADD COLUMN IF NOT EXISTS false_memory_rate DOUBLE PRECISION;
ALTER TABLE omninode_internal.compaction_ab_build_series
    ADD COLUMN IF NOT EXISTS hold_and_scope_violation_rate DOUBLE PRECISION;
ALTER TABLE omninode_internal.compaction_ab_build_series
    ADD COLUMN IF NOT EXISTS cost_ratio DOUBLE PRECISION;
ALTER TABLE omninode_internal.compaction_ab_build_series
    ADD COLUMN IF NOT EXISTS recorded_at TIMESTAMPTZ;
