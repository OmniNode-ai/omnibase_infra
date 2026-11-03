-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- OMN-22320: the work-questions projection tables.
-- Owner: omnimarket.nodes.node_projection_work_questions
-- Target database: omnidash_analytics; physical schema: omninode_internal.
--
-- One fold run is one generation. Every row of the five tables carries the
-- generation_id of the run that wrote it and the source versions that run was
-- computed from (generation_source_versions, identical on every row of a
-- generation), so a store can refuse a generation whose source versions are
-- lower than the stored ones and a reader can select one complete generation.
--
--   work_outcome_units            one row per outcome unit, key (generation_id, unit_id)
--   orchestration_style_outcomes  one row per calibration cell, key (generation_id, key)
--   lane_drift_and_stalls         one row per lane kind
--   brief_quality_vs_outcome      one row per brief feature set
--   subagent_waste                one row per session
--
-- NULL means the fold did not know; an unknown figure is never stored as zero.
-- The unit cost is flattened into cost_* columns and the confidence interval of
-- a cell or brief row into interval_low, interval_high and interval_method.
-- No table has a DELETE path in this migration: a generation leaves by the
-- reducer's retention, not by a statement here.

BEGIN;

CREATE TABLE IF NOT EXISTS omninode_internal.work_outcome_units (
    generation_id                      TEXT NOT NULL,
    unit_id                            TEXT NOT NULL,
    unit_kind                          TEXT NOT NULL,
    ticket                             TEXT,
    claimed_at                         TIMESTAMPTZ,
    ended_at                           TIMESTAMPTZ,
    principal                          TEXT,
    state                              TEXT NOT NULL,
    cause                              TEXT,
    unknown_reason                     TEXT,
    work_type                          TEXT,
    band                               INTEGER,
    brief_stratum                      TEXT,
    style                              TEXT NOT NULL,
    model                              TEXT NOT NULL,
    effort                             TEXT NOT NULL,
    cell_key                           TEXT,
    cluster_id                         TEXT,
    shared_evidence                    BOOLEAN NOT NULL DEFAULT FALSE,
    rework                             BOOLEAN NOT NULL DEFAULT FALSE,
    rework_eligible                    BOOLEAN NOT NULL DEFAULT FALSE,
    mis_route                          BOOLEAN NOT NULL DEFAULT FALSE,
    pr_roles                           JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost_measured_amount               NUMERIC,
    cost_measurement_source            TEXT,
    cost_unit                          TEXT,
    cost_unknown                       BOOLEAN NOT NULL DEFAULT TRUE,
    cost_estimate_amount               NUMERIC,
    cost_estimate_unit                 TEXT,
    cost_estimate_price_date           DATE,
    source_versions                    JSONB NOT NULL DEFAULT '{}'::jsonb,
    generation_source_versions         JSONB NOT NULL DEFAULT '{}'::jsonb,
    projected_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_work_outcome_units PRIMARY KEY (generation_id, unit_id),
    CONSTRAINT ck_work_outcome_units_state
        CHECK (state IN ('resolved', 'unresolved', 'pending', 'unknown', 'unverified')),
    CONSTRAINT ck_work_outcome_units_cause
        CHECK (cause IS NULL OR cause IN ('crashed', 'no-terminal', 'bad-outcome', 'no-pr', 'not-merged', 'reverted')),
    CONSTRAINT ck_work_outcome_units_unknown_reason
        CHECK (unknown_reason IS NULL OR unknown_reason IN ('no-claim-or-terminal', 'unjoined', 'ambiguous', 'rootless', 'duplicate-top-level', 'ledger-window-incomplete', 'pr-ownership-indeterminate', 'unclassified', 'merge-unknown', 'revert-unobservable', 'no-end-time')),
    CONSTRAINT ck_work_outcome_units_cost_unknown_follows_measurement
        CHECK ((cost_measured_amount IS NULL AND cost_unknown) OR (cost_measured_amount IS NOT NULL AND NOT cost_unknown AND cost_measurement_source IS NOT NULL AND cost_unit IS NOT NULL)),
    CONSTRAINT ck_work_outcome_units_cost_estimate_complete
        CHECK ((cost_estimate_amount IS NULL AND cost_estimate_unit IS NULL AND cost_estimate_price_date IS NULL) OR (cost_estimate_amount IS NOT NULL AND cost_estimate_unit IS NOT NULL AND cost_estimate_price_date IS NOT NULL))
);

-- Reconcile a compatible pre-existing work_outcome_units to the declared shape.
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS generation_id TEXT NOT NULL;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS unit_id TEXT NOT NULL;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS unit_kind TEXT NOT NULL;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS ticket TEXT;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS ended_at TIMESTAMPTZ;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS principal TEXT;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS state TEXT NOT NULL;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS cause TEXT;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS unknown_reason TEXT;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS work_type TEXT;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS band INTEGER;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS brief_stratum TEXT;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS style TEXT NOT NULL;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS model TEXT NOT NULL;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS effort TEXT NOT NULL;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS cell_key TEXT;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS cluster_id TEXT;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS shared_evidence BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS rework BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS rework_eligible BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS mis_route BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS pr_roles JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS cost_measured_amount NUMERIC;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS cost_measurement_source TEXT;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS cost_unit TEXT;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS cost_unknown BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS cost_estimate_amount NUMERIC;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS cost_estimate_unit TEXT;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS cost_estimate_price_date DATE;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS source_versions JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS generation_source_versions JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE omninode_internal.work_outcome_units
    ADD COLUMN IF NOT EXISTS projected_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE TABLE IF NOT EXISTS omninode_internal.orchestration_style_outcomes (
    generation_id                      TEXT NOT NULL,
    key                                TEXT NOT NULL,
    stratum_kind                       TEXT NOT NULL,
    work_type                          TEXT NOT NULL,
    band                               INTEGER,
    brief_stratum                      TEXT,
    style                              TEXT NOT NULL,
    model                              TEXT NOT NULL,
    effort                             TEXT NOT NULL,
    units                              INTEGER NOT NULL,
    known_units                        INTEGER NOT NULL,
    known_clusters                     INTEGER NOT NULL,
    resolved                           INTEGER NOT NULL,
    unresolved                         INTEGER NOT NULL,
    pending                            INTEGER NOT NULL,
    unknown                            INTEGER NOT NULL,
    unverified                         INTEGER NOT NULL,
    shared_evidence                    INTEGER NOT NULL,
    unknown_share                      DOUBLE PRECISION NOT NULL,
    insufficient                       BOOLEAN NOT NULL,
    resolved_rate                      DOUBLE PRECISION,
    interval_low                       DOUBLE PRECISION,
    interval_high                      DOUBLE PRECISION,
    interval_method                    TEXT,
    median_turns                       DOUBLE PRECISION,
    median_tokens                      DOUBLE PRECISION,
    median_minutes_to_merge            DOUBLE PRECISION,
    retries                            INTEGER NOT NULL,
    refusals                           INTEGER NOT NULL,
    cost_measured_per_resolved         NUMERIC,
    cost_estimated_per_resolved        NUMERIC,
    rework_rate                        DOUBLE PRECISION,
    rework_denominator                 INTEGER NOT NULL,
    mis_route_rate                     DOUBLE PRECISION,
    mis_route_denominator              INTEGER NOT NULL,
    generation_source_versions         JSONB NOT NULL DEFAULT '{}'::jsonb,
    projected_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_orchestration_style_outcomes PRIMARY KEY (generation_id, key),
    CONSTRAINT ck_orchestration_style_outcomes_stratum_kind
        CHECK (stratum_kind IN ('band', 'brief')),
    CONSTRAINT ck_orchestration_style_outcomes_interval_complete
        CHECK ((interval_low IS NULL AND interval_high IS NULL AND interval_method IS NULL) OR (interval_low IS NOT NULL AND interval_high IS NOT NULL AND interval_method IS NOT NULL))
);

-- Reconcile a compatible pre-existing orchestration_style_outcomes to the declared shape.
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS generation_id TEXT NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS key TEXT NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS stratum_kind TEXT NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS work_type TEXT NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS band INTEGER;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS brief_stratum TEXT;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS style TEXT NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS model TEXT NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS effort TEXT NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS units INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS known_units INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS known_clusters INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS resolved INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS unresolved INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS pending INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS unknown INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS unverified INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS shared_evidence INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS unknown_share DOUBLE PRECISION NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS insufficient BOOLEAN NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS resolved_rate DOUBLE PRECISION;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS interval_low DOUBLE PRECISION;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS interval_high DOUBLE PRECISION;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS interval_method TEXT;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS median_turns DOUBLE PRECISION;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS median_tokens DOUBLE PRECISION;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS median_minutes_to_merge DOUBLE PRECISION;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS retries INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS refusals INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS cost_measured_per_resolved NUMERIC;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS cost_estimated_per_resolved NUMERIC;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS rework_rate DOUBLE PRECISION;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS rework_denominator INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS mis_route_rate DOUBLE PRECISION;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS mis_route_denominator INTEGER NOT NULL;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS generation_source_versions JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE omninode_internal.orchestration_style_outcomes
    ADD COLUMN IF NOT EXISTS projected_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE TABLE IF NOT EXISTS omninode_internal.lane_drift_and_stalls (
    generation_id                      TEXT NOT NULL,
    lane_kind                          TEXT NOT NULL,
    units                              INTEGER NOT NULL,
    scope_paths_not_written            INTEGER NOT NULL,
    written_paths_not_merged           INTEGER NOT NULL,
    mean_compactions_failed            DOUBLE PRECISION,
    mean_compactions_passing           DOUBLE PRECISION,
    hook_refusals_by_guard             JSONB NOT NULL DEFAULT '{}'::jsonb,
    permission_decisions_reported_beside INTEGER NOT NULL,
    corrections                        INTEGER NOT NULL,
    friction_class_counts              JSONB NOT NULL DEFAULT '{}'::jsonb,
    generation_source_versions         JSONB NOT NULL DEFAULT '{}'::jsonb,
    projected_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_lane_drift_and_stalls PRIMARY KEY (generation_id, lane_kind)
);

-- Reconcile a compatible pre-existing lane_drift_and_stalls to the declared shape.
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS generation_id TEXT NOT NULL;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS lane_kind TEXT NOT NULL;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS units INTEGER NOT NULL;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS scope_paths_not_written INTEGER NOT NULL;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS written_paths_not_merged INTEGER NOT NULL;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS mean_compactions_failed DOUBLE PRECISION;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS mean_compactions_passing DOUBLE PRECISION;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS hook_refusals_by_guard JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS permission_decisions_reported_beside INTEGER NOT NULL;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS corrections INTEGER NOT NULL;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS friction_class_counts JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS generation_source_versions JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE omninode_internal.lane_drift_and_stalls
    ADD COLUMN IF NOT EXISTS projected_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE TABLE IF NOT EXISTS omninode_internal.brief_quality_vs_outcome (
    generation_id                      TEXT NOT NULL,
    brief_feature_key                  TEXT NOT NULL,
    units                              INTEGER NOT NULL,
    known_units                        INTEGER NOT NULL,
    known_clusters                     INTEGER NOT NULL,
    resolved                           INTEGER NOT NULL,
    insufficient                       BOOLEAN NOT NULL,
    resolved_rate                      DOUBLE PRECISION,
    interval_low                       DOUBLE PRECISION,
    interval_high                      DOUBLE PRECISION,
    interval_method                    TEXT,
    generation_source_versions         JSONB NOT NULL DEFAULT '{}'::jsonb,
    projected_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_brief_quality_vs_outcome PRIMARY KEY (generation_id, brief_feature_key),
    CONSTRAINT ck_brief_quality_vs_outcome_interval_complete
        CHECK ((interval_low IS NULL AND interval_high IS NULL AND interval_method IS NULL) OR (interval_low IS NOT NULL AND interval_high IS NOT NULL AND interval_method IS NOT NULL))
);

-- Reconcile a compatible pre-existing brief_quality_vs_outcome to the declared shape.
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS generation_id TEXT NOT NULL;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS brief_feature_key TEXT NOT NULL;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS units INTEGER NOT NULL;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS known_units INTEGER NOT NULL;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS known_clusters INTEGER NOT NULL;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS resolved INTEGER NOT NULL;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS insufficient BOOLEAN NOT NULL;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS resolved_rate DOUBLE PRECISION;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS interval_low DOUBLE PRECISION;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS interval_high DOUBLE PRECISION;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS interval_method TEXT;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS generation_source_versions JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE omninode_internal.brief_quality_vs_outcome
    ADD COLUMN IF NOT EXISTS projected_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE TABLE IF NOT EXISTS omninode_internal.subagent_waste (
    generation_id                      TEXT NOT NULL,
    session_id                         TEXT NOT NULL,
    unit_id                            TEXT NOT NULL,
    label                              TEXT NOT NULL,
    waste_streams                      INTEGER NOT NULL,
    surviving_streams                  INTEGER NOT NULL,
    answer_use_unknown_streams         INTEGER NOT NULL,
    unknown_streams                    INTEGER NOT NULL,
    pending_streams                    INTEGER NOT NULL,
    no_written_paths_streams           INTEGER NOT NULL,
    generation_source_versions         JSONB NOT NULL DEFAULT '{}'::jsonb,
    projected_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_subagent_waste PRIMARY KEY (generation_id, session_id)
);

-- Reconcile a compatible pre-existing subagent_waste to the declared shape.
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS generation_id TEXT NOT NULL;
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS session_id TEXT NOT NULL;
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS unit_id TEXT NOT NULL;
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS label TEXT NOT NULL;
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS waste_streams INTEGER NOT NULL;
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS surviving_streams INTEGER NOT NULL;
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS answer_use_unknown_streams INTEGER NOT NULL;
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS unknown_streams INTEGER NOT NULL;
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS pending_streams INTEGER NOT NULL;
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS no_written_paths_streams INTEGER NOT NULL;
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS generation_source_versions JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE omninode_internal.subagent_waste
    ADD COLUMN IF NOT EXISTS projected_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

COMMIT;
