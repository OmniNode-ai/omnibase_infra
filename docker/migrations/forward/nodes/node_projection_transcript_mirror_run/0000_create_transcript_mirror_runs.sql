-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- OMN-22260: the transcript mirror run projection (plan task T0a).
-- Owner: omnimarket.nodes.node_projection_transcript_mirror_run
-- Target database: omnidash_analytics; physical schema: omninode_internal.
--
-- One row per mirror attempt, keyed by (run_id, attempt_id). The reducer folds
-- the mirror's events into it: a start event inserts the row in state 'started'
-- with no finish time; the terminal event sets the terminal state, the finish
-- time and the attempt's counts. A terminal state is never moved back, and the
-- reducer is the only writer.
--
-- The counts are the mirror's ModelMirrorCounts, one column each. They are
-- counts only: no path, no file content and no object key is stored here.
--
--   state 'started'            the attempt began and has no terminal event yet
--   state 'completed'          the attempt ran to the end and put what was new
--   state 'dry-run-completed'  the attempt listed what it would put and put none
--   state 'failed'             the attempt ended without completing
--
-- The orchestrator gates a new run on mirror_interval_s against the newest
-- started_at in this table, so started_at is indexed newest first.
--
-- No statement here drops, deletes or truncates, and no grant is made here:
-- 0001 grants the runtime writer.

BEGIN;

CREATE TABLE IF NOT EXISTS omninode_internal.transcript_mirror_runs (
    run_id             TEXT NOT NULL,
    attempt_id         TEXT NOT NULL,
    state              TEXT NOT NULL,
    started_at         TIMESTAMPTZ NOT NULL,
    finished_at        TIMESTAMPTZ,
    files_seen         BIGINT NOT NULL DEFAULT 0,
    stored             BIGINT NOT NULL DEFAULT 0,
    would_store        BIGINT NOT NULL DEFAULT 0,
    unchanged          BIGINT NOT NULL DEFAULT 0,
    destination_ahead  BIGINT NOT NULL DEFAULT 0,
    refused_stale      BIGINT NOT NULL DEFAULT 0,
    refused_conflict   BIGINT NOT NULL DEFAULT 0,
    source_unreadable  BIGINT NOT NULL DEFAULT 0,
    bytes_put          BIGINT NOT NULL DEFAULT 0,
    CONSTRAINT pk_transcript_mirror_runs PRIMARY KEY (run_id, attempt_id),
    CONSTRAINT ck_transcript_mirror_runs_ids_non_empty
        CHECK (run_id <> '' AND attempt_id <> ''),
    CONSTRAINT ck_transcript_mirror_runs_state
        CHECK (state IN ('started', 'completed', 'dry-run-completed', 'failed')),
    CONSTRAINT ck_transcript_mirror_runs_finish_follows_state
        CHECK ((state = 'started') = (finished_at IS NULL)),
    CONSTRAINT ck_transcript_mirror_runs_counts_non_negative
        CHECK (
            files_seen >= 0 AND stored >= 0 AND would_store >= 0 AND unchanged >= 0
            AND destination_ahead >= 0 AND refused_stale >= 0 AND refused_conflict >= 0
            AND source_unreadable >= 0 AND bytes_put >= 0
        )
);

-- Reconcile a compatible pre-existing transcript_mirror_runs to the declared shape.
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS run_id TEXT;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS attempt_id TEXT;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS state TEXT;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS files_seen BIGINT DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS stored BIGINT DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS would_store BIGINT DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS unchanged BIGINT DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS destination_ahead BIGINT DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS refused_stale BIGINT DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS refused_conflict BIGINT DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS source_unreadable BIGINT DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS bytes_put BIGINT DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_transcript_mirror_runs_started_at
    ON omninode_internal.transcript_mirror_runs (started_at DESC);

COMMIT;
