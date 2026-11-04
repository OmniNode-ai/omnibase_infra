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
-- Existing required values are never inferred: incompatible rows abort this
-- transaction rather than leaving a nullable or unconstrained projection.
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS run_id TEXT NOT NULL;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS attempt_id TEXT NOT NULL;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS state TEXT NOT NULL;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ NOT NULL;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS files_seen BIGINT NOT NULL DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS stored BIGINT NOT NULL DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS would_store BIGINT NOT NULL DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS unchanged BIGINT NOT NULL DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS destination_ahead BIGINT NOT NULL DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS refused_stale BIGINT NOT NULL DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS refused_conflict BIGINT NOT NULL DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS source_unreadable BIGINT NOT NULL DEFAULT 0;
ALTER TABLE omninode_internal.transcript_mirror_runs
    ADD COLUMN IF NOT EXISTS bytes_put BIGINT NOT NULL DEFAULT 0;

-- Existing columns must already have the declared types. Refuse incompatible
-- storage without casting, truncating or otherwise rewriting existing values.
DO $column_types$
DECLARE
    incompatible RECORD;
BEGIN
    SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod) AS actual,
           pg_catalog.format_type(expected.type_oid, -1) AS expected
    INTO incompatible
    FROM pg_catalog.pg_attribute a
    JOIN (VALUES
        ('run_id', 'text'::regtype),
        ('attempt_id', 'text'::regtype),
        ('state', 'text'::regtype),
        ('started_at', 'timestamptz'::regtype),
        ('finished_at', 'timestamptz'::regtype),
        ('files_seen', 'bigint'::regtype),
        ('stored', 'bigint'::regtype),
        ('would_store', 'bigint'::regtype),
        ('unchanged', 'bigint'::regtype),
        ('destination_ahead', 'bigint'::regtype),
        ('refused_stale', 'bigint'::regtype),
        ('refused_conflict', 'bigint'::regtype),
        ('source_unreadable', 'bigint'::regtype),
        ('bytes_put', 'bigint'::regtype)
    ) AS expected(name, type_oid) ON a.attname = expected.name
    WHERE a.attrelid = 'omninode_internal.transcript_mirror_runs'::regclass
      AND NOT a.attisdropped
      AND (a.atttypid <> expected.type_oid OR a.atttypmod <> -1)
    ORDER BY a.attnum
    LIMIT 1;
    IF FOUND THEN
        RAISE EXCEPTION 'transcript_mirror_runs: incompatible column %; expected %, found %',
            incompatible.attname, incompatible.expected, incompatible.actual;
    END IF;
END;
$column_types$;

-- ADD COLUMN IF NOT EXISTS does not repair constraints or defaults on columns
-- that already exist. Apply the same requirements on both creation paths.
ALTER TABLE omninode_internal.transcript_mirror_runs
    ALTER COLUMN run_id SET NOT NULL,
    ALTER COLUMN attempt_id SET NOT NULL,
    ALTER COLUMN state SET NOT NULL,
    ALTER COLUMN started_at SET NOT NULL,
    ALTER COLUMN files_seen SET DEFAULT 0,
    ALTER COLUMN files_seen SET NOT NULL,
    ALTER COLUMN stored SET DEFAULT 0,
    ALTER COLUMN stored SET NOT NULL,
    ALTER COLUMN would_store SET DEFAULT 0,
    ALTER COLUMN would_store SET NOT NULL,
    ALTER COLUMN unchanged SET DEFAULT 0,
    ALTER COLUMN unchanged SET NOT NULL,
    ALTER COLUMN destination_ahead SET DEFAULT 0,
    ALTER COLUMN destination_ahead SET NOT NULL,
    ALTER COLUMN refused_stale SET DEFAULT 0,
    ALTER COLUMN refused_stale SET NOT NULL,
    ALTER COLUMN refused_conflict SET DEFAULT 0,
    ALTER COLUMN refused_conflict SET NOT NULL,
    ALTER COLUMN source_unreadable SET DEFAULT 0,
    ALTER COLUMN source_unreadable SET NOT NULL,
    ALTER COLUMN bytes_put SET DEFAULT 0,
    ALTER COLUMN bytes_put SET NOT NULL;

DO $reconcile$
DECLARE
    expected_check RECORD;
    existing_check RECORD;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'omninode_internal.transcript_mirror_runs'::regclass
          AND contype = 'p'
          AND conkey = ARRAY[
              (SELECT attnum FROM pg_catalog.pg_attribute
               WHERE attrelid = 'omninode_internal.transcript_mirror_runs'::regclass
                 AND attname = 'run_id'),
              (SELECT attnum FROM pg_catalog.pg_attribute
               WHERE attrelid = 'omninode_internal.transcript_mirror_runs'::regclass
                 AND attname = 'attempt_id')
          ]::smallint[]
    ) THEN
        IF EXISTS (
            SELECT 1 FROM pg_catalog.pg_constraint
            WHERE conrelid = 'omninode_internal.transcript_mirror_runs'::regclass
              AND (contype = 'p' OR conname = 'pk_transcript_mirror_runs')
        ) THEN
            RAISE EXCEPTION 'transcript_mirror_runs: incompatible primary key; expected (run_id, attempt_id)';
        END IF;
        ALTER TABLE omninode_internal.transcript_mirror_runs
            ADD CONSTRAINT pk_transcript_mirror_runs PRIMARY KEY (run_id, attempt_id);
    END IF;

    -- Compare PostgreSQL's deparsed expressions, not just constraint names.
    -- A divergent definition is incompatible: refuse atomically rather than
    -- silently keeping weaker checks or removing an existing constraint.
    FOR expected_check IN
        SELECT * FROM (VALUES
            ('ck_transcript_mirror_runs_ids_non_empty',
             $expr$((run_id <> ''::text) AND (attempt_id <> ''::text))$expr$),
            ('ck_transcript_mirror_runs_state',
             $expr$(state = ANY (ARRAY['started'::text, 'completed'::text, 'dry-run-completed'::text, 'failed'::text]))$expr$),
            ('ck_transcript_mirror_runs_finish_follows_state',
             $expr$((state = 'started'::text) = (finished_at IS NULL))$expr$),
            ('ck_transcript_mirror_runs_counts_non_negative',
             $expr$((files_seen >= 0) AND (stored >= 0) AND (would_store >= 0) AND (unchanged >= 0) AND (destination_ahead >= 0) AND (refused_stale >= 0) AND (refused_conflict >= 0) AND (source_unreadable >= 0) AND (bytes_put >= 0))$expr$)
        ) AS checks(name, expression)
    LOOP
        SELECT contype, pg_catalog.pg_get_expr(conbin, conrelid) AS expression
        INTO existing_check
        FROM pg_catalog.pg_constraint
        WHERE conrelid = 'omninode_internal.transcript_mirror_runs'::regclass
          AND conname = expected_check.name;
        IF FOUND AND (
            existing_check.contype <> 'c'
            OR existing_check.expression IS DISTINCT FROM expected_check.expression
        ) THEN
            RAISE EXCEPTION 'transcript_mirror_runs: incompatible constraint %; expected CHECK (%)',
                expected_check.name, expected_check.expression;
        END IF;
    END LOOP;

    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'omninode_internal.transcript_mirror_runs'::regclass
          AND conname = 'ck_transcript_mirror_runs_ids_non_empty'
    ) THEN
        ALTER TABLE omninode_internal.transcript_mirror_runs
            ADD CONSTRAINT ck_transcript_mirror_runs_ids_non_empty
            CHECK (run_id <> '' AND attempt_id <> '');
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'omninode_internal.transcript_mirror_runs'::regclass
          AND conname = 'ck_transcript_mirror_runs_state'
    ) THEN
        ALTER TABLE omninode_internal.transcript_mirror_runs
            ADD CONSTRAINT ck_transcript_mirror_runs_state
            CHECK (state IN ('started', 'completed', 'dry-run-completed', 'failed'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'omninode_internal.transcript_mirror_runs'::regclass
          AND conname = 'ck_transcript_mirror_runs_finish_follows_state'
    ) THEN
        ALTER TABLE omninode_internal.transcript_mirror_runs
            ADD CONSTRAINT ck_transcript_mirror_runs_finish_follows_state
            CHECK ((state = 'started') = (finished_at IS NULL));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'omninode_internal.transcript_mirror_runs'::regclass
          AND conname = 'ck_transcript_mirror_runs_counts_non_negative'
    ) THEN
        ALTER TABLE omninode_internal.transcript_mirror_runs
            ADD CONSTRAINT ck_transcript_mirror_runs_counts_non_negative
            CHECK (files_seen >= 0 AND stored >= 0 AND would_store >= 0 AND unchanged >= 0
                AND destination_ahead >= 0 AND refused_stale >= 0 AND refused_conflict >= 0
                AND source_unreadable >= 0 AND bytes_put >= 0);
    END IF;

    -- Validate previously declared NOT VALID constraints against existing rows too.
    ALTER TABLE omninode_internal.transcript_mirror_runs
        VALIDATE CONSTRAINT ck_transcript_mirror_runs_ids_non_empty;
    ALTER TABLE omninode_internal.transcript_mirror_runs
        VALIDATE CONSTRAINT ck_transcript_mirror_runs_state;
    ALTER TABLE omninode_internal.transcript_mirror_runs
        VALIDATE CONSTRAINT ck_transcript_mirror_runs_finish_follows_state;
    ALTER TABLE omninode_internal.transcript_mirror_runs
        VALIDATE CONSTRAINT ck_transcript_mirror_runs_counts_non_negative;
END;
$reconcile$;

-- CREATE INDEX IF NOT EXISTS skips any relation holding the name. Accept it
-- only when it is the declared index on this table; refuse anything else
-- rather than drop or rebuild it.
DO $index_shape$
DECLARE
    actual_definition TEXT;
BEGIN
    SELECT pg_catalog.pg_get_indexdef(c.oid)
    INTO actual_definition
    FROM pg_catalog.pg_class c
    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'omninode_internal'
      AND c.relname = 'idx_transcript_mirror_runs_started_at';
    IF FOUND AND actual_definition IS DISTINCT FROM
        'CREATE INDEX idx_transcript_mirror_runs_started_at ON omninode_internal.transcript_mirror_runs USING btree (started_at DESC)'
    THEN
        RAISE EXCEPTION 'transcript_mirror_runs: incompatible index idx_transcript_mirror_runs_started_at; found %',
            actual_definition;
    END IF;
END;
$index_shape$;

CREATE INDEX IF NOT EXISTS idx_transcript_mirror_runs_started_at
    ON omninode_internal.transcript_mirror_runs (started_at DESC);

COMMIT;
