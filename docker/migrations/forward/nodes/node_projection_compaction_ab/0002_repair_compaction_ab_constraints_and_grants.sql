-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- OMN-21994: repair the infra-first compaction A/B bridge without rewriting
-- the checksum-bound 0000/0001 vendor history. Legacy tables took the ADD
-- COLUMN path in 0000, which did not install its CREATE-time constraints.
-- Invalid legacy data must stop admission; never discard or invent history.

BEGIN;

ALTER TABLE omninode_internal.compaction_ab_rows
    ALTER COLUMN experiment_id SET NOT NULL,
    ALTER COLUMN session_id SET NOT NULL,
    ALTER COLUMN boundary_uuid SET NOT NULL;

ALTER TABLE omninode_internal.compaction_ab_build_series
    ALTER COLUMN candidate_id SET NOT NULL,
    ALTER COLUMN node_commit SET NOT NULL,
    ALTER COLUMN split SET NOT NULL,
    ALTER COLUMN recorded_at SET NOT NULL;

DO $repair$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'omninode_internal.compaction_ab_rows'::regclass
          AND contype = 'p'
          -- Uniqueness and ON CONFLICT inference depend on the column set,
          -- not the index order or the legacy constraint's name.
          AND (
              SELECT array_agg(a.attname::text ORDER BY a.attname)
              FROM pg_catalog.pg_attribute AS a
              WHERE a.attrelid = conrelid AND a.attnum = ANY (conkey)
          ) = ARRAY['boundary_uuid', 'experiment_id', 'session_id']
    ) THEN
        ALTER TABLE omninode_internal.compaction_ab_rows
            ADD PRIMARY KEY (experiment_id, session_id, boundary_uuid);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'omninode_internal.compaction_ab_build_series'::regclass
          AND contype = 'p'
          AND (
              SELECT array_agg(a.attname::text ORDER BY a.attname)
              FROM pg_catalog.pg_attribute AS a
              WHERE a.attrelid = conrelid AND a.attnum = ANY (conkey)
          ) = ARRAY['candidate_id', 'node_commit']
    ) THEN
        ALTER TABLE omninode_internal.compaction_ab_build_series
            ADD PRIMARY KEY (candidate_id, node_commit);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'omninode_internal.compaction_ab_build_series'::regclass
          AND contype = 'c' AND convalidated
          AND pg_catalog.pg_get_constraintdef(oid) = 'CHECK ((candidate_id <> ''''::text))'
    ) THEN
        ALTER TABLE omninode_internal.compaction_ab_build_series
            ADD CONSTRAINT compaction_ab_build_series_candidate_id_required
            CHECK (candidate_id <> '');
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid = 'omninode_internal.compaction_ab_build_series'::regclass
          AND contype = 'c' AND convalidated
          AND pg_catalog.pg_get_constraintdef(oid) = 'CHECK ((split = ''calibration''::text))'
    ) THEN
        ALTER TABLE omninode_internal.compaction_ab_build_series
            ADD CONSTRAINT compaction_ab_build_series_split_required
            CHECK (split = 'calibration');
    END IF;
END
$repair$;

-- The read_write binding requires UPDATE as well as SELECT and INSERT.
GRANT UPDATE ON omninode_internal.compaction_ab_build_series TO omninode_runtime;

COMMIT;
