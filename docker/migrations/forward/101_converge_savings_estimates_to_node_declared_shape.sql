-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Migration 101 (OMN-16923): converge the SERVICE database's savings_estimates
-- onto the shape the node declaration produces.
--
-- ============================================================================
-- WHY THIS FILE EXISTS
-- ============================================================================
-- OMN-16919's census ran OMN-16915's replay-and-introspect verifier against the
-- .201 stability-test lane's `omnibase_infra` database and returned exactly one
-- DIVERGENT row:
--
--   node:node_projection_savings:074_create_savings_estimates.sql
--     ledger  (public.omnimarket_schema_migrations)
--             d5eedd28f26c32f2e9d2a8554a999209c68216dc6a1ee255a973bd034164ce55
--     manifest b78acc5ba3144f9a7c7d85fd0fd5803b02b60503765fc58f8650a6a2bde27f4e
--     verdict  divergent -- 11 structural difference(s)
--
--   receipt: docker/migrations/forward/_ledger/receipts/
--            omn16919-stability-test-omnibase-infra-2026-08-29.json
--
-- The 11 differences are NOT a stale-revision artifact, and that is the whole
-- point. The ledger's bytes name the 2026-07-21 revision (5b904d881) of the node
-- file, superseded on 2026-07-29 by 78b873110 / OMN-15376 -- but BOTH revisions
-- declare the identical `CREATE TABLE IF NOT EXISTS savings_estimates (...)`
-- with TEXT columns, NUMERIC(18,6) money columns and three named CHECKs. The
-- diff between them is only the OMN-15376 reconciliation block. So the revision
-- gap explains nothing here.
--
-- What explains it is the dual-producer topology this repo already records:
--
--   * flat `docker/migrations/forward/*.sql` run against PGDB   = omnibase_infra
--   * node `docker/migrations/forward/nodes/<n>/*.sql` run against NODE_PGDB
--                                                               = omnidash_analytics
--
--   (scripts/run-forward-migrations.sh; the ownership ruling and its four
--   consequences are pinned in tests/ci/test_llm_call_metrics_ownership_omn15857.py,
--   and the table-level record is docker/migrations/forward/flat-node-shape-parity.yaml,
--   where `savings_estimates` has been `status: accepted_divergence` since OMN-15384.)
--
-- The FLAT producer -- docker/migrations/forward/074_create_savings_estimates.sql
-- -- creates this table in `omnibase_infra` with VARCHAR(255)/VARCHAR(64) and
-- NUMERIC(14,6), under the constraint names `non_negative_local`,
-- `non_negative_cloud` and `savings_consistency`. A run that also pointed the
-- NODE loop at `omnibase_infra` then applied the node file on top, whose
-- `CREATE TABLE IF NOT EXISTS` SILENTLY NO-OPPED against the already-existing
-- flat table. The node row was recorded as applied; the node SHAPE never
-- materialised. That is exactly the class the OMN-15376 block in the node file
-- was written to catch -- but that block only ADDs missing columns, SETs NOT
-- NULL and ADDs missing constraints. It cannot widen the TYPE of a column that
-- already exists, so even a clean re-apply of the current node bytes would
-- leave 8 of these 11 differences standing.
--
-- Hence: a forward migration, not a ledger declaration. A
-- `verified-divergent-adoptions.tsv` entry here would assert that the applied
-- SQL produced the schema the checked-in file produces. It did not.
--
-- ============================================================================
-- WHY THE FLAT SET, NOT THE NODE SET
-- ============================================================================
-- The divergent object is `omnibase_infra.public.savings_estimates`. Node
-- migrations are applied against NODE_PGDB, which every compose lane pins to
-- `omnidash_analytics` (asserted per migrator service in
-- tests/ci/test_llm_call_metrics_ownership_omn15857.py). A node-scoped 085
-- would therefore run against the database that is ALREADY correct and would
-- never reach the one that is wrong. Per the OMN-15857 ownership ruling, the
-- flat corpus is the only declaring owner the service database has, so the
-- convergence belongs here.
--
-- ============================================================================
-- LOSSLESSNESS
-- ============================================================================
-- Every conversion below is a WIDENING, and each one is re-proved against the
-- live catalog at apply time rather than assumed from this comment:
--
--   VARCHAR(n) -> TEXT           unbounded target; no value can fail to fit.
--   NUMERIC(14,6) -> (18,6)      +4 integer digits, identical scale.
--
-- The three CHECK constraints are added under the names the node file declares.
-- They are not new SEMANTICS in this database: `non_negative_local`,
-- `non_negative_cloud` and `savings_consistency` already enforce the identical
-- predicates, so no row can violate the new names. The migration still counts
-- violations first and RAISEs with the count rather than trusting that claim.
--
-- Anything that would NOT be provably lossless RAISEs and applies nothing:
-- a column that is not varchar/text, a numeric with scale > 6 (fractional
-- truncation), a numeric needing more than 12 integer digits (overflow), or an
-- unconstrained NUMERIC (whose contents cannot be bounded a priori). The
-- migration refuses to guess -- same posture as the OMN-15376 NOT NULL loop.
--
-- ============================================================================
-- IDEMPOTENCE AND THE FRESH PATH
-- ============================================================================
-- Every step is guarded on the LIVE catalog state, so:
--   * re-running is a no-op;
--   * a database where the table is absent is a no-op (NOTICE, not an error);
--   * on a fresh service database, flat 074 creates the narrow shape and this
--     file widens it, so the fresh path and the drifted path end at ONE schema
--     -- the same both-paths-converge property the OMN-15376 block guarantees
--     inside the node corpus.
--
-- This does NOT edit 074. Editing an applied migration in place is the
-- OMN-16705 class that check_migration_append_only.py exists to reject; 074
-- stays the historical record of what the service database was first given.
--
-- Gated by tests/integration/migrations/test_101_savings_estimates_convergence_omn16923.py
-- (RED-first: the pre-101 schema is proved DIVERGENT against the node file's
-- executed surface, and the post-101 schema EQUIVALENT, using the OMN-16915
-- verifier's own snapshot/diff engine rather than a restatement of it).
--
-- Ticket: OMN-16923. Family: OMN-15857 / OMN-16915 / OMN-16919. Related:
-- OMN-15384 (the flat-vs-node parity ledger), OMN-15376 (in-corpus shape
-- reconciliation), OMN-15857 (the flat/node ownership ruling).

DO $omn16923$
DECLARE
    v_rel        regclass := to_regclass('public.savings_estimates');
    v_col        TEXT;
    v_type       TEXT;
    v_typmod     INT;
    v_precision  INT;
    v_scale      INT;
    v_violations BIGINT;
    -- Declared TEXT by nodes/node_projection_savings/074_create_savings_estimates.sql.
    v_text_columns CONSTANT TEXT[] := ARRAY[
        'session_id', 'model_local', 'model_cloud_baseline', 'repo_name', 'machine_id'
    ];
    -- Declared NUMERIC(18, 6) by the same file.
    v_numeric_columns CONSTANT TEXT[] := ARRAY[
        'local_cost_usd', 'cloud_cost_usd', 'savings_usd'
    ];
BEGIN
    IF v_rel IS NULL THEN
        RAISE NOTICE
            'OMN-16923: public.savings_estimates does not exist here -- nothing to converge.';
        RETURN;
    END IF;

    -- ---- 1. VARCHAR(n) -> TEXT -------------------------------------------
    FOREACH v_col IN ARRAY v_text_columns
    LOOP
        SELECT format_type(a.atttypid, a.atttypmod)
          INTO v_type
          FROM pg_attribute a
         WHERE a.attrelid = v_rel
           AND a.attname = v_col
           AND a.attnum > 0
           AND NOT a.attisdropped;

        IF v_type IS NULL THEN
            RAISE EXCEPTION
                'OMN-16923: public.savings_estimates.% is absent, so the declared TEXT '
                'shape cannot be reached by widening. This is a different divergence '
                'class than the one this migration was written for; it needs its own '
                'ruling. Nothing was applied.', v_col;
        ELSIF v_type = 'text' THEN
            CONTINUE;  -- already converged
        ELSIF v_type LIKE 'character varying%' THEN
            EXECUTE format(
                'ALTER TABLE public.savings_estimates ALTER COLUMN %I TYPE TEXT', v_col
            );
            RAISE NOTICE 'OMN-16923: widened public.savings_estimates.% from % to text',
                v_col, v_type;
        ELSE
            RAISE EXCEPTION
                'OMN-16923: refusing to convert public.savings_estimates.% from % to TEXT '
                '-- only character varying and text are provably lossless sources. '
                'Nothing was applied.', v_col, v_type;
        END IF;
    END LOOP;

    -- ---- 2. NUMERIC(p,s) -> NUMERIC(18,6) --------------------------------
    FOREACH v_col IN ARRAY v_numeric_columns
    LOOP
        SELECT a.atttypid::regtype::TEXT, a.atttypmod
          INTO v_type, v_typmod
          FROM pg_attribute a
         WHERE a.attrelid = v_rel
           AND a.attname = v_col
           AND a.attnum > 0
           AND NOT a.attisdropped;

        IF v_type IS NULL THEN
            RAISE EXCEPTION
                'OMN-16923: public.savings_estimates.% is absent, so the declared '
                'NUMERIC(18,6) shape cannot be reached by widening. Nothing was applied.',
                v_col;
        ELSIF v_type <> 'numeric' THEN
            RAISE EXCEPTION
                'OMN-16923: refusing to convert public.savings_estimates.% from % to '
                'NUMERIC(18,6) -- a non-numeric source is not a widening and this '
                'migration will not guess at a cast. Nothing was applied.', v_col, v_type;
        ELSIF v_typmod = -1 THEN
            RAISE EXCEPTION
                'OMN-16923: public.savings_estimates.% is an UNCONSTRAINED numeric. '
                'Constraining it to NUMERIC(18,6) is a NARROWING -- any stored value '
                'with more than 12 integer digits or more than 6 fractional digits '
                'would be rejected or rounded. This needs a data ruling, not a '
                'migration that guesses. Nothing was applied.', v_col;
        ELSE
            v_precision := ((v_typmod - 4) >> 16) & 65535;
            v_scale     := (v_typmod - 4) & 65535;

            IF v_precision = 18 AND v_scale = 6 THEN
                CONTINUE;  -- already converged
            ELSIF v_scale > 6 THEN
                RAISE EXCEPTION
                    'OMN-16923: refusing to convert public.savings_estimates.% from '
                    'NUMERIC(%,%) to NUMERIC(18,6) -- scale % exceeds 6, so the '
                    'conversion would ROUND stored fractional digits away. That is a '
                    'data loss, not a convergence. Nothing was applied.',
                    v_col, v_precision, v_scale, v_scale;
            ELSIF (v_precision - v_scale) > 12 THEN
                RAISE EXCEPTION
                    'OMN-16923: refusing to convert public.savings_estimates.% from '
                    'NUMERIC(%,%) to NUMERIC(18,6) -- it admits % integer digits and '
                    'the target admits only 12, so a stored value could OVERFLOW. '
                    'Nothing was applied.',
                    v_col, v_precision, v_scale, v_precision - v_scale;
            ELSE
                EXECUTE format(
                    'ALTER TABLE public.savings_estimates ALTER COLUMN %I TYPE NUMERIC(18, 6)',
                    v_col
                );
                RAISE NOTICE
                    'OMN-16923: widened public.savings_estimates.% from numeric(%,%) to numeric(18,6)',
                    v_col, v_precision, v_scale;
            END IF;
        END IF;
    END LOOP;

    -- ---- 3. the three CHECK constraints, under the DECLARED names ---------
    -- The predicates are already enforced here under the flat file's own names
    -- (non_negative_local / non_negative_cloud / savings_consistency), which is
    -- why no row can violate these. Counted anyway: a claim that costs nothing
    -- to check is not a claim worth trusting.
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = v_rel AND conname = 'savings_estimates_local_cost_usd_check'
    ) THEN
        SELECT count(*) INTO v_violations
          FROM public.savings_estimates WHERE local_cost_usd < 0;
        IF v_violations > 0 THEN
            RAISE EXCEPTION
                'OMN-16923: cannot add savings_estimates_local_cost_usd_check -- % row(s) '
                'hold a negative local_cost_usd. Nothing was applied.', v_violations;
        END IF;
        ALTER TABLE public.savings_estimates
            ADD CONSTRAINT savings_estimates_local_cost_usd_check CHECK (local_cost_usd >= 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = v_rel AND conname = 'savings_estimates_cloud_cost_usd_check'
    ) THEN
        SELECT count(*) INTO v_violations
          FROM public.savings_estimates WHERE cloud_cost_usd < 0;
        IF v_violations > 0 THEN
            RAISE EXCEPTION
                'OMN-16923: cannot add savings_estimates_cloud_cost_usd_check -- % row(s) '
                'hold a negative cloud_cost_usd. Nothing was applied.', v_violations;
        END IF;
        ALTER TABLE public.savings_estimates
            ADD CONSTRAINT savings_estimates_cloud_cost_usd_check CHECK (cloud_cost_usd >= 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = v_rel AND conname = 'savings_estimates_amounts_match'
    ) THEN
        SELECT count(*) INTO v_violations
          FROM public.savings_estimates
         WHERE savings_usd <> cloud_cost_usd - local_cost_usd;
        IF v_violations > 0 THEN
            RAISE EXCEPTION
                'OMN-16923: cannot add savings_estimates_amounts_match -- % row(s) do not '
                'satisfy savings_usd = cloud_cost_usd - local_cost_usd. Nothing was '
                'applied.', v_violations;
        END IF;
        ALTER TABLE public.savings_estimates
            ADD CONSTRAINT savings_estimates_amounts_match
            CHECK (savings_usd = cloud_cost_usd - local_cost_usd);
    END IF;
END
$omn16923$;
