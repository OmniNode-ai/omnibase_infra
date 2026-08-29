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
-- NODE loop at `omnibase_infra` then applied the node file, whose
-- `CREATE TABLE IF NOT EXISTS` SILENTLY NO-OPPED against the already-existing
-- flat table. The node row was recorded as applied; the node SHAPE never
-- materialised. That is exactly the class the OMN-15376 block in the node file
-- was written to catch -- but that block only ADDs missing columns, SETs NOT
-- NULL and ADDs missing constraints. It cannot widen the TYPE of a column that
-- already exists, so even a clean re-apply of the current node bytes would
-- leave 8 of these 11 differences standing.
--
-- Hence: a forward migration, not a ledger declaration. A
-- `verified-divergent-adoptions.tsv` entry on its own would assert that the
-- applied SQL produced the schema the checked-in file produces. It did not --
-- until this file runs.
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
-- LOSSLESSNESS, AND WHY THE GUARDS ARE PER-ROW
-- ============================================================================
-- Every conversion below is a WIDENING:
--
--   VARCHAR(n) -> TEXT           unbounded target; no value can fail to fit.
--   NUMERIC(14,6) -> (18,6)      +4 integer digits, identical scale.
--
-- The two ways a NUMERIC widening could stop being a widening -- a stored value
-- with more than 6 fractional digits (rounded away) or one needing more than 12
-- integer digits (overflow) -- are checked in the USING clause, per row, at the
-- exact point the conversion happens. A row that would lose data aborts the
-- whole ALTER with a message naming this ticket and the reason; nothing is
-- applied, because a single ALTER TABLE is one transaction. That is deliberately
-- stricter than a table-level precondition: it cannot be true when the check
-- runs and false when the cast runs.
--
-- The guards are expressed as a CAST of an explanatory string to numeric rather
-- than as a PL/pgSQL RAISE because this file may not contain a procedural block
-- at all: `scripts/ci/check_application_database_sql.py` rejects every `DO`
-- block in changed SQL outright (its relation targets cannot be proven
-- statically), and adding this file to that gate's exemption list to keep a
-- nicer error string would be trading a real static guarantee for cosmetics.
-- The cast fails with the message inline, which is the same information.
--
-- Each message is anchored to the column (`|| left(<col>::text, 0)`, which
-- appends nothing) ON PURPOSE. A bare `'...'::NUMERIC` is a constant, and
-- PostgreSQL resolves constant casts at PARSE time -- so the untaken branch
-- would abort the migration on EVERY database, including the ones with no
-- offending row. Verified both ways on a scratch cluster before shipping.
--
-- The table is named unqualified, matching 074 and 076 in this same corpus:
-- `savings_estimates` is one of the relations whose physical table
-- intentionally remains in `public` until the governed OMN-15359 schema
-- cutover, and the changed-SQL gate's own allowlist
-- (TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359) is what admits it.
--
-- The TEXT conversions carry a type guard for a reason that is easy to get
-- wrong: `ALTER COLUMN ... TYPE TEXT` does NOT refuse an unrelated source type.
-- Postgres will happily I/O-convert one -- a `bytea` column would become the
-- string `\x6d61632d3031` and the migration would report success. Verified, not
-- assumed. So the character conversions check `pg_typeof` and refuse anything
-- outside the character family instead of relying on a refusal that does not
-- happen.
--
-- One unsafe shape genuinely needs no hand-written guard, because Postgres does
-- refuse it by name and this file does not suppress it: a declared column that
-- is absent -> `column "x" does not exist`.
--
-- The whole file runs inside one explicit transaction (as 051 and 077 in this
-- same corpus do), so "Nothing was applied" is true of the FILE and not merely
-- of the failing statement -- the runner invokes psql without
-- `--single-transaction`, so without this a later statement's refusal would
-- leave an earlier one's widening committed.
--
-- The three CHECK constraints are added under the names the node file declares.
-- They are not new SEMANTICS in this database: `non_negative_local`,
-- `non_negative_cloud` and `savings_consistency` already enforce the identical
-- predicates, so no row can violate the new names. DROP IF EXISTS + ADD in one
-- statement makes each idempotent without a procedural guard, and re-adding
-- revalidates every row -- so a future lane whose data HAS drifted fails here,
-- loudly, instead of acquiring a constraint that does not hold.
--
-- ============================================================================
-- IDEMPOTENCE AND THE FRESH PATH
-- ============================================================================
-- Re-running is a no-op: each ALTER COLUMN names the type it converges TO, each
-- ADD is guarded IF NOT EXISTS, and each constraint is dropped-if-present before
-- being added.
--
-- The fresh path is the reason step 4 exists at all. On the LIVE lane the
-- divergence is only the 11 differences above, because node 074 and node 075
-- had already run there and left `updated_at` and `ux_savings_estimates_identity`
-- behind. On a FRESH service database nothing from the node corpus ever runs --
-- flat 074 and flat 076 declare neither object -- so widening the eight columns
-- alone would leave the fresh path and the drifted path at two DIFFERENT
-- schemas, and this file's own claim to converge them would be false. Both are
-- in the surface node 074 declares, so both are added here, guarded. On the live
-- lane every statement in step 4 is a no-op. (Found in review, not by the live
-- proof: the live-lane replica could not have surfaced it.)
--
-- What step 4 deliberately does NOT reproduce is node 074's
-- `refresh_savings_estimates_updated_at()` function and its trigger. Those are
-- behaviour, not shape; they sit outside the surface the OMN-16915 verifier
-- measures (relations, columns, constraints, indexes, enums), and installing a
-- write-path trigger into the service database is a change this ticket did not
-- adjudicate and must not smuggle in.
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

BEGIN;

-- ---- 1. VARCHAR(n) -> TEXT ------------------------------------------------
-- Unbounded target, so no stored character value can fail to fit. The guard is
-- on the SOURCE TYPE, not on the value: see the header for why TEXT does not
-- refuse an unrelated source on its own.

ALTER TABLE savings_estimates
    ALTER COLUMN session_id TYPE TEXT
        USING CASE
            WHEN pg_typeof(session_id)::TEXT IN ('character varying', 'text', 'character') THEN
                session_id::TEXT
            ELSE
                ('OMN-16923: refusing to convert savings_estimates.session_id to TEXT '
                 '-- its live type is outside the character family, so this would '
                 'not be a widening: Postgres would I/O-convert the stored value '
                 'into its text rendering. That needs a ruling, not a guess. '
                 'Nothing was applied.'
                 || left(session_id::TEXT, 0))::NUMERIC::TEXT
        END,
    ALTER COLUMN model_local TYPE TEXT
        USING CASE
            WHEN pg_typeof(model_local)::TEXT IN ('character varying', 'text', 'character') THEN
                model_local::TEXT
            ELSE
                ('OMN-16923: refusing to convert savings_estimates.model_local to TEXT '
                 '-- its live type is outside the character family, so this would '
                 'not be a widening: Postgres would I/O-convert the stored value '
                 'into its text rendering. That needs a ruling, not a guess. '
                 'Nothing was applied.'
                 || left(model_local::TEXT, 0))::NUMERIC::TEXT
        END,
    ALTER COLUMN model_cloud_baseline TYPE TEXT
        USING CASE
            WHEN pg_typeof(model_cloud_baseline)::TEXT IN ('character varying', 'text', 'character') THEN
                model_cloud_baseline::TEXT
            ELSE
                ('OMN-16923: refusing to convert savings_estimates.model_cloud_baseline to TEXT '
                 '-- its live type is outside the character family, so this would '
                 'not be a widening: Postgres would I/O-convert the stored value '
                 'into its text rendering. That needs a ruling, not a guess. '
                 'Nothing was applied.'
                 || left(model_cloud_baseline::TEXT, 0))::NUMERIC::TEXT
        END,
    ALTER COLUMN repo_name TYPE TEXT
        USING CASE
            WHEN pg_typeof(repo_name)::TEXT IN ('character varying', 'text', 'character') THEN
                repo_name::TEXT
            ELSE
                ('OMN-16923: refusing to convert savings_estimates.repo_name to TEXT '
                 '-- its live type is outside the character family, so this would '
                 'not be a widening: Postgres would I/O-convert the stored value '
                 'into its text rendering. That needs a ruling, not a guess. '
                 'Nothing was applied.'
                 || left(repo_name::TEXT, 0))::NUMERIC::TEXT
        END,
    ALTER COLUMN machine_id TYPE TEXT
        USING CASE
            WHEN pg_typeof(machine_id)::TEXT IN ('character varying', 'text', 'character') THEN
                machine_id::TEXT
            ELSE
                ('OMN-16923: refusing to convert savings_estimates.machine_id to TEXT '
                 '-- its live type is outside the character family, so this would '
                 'not be a widening: Postgres would I/O-convert the stored value '
                 'into its text rendering. That needs a ruling, not a guess. '
                 'Nothing was applied.'
                 || left(machine_id::TEXT, 0))::NUMERIC::TEXT
        END;

-- ---- 2. NUMERIC(p,s) -> NUMERIC(18,6) -------------------------------------
-- scale(x) > 6      the conversion would ROUND stored digits away.
-- abs(x) >= 10^12   the target admits only 12 integer digits.

ALTER TABLE savings_estimates
    ALTER COLUMN local_cost_usd TYPE NUMERIC(18, 6)
        USING CASE
            WHEN scale(local_cost_usd) > 6 THEN
                ('OMN-16923: refusing to convert savings_estimates.local_cost_usd to '
                 'NUMERIC(18,6) -- a stored value carries more than 6 fractional '
                 'digits and would be ROUNDED. That is data loss, not a '
                 'convergence. Nothing was applied.'
                 || left(local_cost_usd::TEXT, 0))::NUMERIC
            WHEN abs(local_cost_usd) >= 1000000000000 THEN
                ('OMN-16923: refusing to convert savings_estimates.local_cost_usd to '
                 'NUMERIC(18,6) -- a stored value needs more than the 12 integer '
                 'digits the target admits and would OVERFLOW. Nothing was applied.'
                 || left(local_cost_usd::TEXT, 0))::NUMERIC
            ELSE local_cost_usd
        END,
    ALTER COLUMN cloud_cost_usd TYPE NUMERIC(18, 6)
        USING CASE
            WHEN scale(cloud_cost_usd) > 6 THEN
                ('OMN-16923: refusing to convert savings_estimates.cloud_cost_usd to '
                 'NUMERIC(18,6) -- a stored value carries more than 6 fractional '
                 'digits and would be ROUNDED. That is data loss, not a '
                 'convergence. Nothing was applied.'
                 || left(cloud_cost_usd::TEXT, 0))::NUMERIC
            WHEN abs(cloud_cost_usd) >= 1000000000000 THEN
                ('OMN-16923: refusing to convert savings_estimates.cloud_cost_usd to '
                 'NUMERIC(18,6) -- a stored value needs more than the 12 integer '
                 'digits the target admits and would OVERFLOW. Nothing was applied.'
                 || left(cloud_cost_usd::TEXT, 0))::NUMERIC
            ELSE cloud_cost_usd
        END,
    ALTER COLUMN savings_usd TYPE NUMERIC(18, 6)
        USING CASE
            WHEN scale(savings_usd) > 6 THEN
                ('OMN-16923: refusing to convert savings_estimates.savings_usd to '
                 'NUMERIC(18,6) -- a stored value carries more than 6 fractional '
                 'digits and would be ROUNDED. That is data loss, not a '
                 'convergence. Nothing was applied.'
                 || left(savings_usd::TEXT, 0))::NUMERIC
            WHEN abs(savings_usd) >= 1000000000000 THEN
                ('OMN-16923: refusing to convert savings_estimates.savings_usd to '
                 'NUMERIC(18,6) -- a stored value needs more than the 12 integer '
                 'digits the target admits and would OVERFLOW. Nothing was applied.'
                 || left(savings_usd::TEXT, 0))::NUMERIC
            ELSE savings_usd
        END;

-- ---- 3. the three CHECK constraints, under the DECLARED names --------------
-- DROP IF EXISTS + ADD in ONE statement: idempotent without a procedural guard,
-- never leaving a window where the predicate is unenforced, and revalidating
-- every row on the way in. The predicates are already enforced here under 074's
-- own names (non_negative_local / non_negative_cloud / savings_consistency), so
-- no row can violate them -- and a future lane whose data HAS drifted fails
-- here, loudly, instead of acquiring a constraint that does not hold.

ALTER TABLE savings_estimates
    DROP CONSTRAINT IF EXISTS savings_estimates_local_cost_usd_check,
    ADD CONSTRAINT savings_estimates_local_cost_usd_check CHECK (local_cost_usd >= 0);

ALTER TABLE savings_estimates
    DROP CONSTRAINT IF EXISTS savings_estimates_cloud_cost_usd_check,
    ADD CONSTRAINT savings_estimates_cloud_cost_usd_check CHECK (cloud_cost_usd >= 0);

ALTER TABLE savings_estimates
    DROP CONSTRAINT IF EXISTS savings_estimates_amounts_match,
    ADD CONSTRAINT savings_estimates_amounts_match
        CHECK (savings_usd = cloud_cost_usd - local_cost_usd);

-- ---- 4. the two objects only the NODE corpus ever declared -----------------
-- No-ops in effect on the live lane (node 074/075 already left both). Load-
-- bearing on a fresh service database, where nothing from the node corpus runs
-- and flat 074/076 declare neither -- without these the fresh path and the
-- drifted path would end at two different schemas. See the header.
--
-- `IF NOT EXISTS` guards a NAME, never a DEFINITION. A pre-existing
-- `updated_at` of the wrong type, or an `ux_savings_estimates_identity` over
-- different columns, would survive a bare guarded add and leave this file
-- claiming a convergence it did not perform. So the column is added when
-- missing and then converged UNCONDITIONALLY (type, default, nullability), and
-- the index is dropped and recreated from the declaration rather than merely
-- asserted to exist. Both are inside this file's single transaction.

ALTER TABLE savings_estimates
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

-- Converge whatever is there now onto the declared shape. A source outside the
-- timestamptz family is refused rather than reinterpreted: casting
-- `timestamp without time zone` would silently re-stamp every row with the
-- session's time zone, which is a data change wearing a type change's clothes.
ALTER TABLE savings_estimates
    ALTER COLUMN updated_at TYPE TIMESTAMPTZ
        USING CASE
            WHEN pg_typeof(updated_at)::TEXT = 'timestamp with time zone' THEN
                updated_at
            ELSE
                ('OMN-16923: refusing to convert savings_estimates.updated_at to '
                 'TIMESTAMPTZ -- its live type is not already timestamptz, so the '
                 'conversion would REINTERPRET every stored value against the '
                 'session time zone rather than widen it. Nothing was applied.'
                 || left(updated_at::TEXT, 0))::TIMESTAMPTZ
        END,
    ALTER COLUMN updated_at SET DEFAULT NOW();

-- Fails loud (and rolls the file back) if a pre-existing row holds NULL, rather
-- than inventing a timestamp -- the same posture as OMN-15376's NOT NULL loop.
ALTER TABLE savings_estimates
    ALTER COLUMN updated_at SET NOT NULL;

-- Recreated from the declaration, not merely asserted by name.
--
-- On duplicates: they cannot arise on any database the flat corpus built. 074
-- declares `CONSTRAINT unique_savings_estimate_event UNIQUE (session_id,
-- event_timestamp, model_local, model_cloud_baseline)` -- the SAME four columns
-- in the same order this index covers -- so a duplicate tuple was already
-- impossible before this file ran. (Asserted, not assumed:
-- test_the_flat_unique_constraint_already_covers_the_identity_tuple.) Should a
-- later lane have dropped that constraint and admitted duplicates, this file
-- refuses with an OMN-16923 diagnostic before it reaches CREATE UNIQUE INDEX.
SELECT CASE
    WHEN EXISTS (
        SELECT 1
        FROM savings_estimates
        GROUP BY
            session_id,
            event_timestamp,
            model_local,
            model_cloud_baseline
        HAVING COUNT(*) > 1
    ) THEN
        ('OMN-16923: refusing to create ux_savings_estimates_identity -- '
         'duplicate savings_estimates identity tuples already exist. Resolve '
         'the duplicate data before applying this convergence.'
         || left((SELECT COUNT(*)::TEXT FROM savings_estimates), 0))::INTEGER
    ELSE 0
END;

DROP INDEX IF EXISTS ux_savings_estimates_identity;

CREATE UNIQUE INDEX ux_savings_estimates_identity
    ON savings_estimates (
        session_id,
        event_timestamp,
        model_local,
        model_cloud_baseline
    );

COMMIT;
