-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- =============================================================================
-- MIGRATION 105: Create the ledger_chain relation (OMN-16964)
-- =============================================================================
-- Ticket: OMN-16964 (chain-canary link 5 has no leg). Gate: OMN-16025 link 5,
--         verbatim: "Complete ledger chain + replay green through an HONEST
--         tier-2 verifier (SKIP != PASS)".
-- Version: 1.0.0
--
-- WHY 105 AND NOT 104
-- -------------------
-- Ordinal 104 is BURNED. `104_create_validator_ro_role.sql` (OMN-17792, #3190)
-- was retired by OMN-17923 (#3197) because it aborts the onex-dev staging
-- migration step, and the retirement record
-- (`_ledger/retired-flat-migrations.tsv`) requires the migration to come back
-- as a NEW number, never as 104 again, so that a lane which recorded 104 can
-- never be confused with one that applied the re-issue.
-- `tests/unit/db/test_migration_104_retired_omn17923.py` enforces it. This is
-- the first new forward migration since that retirement, so it takes 105 and
-- restamps the stream fingerprint.
--
-- =============================================================================
-- WHY THIS FILE EXISTS — A MERGED CONSUMER READS A RELATION THAT DOES NOT EXIST
-- =============================================================================
-- omnibase_infra#3072 and #3079 landed the READER half of link 5 in
-- `node_chain_canary_effect`. `_replay_ledger_chain_via_asyncpg` issues,
-- verbatim:
--
--     SELECT hop, replay_green, verifier_verdict FROM ledger_chain
--     WHERE correlation_id = $1 ORDER BY hop_index
--
-- That relation exists in no database on any lane. Probed read-only on the
-- .201 dev lane 2026-09-10: `to_regclass` returns NULL in omnibase_infra,
-- omnidash_analytics and omninode_cloud alike, and the string `ledger_chain`
-- has zero references across omnibase_infra, omnibase_core, omnimarket and
-- omnimemory outside the canary node itself. The canary has consequently
-- reported `ledger_replay_not_configured` — 4 of 5 links, red and honest —
-- on every run since #3345 fixed link 2 (runs 34355201941, 34356615155).
--
-- This migration creates the relation. The columns are NOT a design choice
-- made here: five of them are a contract with the merged consumer above and
-- may not be renamed.
--
-- =============================================================================
-- WHY THIS DATABASE, AND NOT omnidash_analytics
-- =============================================================================
-- `ledger_chain` targets `omnibase_infra` via the flat forward-migration set,
-- NOT the node-vendored set under docker/migrations/forward/nodes/ (which
-- applies to NODE_PGDB=omnidash_analytics). Three reasons, in order of weight:
--
--   1. It is a VERIFICATION ledger, not an analytics projection. Nothing
--      renders it. The direct precedents are its two immediate neighbours in
--      this same flat set — 088 `overseer_tick_ledger` and 089
--      `verification_receipt_ledger` — both verification ledgers, both here.
--
--   2. The reader identity already exists here and nowhere else.
--      `chain_canary_reader` (OMN-18060) holds CONNECT on this database,
--      USAGE on schema public, and column-scoped SELECT on
--      `delegation_workflow_state (correlation_id, state)` for link 2. Putting
--      link 5's relation in the same database means one added column grant
--      rather than a second least-privilege identity, a second credential and
--      a second job secret — and no credential is minted by this change at all.
--
--   3. `delegation_workflow_state`, the relation link 2 reads for the SAME
--      correlation id, is here (migration 090). Link 2 and link 5 corroborate
--      each other; splitting them across two databases would mean the canary
--      could not join them even in a diagnostic query run by hand.
--
-- =============================================================================
-- THE COLUMN SET
-- =============================================================================
-- Consumer-contract columns (renaming any of these breaks a merged reader):
--   correlation_id, hop, hop_index, replay_green, verifier_verdict
--
-- Diagnostic columns (the consumer does not read these; a human diagnosing a
-- red canary does):
--   observed_topic, envelope_id, parent_envelope_id, replay_detail,
--   verifier_detail, recorded_at
--
-- `verifier_verdict` carries a CHECK constraint pinning it to the three
-- tokens the consumer classifies. The consumer treats anything that is not
-- the literal `pass` as non-passing, so an unconstrained column could carry a
-- typo that silently renders as a replay failure rather than as the write bug
-- it is. The constraint makes that unrepresentable rather than merely unlikely.
--
-- There is deliberately NO column on this table that could carry an event
-- payload. The chain is evidence about SHAPE — which hops happened, in what
-- order, with what causal linkage — and a payload column would make this table
-- a second, ungoverned copy of tenant data.
--
-- WHY correlation_id IS TEXT AND NOT uuid
-- ---------------------------------------
-- Because the merged reader binds a Python `str`. `_replay_ledger_chain_via_asyncpg`
-- calls `connection.fetch(..., correlation_id)` where `correlation_id` is the
-- probe's correlation id as a string; asyncpg type-checks query arguments
-- against the column, so a `uuid` column would reject that bind outright and
-- the leg would report ERROR on every run. `delegation_workflow_state`
-- (migration 090) makes the same choice for the same reason, and its header
-- records it. The WRITER types these as `uuid` in its own models, per the
-- repo's pattern gate, and renders them at the database boundary — typed in
-- the domain, TEXT on the wire the reader already speaks.
--
-- EVERY RELATION REFERENCE IS SCHEMA-QUALIFIED. The application-database
-- domain enforcement gate (OMN-15361, ci.yml "Enforce schema qualification in
-- changed SQL") requires it of CHANGED SQL. Older files in this set are
-- unqualified because the gate only ever inspects the diff; that is
-- grandfathering, not a licence to add more.
--
-- Idempotent CREATE so warm dev/stability volumes reconcile cleanly, matching
-- 090's own header.
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.ledger_chain (
    -- Consumer contract: the five columns node_chain_canary_effect reads.
    correlation_id      TEXT        NOT NULL,
    hop_index           INTEGER     NOT NULL,
    hop                 TEXT        NOT NULL,
    replay_green        BOOLEAN     NOT NULL,
    verifier_verdict    TEXT        NOT NULL,

    -- Diagnostics. Never read by the canary; read by whoever it sends here.
    observed_topic      TEXT        NOT NULL DEFAULT '',
    envelope_id         TEXT        NOT NULL DEFAULT '',
    parent_envelope_id  TEXT        NOT NULL DEFAULT '',
    replay_detail       TEXT        NOT NULL DEFAULT '',
    verifier_detail     TEXT        NOT NULL DEFAULT '',
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One row per (chain, position). The writer is idempotent on re-delivery:
    -- a redelivered envelope re-derives the same verdict for the same
    -- position, so ON CONFLICT DO UPDATE converges rather than duplicating.
    CONSTRAINT pk_ledger_chain PRIMARY KEY (correlation_id, hop_index),

    -- The tier-2 verifier's vocabulary, pinned. `skip` is a first-class
    -- member: OMN-16025 requires SKIP to be distinguishable from PASS, and a
    -- column that cannot represent SKIP forces the writer to lie.
    CONSTRAINT ck_ledger_chain_verifier_verdict
        CHECK (verifier_verdict IN ('pass', 'fail', 'skip'))
);

-- The canary's only access path is by correlation id; the primary key's
-- leading column already serves it. No second index is created — an unused
-- index on a write-path table is cost with no reader.

COMMENT ON TABLE public.ledger_chain IS
    'OMN-16964: per-hop delegation chain with a re-derived replay result and '
    'an honest tier-2 verifier verdict. Read by node_chain_canary_effect for '
    'OMN-16025 link 5. verifier_verdict = skip is NOT a pass.';

COMMENT ON COLUMN public.ledger_chain.replay_green IS
    'Did this hop''s causal linkage RE-DERIVE from the recorded evidence? '
    'Recomputed from the preceding hop and compared — never a flag copied off '
    'the envelope. Covers linkage only; the inference hop is not deterministic '
    'and is not claimed to have been re-executed.';

COMMENT ON COLUMN public.ledger_chain.verifier_verdict IS
    'Tier-2: does the observed hop match the DECLARED chain topology at this '
    'position? pass / fail / skip. skip means no declaration reached this hop, '
    'so no check ran — it is never counted as green (OMN-16025, OMN-16773).';

-- =============================================================================
-- GRANT: the canary's read-only instrument identity
-- =============================================================================
-- Column-scoped SELECT on exactly the five columns the consumer selects, and
-- nothing else. This mirrors the link-2 grant, which is column-scoped on
-- `delegation_workflow_state (correlation_id, state)` — verified live on the
-- .201 dev lane 2026-09-10: `information_schema.column_privileges` returns
-- exactly two rows for this grantee and `has_table_privilege(...,'SELECT')`
-- is FALSE, which is the intended shape rather than a defect.
--
-- Guarded on role existence and issued through a DO block, following 103's
-- pattern (OMN-17301): a lane with no canary provisioned leaves the role
-- absent, this block skips with a NOTICE, and the canary reports
-- SKIPPED_NOT_CONFIGURED — red and honest, never a green from a check that
-- never ran. docker-compose.infra.yml's CHAIN_CANARY_READER_PASSWORD comment
-- already documents exactly this contract; that comment names 'migration 104',
-- which is now a burned ordinal, and is corrected in the same change.
--
-- NOTE ON REPRODUCIBILITY, recorded because it is a live gap this file closes
-- by half: the link-2 column grant on the .201 dev lane is NOT reproducible
-- from this repository. `chain_canary_reader` holds it live, but no migration
-- in docker/migrations/forward/ issues it and `grep -rn chain_canary_reader
-- src/` returns nothing — it was applied out of band. This file makes the
-- link-5 grant reproducible; the link-2 grant remains hand-applied and is
-- filed separately.
DO $$
DECLARE
    executing_role TEXT := current_user;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'chain_canary_reader'
    ) THEN
        RAISE NOTICE
            'role chain_canary_reader is absent on this lane; skipping the '
            'ledger_chain grant. node_chain_canary_effect will report '
            'OMN-16025 link 5 as skipped_not_configured, which is RED.';
        RETURN;
    END IF;

    BEGIN
        GRANT USAGE ON SCHEMA public TO chain_canary_reader;
        GRANT SELECT (
            correlation_id, hop, hop_index, replay_green, verifier_verdict
        ) ON public.ledger_chain TO chain_canary_reader;
        RAISE NOTICE
            'granted column-scoped SELECT on ledger_chain to '
            'chain_canary_reader as %', executing_role;
    EXCEPTION
        WHEN insufficient_privilege THEN
            -- Same fall-through as 103: on a managed lane the migration
            -- identity may hold no grant option. The readback in the canary
            -- is what decides, and it fails closed.
            RAISE NOTICE
                'GRANT on ledger_chain was refused for %; the canary will '
                'report link 5 as an ERROR on read, which is RED and '
                'diagnosable rather than silently green.', executing_role;
    END;
END
$$;
