-- Migration: 089_create_verification_receipt_ledger.sql
-- Purpose:   Create the append-only verification_receipt_ledger projection table —
--            the receiving end of the verification-receipts ledger migration
--            (WS-L slice 2). This is THE ledger the initiative exists to fix:
--            the completion-evidence ledger currently lives on local disk
--            (.onex_state/verification-receipts/*.yaml), a direct contradiction
--            of the deterministic-truth doctrine §1 (plan §1.1, §6 Phase 1).
-- Ticket:    OMN-13997 (child of epic OMN-13989)
-- Plan:      docs/plans/2026-07-05-ledger-learning-substrate-plan.md (§2.5 raw
--            projection, §6 Phase 1 / §7 issue 7)
--
-- Design decisions:
--
--   1. RAW ledger projection (append-only, full-fidelity, replayable) for the
--      verification-receipt ledger written to .onex_state/verification-receipts/
--      by the omniclaude verification_sweep / epic_team skills. Materialized from
--      onex.evt.omniclaude.verification_receipt.v1 events by the generic
--      ProjectorShell driven by verification_receipt_projector.yaml — no bespoke
--      node or class (CLAUDE.md rule 7a; reference pattern: decision_store / 046,
--      overseer_tick_ledger / 088).
--
--   2. envelope_id is the PRIMARY KEY and idempotency key. Duplicate replay
--      raises a unique violation the capture consumer treats as already-ingested
--      (event_ledger / 044 discipline). Append mode = plain INSERT; history is
--      never mutated. (Numbered 089: 088 is reserved by the in-flight
--      overseer_tick_ledger PR #2216.)
--
--   3. Ordering authority is single_partition (plan §2.4): low-volume,
--      single-writer ledger, so the per-partition offset is the monotonic cursor.
--
--   4. HETEROGENEOUS on-disk shapes (verified across the live ledger): the
--      ticket receipts and the overseer-verify-tick receipts share a stable core
--      (sweep_timestamp, overall_status always present; ticket_id / phases /
--      idempotent on the standard shape; overseer_verdict only on tick receipts),
--      and one legacy `type: baseline` snapshot omits ticket_id/phases entirely.
--      The queryable core is projected into typed NULLABLE columns; the FULL raw
--      payload is captured losslessly in receipt_body JSONB so no shape is dropped
--      (full-fidelity raw projection, plan §2.5). This is the "minimal receipt
--      projection" plan §7 issue 7 calls for, made lossless.
--
--   5. ingested_at is DB-set (DEFAULT NOW()) and is NOT projector-extracted —
--      server-side capture time, distinct from sweep_timestamp.
--
--   6. Idempotent via IF NOT EXISTS so re-running the migration is safe.

-- =============================================================================
-- TABLE: verification_receipt_ledger
-- =============================================================================

CREATE TABLE IF NOT EXISTS verification_receipt_ledger (
    envelope_id       UUID PRIMARY KEY,
    correlation_id    UUID NOT NULL,
    ticket_id         TEXT,
    sweep_timestamp   TIMESTAMPTZ NOT NULL,
    overall_status    TEXT NOT NULL,
    overseer_verdict  TEXT,
    idempotent        BOOLEAN,
    phases            JSONB,
    receipt_body      JSONB NOT NULL DEFAULT '{}',
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Object-shape guards on the JSONB columns (phases is nullable).
    CONSTRAINT chk_verification_receipt_phases_is_object
        CHECK (phases IS NULL OR jsonb_typeof(phases) = 'object'),
    CONSTRAINT chk_verification_receipt_body_is_object
        CHECK (jsonb_typeof(receipt_body) = 'object')
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Per-ticket receipt history (partial — many rows are per-ticket, the baseline
-- outlier has NULL ticket_id).
CREATE INDEX IF NOT EXISTS idx_verification_receipt_ledger_ticket_swept
    ON verification_receipt_ledger (ticket_id, sweep_timestamp DESC)
    WHERE ticket_id IS NOT NULL;

-- Global recency / cursor scan ordered by sweep time.
CREATE INDEX IF NOT EXISTS idx_verification_receipt_ledger_swept
    ON verification_receipt_ledger (sweep_timestamp DESC);

-- Status filtering (e.g. "recent failing sweeps").
CREATE INDEX IF NOT EXISTS idx_verification_receipt_ledger_status
    ON verification_receipt_ledger (overall_status);

-- Trace correlation lookups.
CREATE INDEX IF NOT EXISTS idx_verification_receipt_ledger_correlation
    ON verification_receipt_ledger (correlation_id);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE verification_receipt_ledger IS
    'Append-only raw projection of the verification-receipt completion-evidence '
    'ledger (onex.evt.omniclaude.verification_receipt.v1), replacing the flat '
    'files .onex_state/verification-receipts/*.yaml that violate deterministic-'
    'truth doctrine §1. Materialized by the generic ProjectorShell from '
    'verification_receipt_projector.yaml. WS-L slice 2 (OMN-13997 / epic OMN-13989).';

COMMENT ON COLUMN verification_receipt_ledger.envelope_id IS
    'Event envelope id — primary key and idempotency key. Duplicate replay '
    'raises a unique violation the capture consumer treats as already-ingested.';

COMMENT ON COLUMN verification_receipt_ledger.ticket_id IS
    'Ticket the receipt verifies. NULLABLE: the legacy overseer baseline snapshot '
    'shape omits it.';

COMMENT ON COLUMN verification_receipt_ledger.receipt_body IS
    'Full raw receipt payload as JSONB — lossless capture of every on-disk shape '
    '(standard ticket receipts, overseer-verify-tick receipts, and the legacy '
    'baseline snapshot). Full-fidelity raw projection (plan §2.5).';

COMMENT ON COLUMN verification_receipt_ledger.ingested_at IS
    'Server-side capture time (DEFAULT NOW()); distinct from sweep_timestamp. '
    'Not a projector-extracted column.';
