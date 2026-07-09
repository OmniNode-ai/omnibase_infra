-- Migration: 088_create_overseer_tick_ledger.sql
-- Purpose:   Create the append-only overseer_tick_ledger projection table — the
--            receiving end of the overseer-tick ledger migration (WS-L slice 1).
-- Ticket:    OMN-13996 (child of epic OMN-13989)
-- Plan:      docs/plans/2026-07-05-ledger-learning-substrate-plan.md (§2.5 raw
--            projection, §6 Phase 2)
--
-- Design decisions:
--
--   1. This is the RAW ledger projection (append-only, full-fidelity, replayable)
--      for the overseer-tick control-plane ledger currently written to the flat
--      file .onex_state/overseer-ticks.jsonl by
--      omnimarket/.../node_overnight/handlers/overseer_tick.py::append_tick_log.
--      It is materialized from onex.evt.omnimarket.overseer-tick.v1 events by the
--      generic ProjectorShell driven by overseer_tick_projector.yaml — no bespoke
--      node or class (CLAUDE.md rule 7a; reference pattern: decision_store / 046).
--
--   2. envelope_id is the PRIMARY KEY and the idempotency key. On at-least-once
--      replay a duplicate envelope_id raises a unique violation, which the
--      capture consumer treats as "already ingested" — the same dedup discipline
--      as event_ledger (044). History is never mutated (append mode = plain
--      INSERT; no ON CONFLICT DO UPDATE).
--
--   3. Ordering authority is single_partition (plan §2.4): the overseer-tick
--      topic is low-volume and single-writer, so the per-partition offset is the
--      monotonic cursor. This is declared in the projector contract; recorded
--      here for schema provenance.
--
--   4. ingested_at is set by the database (DEFAULT now()) and is intentionally
--      NOT a projector-extracted column — it is the server-side capture time,
--      distinct from emitted_at (the application-supplied logical emit time).
--
--   5. Idempotent via IF NOT EXISTS so re-running the migration is safe.

-- =============================================================================
-- TABLE: overseer_tick_ledger
-- =============================================================================

CREATE TABLE IF NOT EXISTS overseer_tick_ledger (
    envelope_id                 UUID PRIMARY KEY,
    correlation_id              UUID NOT NULL,
    topic                       TEXT NOT NULL,
    session_id                  TEXT NOT NULL,
    current_phase               TEXT NOT NULL,
    phase_progress              NUMERIC(6, 4) NOT NULL
        CHECK (phase_progress >= 0 AND phase_progress <= 1),
    phase_outcomes              JSONB NOT NULL DEFAULT '{}',
    next_required_outcome       TEXT,
    approaching_halt_conditions JSONB NOT NULL DEFAULT '[]',
    accumulated_cost            NUMERIC(14, 6) NOT NULL DEFAULT 0,
    started_at                  TIMESTAMPTZ NOT NULL,
    emitted_at                  TIMESTAMPTZ NOT NULL,
    ingested_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Object/array-shape guards on the JSONB columns.
    CONSTRAINT chk_overseer_tick_phase_outcomes_is_object
        CHECK (jsonb_typeof(phase_outcomes) = 'object'),
    CONSTRAINT chk_overseer_tick_halt_conditions_is_array
        CHECK (jsonb_typeof(approaching_halt_conditions) = 'array')
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Per-session timeline: "show this overnight session's ticks in emit order".
CREATE INDEX IF NOT EXISTS idx_overseer_tick_ledger_session_emitted
    ON overseer_tick_ledger (session_id, emitted_at DESC);

-- Global recency / cursor scan ordered by emit time.
CREATE INDEX IF NOT EXISTS idx_overseer_tick_ledger_emitted_at
    ON overseer_tick_ledger (emitted_at DESC);

-- Trace correlation lookups.
CREATE INDEX IF NOT EXISTS idx_overseer_tick_ledger_correlation
    ON overseer_tick_ledger (correlation_id);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE overseer_tick_ledger IS
    'Append-only raw projection of the overseer-tick control-plane ledger '
    '(onex.evt.omnimarket.overseer-tick.v1), replacing the flat file '
    '.onex_state/overseer-ticks.jsonl. Materialized by the generic ProjectorShell '
    'from overseer_tick_projector.yaml. WS-L slice 1 (OMN-13996 / epic OMN-13989).';

COMMENT ON COLUMN overseer_tick_ledger.envelope_id IS
    'Event envelope id — primary key and idempotency key. Duplicate replay '
    'raises a unique violation the capture consumer treats as already-ingested.';

COMMENT ON COLUMN overseer_tick_ledger.emitted_at IS
    'Application-supplied logical emit time of the tick (from build_tick_snapshot).';

COMMENT ON COLUMN overseer_tick_ledger.ingested_at IS
    'Server-side capture time (DEFAULT NOW()); distinct from emitted_at. '
    'Not a projector-extracted column.';
