-- Migration 060: Create build_loop_cycles projection table
-- Related: OMN-7321, OMN-5113 (Autonomous Build Loop)
--
-- Projection table for build loop cycle history. Written by the
-- node_autonomous_loop_orchestrator after each cycle completes.

CREATE TABLE IF NOT EXISTS build_loop_cycles (
    id              BIGSERIAL PRIMARY KEY,
    correlation_id  UUID        NOT NULL,
    cycle_number    INTEGER     NOT NULL CHECK (cycle_number >= 1),
    final_phase     TEXT        NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ NOT NULL,
    tickets_filled      INTEGER NOT NULL DEFAULT 0,
    tickets_classified  INTEGER NOT NULL DEFAULT 0,
    tickets_dispatched  INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    dry_run         BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One row per (correlation_id, cycle_number)
    CONSTRAINT uq_build_loop_correlation_cycle
        UNIQUE (correlation_id, cycle_number)
);

-- Index for querying recent cycles
CREATE INDEX IF NOT EXISTS idx_build_loop_cycles_started_at
    ON build_loop_cycles (started_at DESC);

-- Index for correlation-based lookups
CREATE INDEX IF NOT EXISTS idx_build_loop_cycles_correlation
    ON build_loop_cycles (correlation_id);

COMMENT ON TABLE build_loop_cycles IS
    'Projection table for autonomous build loop cycle history (OMN-5113)';
