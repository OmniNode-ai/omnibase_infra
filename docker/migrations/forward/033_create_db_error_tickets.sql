-- Migration 033: Create db_error_tickets table
--
-- Stores deduplication state and frequency tracking for PostgreSQL errors
-- reported by NodeDbErrorLinearEffect -> HandlerLinearDbErrorReporter.
--
-- Each unique error is identified by a 32-char SHA-256 fingerprint
-- (computed by the emitter in scripts/monitor_logs.py).  A fingerprint
-- that already exists causes occurrence_count to increment rather than
-- creating a new Linear ticket.
--
-- Related Tickets:
--   OMN-3408: Kafka Consumer -> Linear Ticket Reporter (ONEX Node)
--   OMN-3407: PostgreSQL Error Emitter (hard prerequisite)

CREATE TABLE IF NOT EXISTS db_error_tickets (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    fingerprint       VARCHAR(32) NOT NULL UNIQUE,
    error_code        VARCHAR(10),
    error_message     TEXT        NOT NULL,
    table_name        TEXT,
    service           TEXT        NOT NULL,
    linear_issue_id   TEXT        NOT NULL,
    linear_issue_url  TEXT,
    occurrence_count  INT         NOT NULL DEFAULT 1,
    first_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Note: fingerprint has a UNIQUE constraint which auto-creates a btree index;
-- no additional index is needed for the dedup check hot path.

-- Index for recency queries (most recently active errors first)
CREATE INDEX IF NOT EXISTS idx_db_error_tickets_last_seen_at
    ON db_error_tickets (last_seen_at DESC);
