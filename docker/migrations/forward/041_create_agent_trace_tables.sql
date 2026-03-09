-- =============================================================================
-- MIGRATION: Create Agent Trace system tables
-- =============================================================================
-- Ticket: OMN-4080 (Lift migration freeze OMN-2055; apply TRACE_MIGRATION_DDL)
-- Source: omniclaude/src/omniclaude/trace/db_schema.py (TRACE_MIGRATION_DDL)
-- Version: 1.0.0
--
-- PURPOSE:
--   Creates the 5 Agent Trace / Debug Intelligence tables that were blocked
--   by migration freeze OMN-2055. These tables back the FixTransition,
--   ChangeFrame, FailureSignature, and PREnvelope persistence layer in
--   omniclaude. OMN-3556 (Debug Intelligence Phase 1) depends on this schema.
--
-- TABLES (in dependency order):
--   1. failure_signatures    — unique failure fingerprints
--   2. change_frames         — immutable agent execution records
--   3. pr_envelopes          — PR containers for frames
--   4. frame_pr_association  — many-to-many frames <-> PRs
--   5. fix_transitions       — failure -> success delta pairs
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   See rollback/rollback_041_create_agent_trace_tables.sql
-- =============================================================================

-- Failure signatures must be created first (referenced by change_frames)
CREATE TABLE IF NOT EXISTS failure_signatures (
    signature_id        TEXT            PRIMARY KEY,
    failure_type        TEXT            NOT NULL,
    primary_signal      TEXT            NOT NULL,
    fingerprint         TEXT            NOT NULL UNIQUE,
    repro_command       TEXT            NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_failure_signatures_fingerprint
    ON failure_signatures (fingerprint);

-- =============================================================================

-- Main frame table (immutable records — no updates, no deletes)
CREATE TABLE IF NOT EXISTS change_frames (
    frame_id            UUID            PRIMARY KEY,
    parent_frame_id     UUID            REFERENCES change_frames (frame_id),
    timestamp_utc       TIMESTAMPTZ     NOT NULL,
    agent_id            TEXT            NOT NULL,
    model_id            TEXT            NOT NULL,
    base_commit         TEXT            NOT NULL,
    repo                TEXT            NOT NULL,
    branch_name         TEXT            NOT NULL,
    outcome_status      TEXT            NOT NULL,
    failure_signature_id TEXT           REFERENCES failure_signatures (signature_id),
    frame_blob_ref      TEXT            NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_change_frames_failure_sig
    ON change_frames (failure_signature_id);

CREATE INDEX IF NOT EXISTS idx_change_frames_base_commit
    ON change_frames (base_commit);

-- =============================================================================

-- PR envelopes (containers for frames)
CREATE TABLE IF NOT EXISTS pr_envelopes (
    pr_id               UUID            PRIMARY KEY,
    repo                TEXT            NOT NULL,
    pr_number           INT             NOT NULL,
    head_sha            TEXT            NOT NULL,
    base_sha            TEXT            NOT NULL,
    branch_name         TEXT            NOT NULL,
    merged_at           TIMESTAMPTZ,
    envelope_blob_ref   TEXT            NOT NULL
);

-- =============================================================================

-- Frame to PR association (many-to-many)
CREATE TABLE IF NOT EXISTS frame_pr_association (
    frame_id            UUID            REFERENCES change_frames (frame_id),
    pr_id               UUID            REFERENCES pr_envelopes (pr_id),
    association_method  TEXT            NOT NULL,
    PRIMARY KEY (frame_id, pr_id)
);

CREATE INDEX IF NOT EXISTS idx_frame_pr_association_pr_id
    ON frame_pr_association (pr_id);

-- =============================================================================

-- Fix transitions (failure -> success pairs)
CREATE TABLE IF NOT EXISTS fix_transitions (
    transition_id       UUID            PRIMARY KEY,
    failure_signature_id TEXT           REFERENCES failure_signatures (signature_id),
    initial_frame_id    UUID            NOT NULL REFERENCES change_frames (frame_id),
    success_frame_id    UUID            NOT NULL REFERENCES change_frames (frame_id),
    delta_hash          TEXT            NOT NULL,
    files_involved      TEXT[]          NOT NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fix_transitions_failure_sig
    ON fix_transitions (failure_signature_id);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE failure_signatures IS
    'Unique failure fingerprints for the Agent Trace system (OMN-4080). '
    'Immutable after insert. Referenced by change_frames and fix_transitions.';

COMMENT ON TABLE change_frames IS
    'Immutable agent execution records for the Agent Trace system (OMN-4080). '
    'No updates or deletes — append-only. Records agent state at each step.';

COMMENT ON TABLE pr_envelopes IS
    'PR containers grouping related change_frames (OMN-4080). '
    'Linked to frames via frame_pr_association (many-to-many).';

COMMENT ON TABLE frame_pr_association IS
    'Many-to-many association between change_frames and pr_envelopes (OMN-4080).';

COMMENT ON TABLE fix_transitions IS
    'Failure-to-success delta pairs for debug intelligence (OMN-4080). '
    'Backs FixTransition persistence in omniclaude. Used by OMN-3556.';

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '041',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
