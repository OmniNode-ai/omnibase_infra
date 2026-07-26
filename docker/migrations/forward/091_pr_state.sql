-- =============================================================================
-- MIGRATION: Create pr_state table
-- =============================================================================
-- Ticket: OMN-14375 (WS-L fan-in producer)
-- Epic context: GitHub-state projection (webhook/poller -> local pr_state
--   projection so agents read PR/CI/review/merge-queue state without live gh).
-- Version: 1.0.0
--
-- PURPOSE:
--   Latest-known-state projection, one row per (repo, pr_number), materialized
--   from GitHub PR status events. Consumed by node_pr_state_projection_compute
--   (COMPUTE, extracts fields + emits the upsert intent) and persisted by
--   node_pr_state_write_effect (EFFECT, executes the upsert).
--
-- DESIGN — LATEST-STATE PROJECTION (contrast with build_loop_runs, which is
--   append-only audit): pr_state answers "what is PR #n's state right now",
--   not "show me every event". ON CONFLICT (repo, pr_number) DO UPDATE keeps
--   exactly one row per PR, refreshed on every producer cycle.
--
-- SEAM (shared with the OMN-14374 structured-only read skill):
--   Columns below are the full target schema per the OMN-14375 acceptance
--   criteria (PR status, CI conclusions, review-thread state, merge-queue
--   state). The interim producer (node_github_pr_poller_effect) only
--   populates triage_state/title today via onex.evt.github.pr-status.v1;
--   ci_status/review_decision/mergeable/merge_state_status/merge_queue_state
--   stay NULL until a richer poller payload or the webhook receiver lands
--   (both tracked under the OMN-14375 epic, not re-litigated here). Consumers
--   must treat NULL in those columns as "not yet populated by any producer",
--   not "known-empty".
--
-- FRESHNESS:
--   as_of is producer-supplied event time (when GitHub reported this state);
--   projected_at is DB write time. A reader checks `now() - as_of` against its
--   own staleness threshold rather than trusting row presence alone — this is
--   the "stale -> explicit, never silently served as current" acceptance
--   criterion; enforcing a threshold is a reader-side (OMN-14374) concern.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS is safe to re-run.
--   - CREATE INDEX IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   docker/migrations/rollback/rollback_091_pr_state.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.pr_state (
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    triage_state TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    ci_status TEXT,
    review_decision TEXT,
    mergeable TEXT,
    merge_state_status TEXT,
    merge_queue_state TEXT,
    base_ref TEXT,
    head_ref TEXT,
    source TEXT NOT NULL DEFAULT 'poller',
    correlation_id UUID,
    as_of TIMESTAMPTZ NOT NULL,
    projected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (repo, pr_number)
);

CREATE INDEX IF NOT EXISTS pr_state_as_of_idx
    ON public.pr_state (as_of DESC);
