-- =============================================================================
-- MIGRATION: Add is_draft column to pr_state
-- =============================================================================
-- Ticket: OMN-14394 (seam gap follow-up to OMN-14375's 091_pr_state.sql)
--
-- PURPOSE:
--   091_pr_state.sql shipped without a draft-equivalent column. The OMN-14374
--   read skill's interim output shape (ModelOpenPrSummary.is_draft in
--   omnimarket's node_github_repo_gateway_effect) has a required, always-
--   populated `is_draft: bool` field with zero counterpart in pr_state --
--   if a future consumer cuts over from live-gh polling to reading this
--   projection table, the draft signal would silently disappear.
--
-- SEAM MATCH (OMN-14208 field-by-field discipline):
--   pr_state.is_draft (this column) <-> ModelPayloadPrStateUpsert.is_draft
--   (node_pr_state_projection_compute) <-> ModelOpenPrSummary.is_draft
--   (omnimarket, reader). All three are non-nullable bool -- unlike
--   ci_status/review_decision/mergeable/merge_state_status/merge_queue_state,
--   which stay NULL pending a richer producer, the poller can always
--   determine draft status from `pr["draft"]` today, so this column is
--   populated on day one (NOT NULL DEFAULT false, not a reserved column).
--
-- IDEMPOTENCY:
--   ALTER TABLE ... ADD COLUMN IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   docker/migrations/rollback/rollback_092_pr_state_add_is_draft.sql
-- =============================================================================

ALTER TABLE public.pr_state
    ADD COLUMN IF NOT EXISTS is_draft BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN public.pr_state.is_draft IS
    'GitHub PR draft status, mirrored from the poller''s pr["draft"] field. '
    'Matches ModelOpenPrSummary.is_draft (omnimarket reader) field-for-field '
    '-- see OMN-14394.';
