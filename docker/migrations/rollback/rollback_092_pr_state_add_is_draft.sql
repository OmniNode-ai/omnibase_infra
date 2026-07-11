-- OMN-14394: Rollback for 092_pr_state_add_is_draft.sql.
--
-- Reverses addition of the is_draft column on public.pr_state. Manual
-- execution only -- never auto-applied (rollback/ is not mounted to
-- docker-entrypoint-initdb.d).

ALTER TABLE public.pr_state DROP COLUMN IF EXISTS is_draft;
