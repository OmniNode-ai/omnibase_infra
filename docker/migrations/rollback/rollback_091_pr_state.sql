-- OMN-14375: Rollback for 091_pr_state.sql.
--
-- Reverses creation of the pr_state latest-known-state projection table and
-- its freshness index. Manual execution only -- never auto-applied (rollback/
-- is not mounted to docker-entrypoint-initdb.d).

DROP INDEX IF EXISTS public.pr_state_as_of_idx;
DROP TABLE IF EXISTS public.pr_state;
