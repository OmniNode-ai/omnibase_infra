-- OMN-9774: Rollback for 070_build_loop_runs.sql.
--
-- Reverses creation of the build_loop_runs append-only audit table and its
-- per-workflow index. Manual execution only — never auto-applied (rollback/
-- is not mounted to docker-entrypoint-initdb.d).

DROP INDEX IF EXISTS build_loop_runs_workflow_created_idx;
DROP TABLE IF EXISTS build_loop_runs;
