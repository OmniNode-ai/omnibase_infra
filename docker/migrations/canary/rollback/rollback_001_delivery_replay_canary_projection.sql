-- =============================================================================
-- ROLLBACK: rollback_001_delivery_replay_canary_projection (managed-staging canary)
-- =============================================================================
-- Reverses: docker/migrations/canary/forward/001_create_delivery_replay_canary_projection.sql
-- Ticket:   OMN-14737 (Lane B / task B12)
--
-- Drops the canary delivery/replay landing table, its trigger, and the trigger
-- function. Scoped to the DEDICATED canary logical DB only — run with
-- `psql -d <canary_db>` (default `omninode_canary_mstg1`), NEVER against
-- omnibase_infra or any other logical DB on omninode-dev-postgres.
--
-- This is table-level teardown only. Dropping the canary logical DB itself and
-- rotating/revoking its runtime creds is the B13 teardown step (T-5) — see
-- docs/runbooks/managed-staging-canary-teardown-rollback.md. Per B13, the
-- DEFAULT is to RETAIN the DB read-only as forensic evidence until the OCC
-- receipt is durable, then drop.
--
-- HELD FOR OPERATOR: never auto-applied; run manually only, one gated step at a
-- time. Rollback never deletes the durable evidence packet.
-- =============================================================================

DROP TRIGGER IF EXISTS trg_delivery_replay_canary_projection_updated_at
    ON delivery_replay_canary_projection;

DROP FUNCTION IF EXISTS refresh_delivery_replay_canary_projection_updated_at();

DROP TABLE IF EXISTS delivery_replay_canary_projection;
