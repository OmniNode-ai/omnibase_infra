-- =============================================================================
-- MIGRATION: omninode_internal.gateway_link_health (+ status view), node-owned
-- =============================================================================
-- Ticket: OMN-16759 (re-home the relation onto the loop that can actually
--         deliver it); originally authored as
--         docker/migrations/forward/100_create_gateway_link_health.sql by
--         OMN-15570 (Gateway lift Phase 0, item G3)
-- Design source: docs/design/2026-08-08-gateway-node-architecture-lift.md
--   (commit 9422d30b1), §3 Phase 0 row G3; liveness thresholds cited from
--   node_bus_forwarder_effect/contract.yaml:59-63.
--
-- WHY THIS FILE EXISTS INSTEAD OF THE FLAT 100 (OMN-16759)
--   100 was a FLAT migration. The flat loop connects to exactly one database,
--   `omnibase_infra` (omninode_infra k8s/migrations/omnibase-infra-migrate.yaml,
--   DB_NAME) -- but `omninode_internal` is a schema of the APPLICATION database,
--   physical name `omnidash_analytics`
--   (docker/catalog/database-topology/*.yaml). To make its qualified names
--   resolve in the wrong database, 100 opened with
--
--       CREATE SCHEMA IF NOT EXISTS omninode_internal;
--
--   which needs CREATE on the DATABASE. That aborted the migrate Job with
--
--       psql:/work/100_create_gateway_link_health.sql:77:
--         ERROR:  permission denied for database omnibase_infra
--
--   and, because the Job is migration-order 1 of 6 and runs BEFORE the overlay
--   apply and the runtime digest pin, it blocked EVERY onex-dev staging deploy
--   (omninode_infra run 33080116991), not just the one that surfaced it.
--
--   Read live from the managed instance before writing this file, not inferred:
--     omnibase_infra:      omninode_internal_schema_count=0
--                          has_database_privilege(role_omnibase_infra,
--                            omnibase_infra, CREATE)=false
--                          schema_migrations rows for
--                            '100_create_gateway_link_health.sql' = 0
--     omnidash_analytics:  omninode_internal_schema_count=1
--                          to_regclass('omninode_internal.gateway_link_health')
--                            = ABSENT
--     onex-dev runtime DSN (AWS Secrets Manager onex-dev/omninode-runtime-db):
--                          user omninode_runtime, database omnidash_analytics
--
--   So: the schema does not exist in the database 100 was delivered to and
--   cannot be created there; it DOES exist in the database the writer
--   (HandlerGatewayLinkHealthUpsert) actually connects to; and the relation is
--   still absent, so this migration has real work to do. Asserting the schema
--   in place -- the OMN-16249 remedy for the same statement in
--   nodes/node_projection_registration/0005_create_projection_watermarks.sql --
--   would only have traded the permission error for a divide-by-zero on the
--   same lane. The node loop is the delivery path that reaches
--   omnidash_analytics as its own role, and is what
--   scripts/ci/check_flat_migration_foreign_connect.py names as the fix for a
--   flat migration whose target is a database the flat runner never connects to.
--
--   Ownership is unchanged and was already declared this way before the move:
--   omninode_infra k8s/migrations/application-relation-ownership.yaml records
--   gateway_link_health as database_ref `application`, schema
--   `omninode_internal`. The flat file was the drift; this file matches the
--   declaration.
--
-- SCHEMA PRECONDITION (OMN-16249 probe shape)
--   This file ASSERTS omninode_internal; it does not create it. Reading
--   pg_catalog.pg_namespace needs no schema-level privilege, so it is safe under
--   any connecting role, and integer division by zero fails the migration loudly
--   if the schema is genuinely absent. Statically provable, so it needs no
--   DO/RAISE block (which the repo's dynamic-SQL rejection would refuse).
--   role_omnidash does NOT hold CREATE on omnidash_analytics either
--   (has_database_privilege = false, read live) -- asserting is the only correct
--   shape in this loop, not merely the safer one.
--
-- PURPOSE:
--   Latest-known-state projection, one row per tenant edge (tenant_id),
--   materialized from onex.evt.omnibase-infra.gateway-heartbeat.v1 events.
--   Consumed by node_gateway_link_health_projection_compute (COMPUTE,
--   extracts fields + emits the upsert intent) and persisted by
--   node_gateway_link_health_write_effect (EFFECT, executes the upsert).
--   Mirrors the pr_state (091_pr_state.sql) latest-known-state pattern.
--
-- ABSENCE-OF-PROGRESS DESIGN (the load-bearing property of this migration):
--   ON CONFLICT (tenant_id) DO UPDATE means a tenant edge that stops sending
--   heartbeats keeps its EXISTING row with a staling last_seen_at -- the row
--   is never deleted and never simply absent. gateway_link_health_status
--   (the view below) computes health_status live, at query time, by diffing
--   NOW() against last_seen_at against the contract's
--   max_silence_window_seconds threshold. This is deliberate: a stored,
--   write-time-computed health_status column would always read HEALTHY
--   (it is only written when a heartbeat DOES arrive), which cannot express
--   "went quiet" -- the exact alerting requirement ("alerting on ABSENCE of
--   progress ... never a missing row").
--
-- SCOPE DISCLOSURE (read before trusting lag_messages/lag_seconds):
--   The contract also declares lag_threshold_messages (500) and
--   lag_threshold_seconds (120), but no producer in this codebase publishes
--   lag data on the heartbeat topic today. lag_messages/lag_seconds are
--   therefore always NULL, and gateway_link_health_status only evaluates the
--   silence-window threshold it can actually observe -- a NULL lag column
--   never contributes a false HEALTHY/UNHEALTHY verdict (NULL comparisons
--   are neither TRUE nor FALSE in SQL, so the lag CASE arms are inert until
--   a real producer populates these columns).
--
-- LIVENESS THRESHOLDS (node_bus_forwarder_effect/contract.yaml:59-63,
--   gateway_forwarder.liveness block):
--     max_silence_window_seconds: 60
--     lag_threshold_messages:     500
--     lag_threshold_seconds:      120
--   Hardcoded into the view rather than read from a config table, for the same
--   reason and with the same OMN-15762 4th-copy disclosure the flat file
--   carried: tests/unit/db/test_migration_100.py pins these three literals
--   against the contract's declared values and fails the suite the moment they
--   diverge.
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS are safe to
--     re-run.
--   - CREATE OR REPLACE VIEW is safe to re-run.
--   - NOT safe on a database without the omninode_internal schema: this file
--     asserts that schema rather than creating it (see above), so schema
--     provisioning is a prerequisite, not something this migration performs.
--
-- ROLLBACK:
--   docker/migrations/rollback/rollback_node_gateway_link_health_0001.sql
-- =============================================================================

SELECT 1 / count(*) AS omninode_internal_schema_exists_precondition
  FROM pg_catalog.pg_namespace
 WHERE nspname = 'omninode_internal';

CREATE TABLE IF NOT EXISTS omninode_internal.gateway_link_health (
    tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    local_transport_flavor TEXT NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    reported_status TEXT NOT NULL,
    consecutive_failures INTEGER NOT NULL,
    lag_messages BIGINT,
    lag_seconds DOUBLE PRECISION,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id)
);

-- ---- BEGIN OMN-15376 shape reconciliation: omninode_internal.gateway_link_health ----
-- CREATE TABLE IF NOT EXISTS silently no-ops against a drifted pre-existing
-- table; the guarded adds below converge such a table onto the shape declared
-- above (no-ops on the fresh-create path, since every column already exists
-- there). No DROP, no recreate, no TRUNCATE. Matches
-- node_projection_registration/0005_create_projection_watermarks.sql's
-- precedent. Columns are added nullable: ADD COLUMN ... NOT NULL without a
-- default fails outright on a non-empty drifted table, which would turn a
-- convergence step into the deploy-stopping failure it exists to prevent.
ALTER TABLE omninode_internal.gateway_link_health ADD COLUMN IF NOT EXISTS tenant_id TEXT;
ALTER TABLE omninode_internal.gateway_link_health ADD COLUMN IF NOT EXISTS principal_id TEXT;
ALTER TABLE omninode_internal.gateway_link_health ADD COLUMN IF NOT EXISTS local_transport_flavor TEXT;
ALTER TABLE omninode_internal.gateway_link_health ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ;
ALTER TABLE omninode_internal.gateway_link_health ADD COLUMN IF NOT EXISTS reported_status TEXT;
ALTER TABLE omninode_internal.gateway_link_health ADD COLUMN IF NOT EXISTS consecutive_failures INTEGER;
ALTER TABLE omninode_internal.gateway_link_health ADD COLUMN IF NOT EXISTS lag_messages BIGINT;
ALTER TABLE omninode_internal.gateway_link_health ADD COLUMN IF NOT EXISTS lag_seconds DOUBLE PRECISION;
ALTER TABLE omninode_internal.gateway_link_health ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- ADD COLUMN IF NOT EXISTS ... DEFAULT is a no-op on a column that already
-- existed without one -- restore the declared default explicitly so a drifted
-- pre-existing column converges too, not only a brand-new one.
ALTER TABLE omninode_internal.gateway_link_health ALTER COLUMN updated_at SET DEFAULT NOW();

-- No NOT NULL / PRIMARY KEY convergence block: this relation has never
-- physically existed in the application database (read live before this file
-- was written -- to_regclass('omninode_internal.gateway_link_health') = ABSENT
-- on the managed lane), so there is no drifted row set to reconcile against.
-- The CREATE TABLE above declares the NOT NULLs and the PRIMARY KEY directly,
-- which is sufficient on a genuine fresh-create path.
-- ---- END OMN-15376 shape reconciliation ----

CREATE INDEX IF NOT EXISTS idx_gateway_link_health_last_seen_at
    ON omninode_internal.gateway_link_health (last_seen_at);

COMMENT ON TABLE omninode_internal.gateway_link_health IS
    'Latest-known-state projection of gateway tenant-edge heartbeats '
    '(OMN-15570, G3). One row per tenant_id, upserted on every heartbeat; '
    'never deleted. Query gateway_link_health_status for live health, not '
    'this table directly -- health is a function of NOW() - last_seen_at, '
    'not a stored column.';

COMMENT ON COLUMN omninode_internal.gateway_link_health.last_seen_at IS
    'Producer-supplied ModelGatewayHeartbeat.emitted_at -- the freshness '
    'stamp gateway_link_health_status diffs against NOW().';

COMMENT ON COLUMN omninode_internal.gateway_link_health.reported_status IS
    'The edge''s OWN verdict on itself, verbatim from '
    'ModelGatewayHeartbeat.status (OMN-15742/G2). Stored as free TEXT, not a '
    'CHECK-constrained enum, deliberately: gateway_link_health_status treats '
    'any value other than ''active'' as degraded, so a status the producer '
    'adds later (e.g. a drain state) is read as not-healthy rather than '
    'crashing the projection or being silently scored HEALTHY.';

COMMENT ON COLUMN omninode_internal.gateway_link_health.consecutive_failures IS
    'ModelGatewayHeartbeat.consecutive_failures -- the evidence behind a '
    'degraded reported_status. Recorded so an operator can tell a single '
    'blip from a sustained one without consulting process memory. It drives '
    'no verdict arm of its own: the producer already folds it into '
    'reported_status, and re-deriving it here would let the two disagree.';

COMMENT ON COLUMN omninode_internal.gateway_link_health.lag_messages IS
    'Consumer lag in messages, when a producer supplies it. Always NULL '
    'today -- see migration header SCOPE DISCLOSURE.';

COMMENT ON COLUMN omninode_internal.gateway_link_health.lag_seconds IS
    'Consumer lag in seconds, when a producer supplies it. Always NULL '
    'today -- see migration header SCOPE DISCLOSURE.';

-- =============================================================================
-- READ SURFACE: gateway_link_health_status
-- =============================================================================
-- Live per-tenant health, computed at query time. This is the surface
-- intended for a future omnidash widget (out of scope for OMN-15570 -- see
-- PR body) and for operator/alerting queries today, e.g.:
--   SELECT tenant_id, health_status, seconds_since_last_seen
--   FROM gateway_link_health_status
--   WHERE health_status != 'HEALTHY';
-- PRECEDENCE (fixed, deterministic -- the same row always yields the same
-- verdict, and exactly one arm can win because CASE stops at the first TRUE):
--   1. UNHEALTHY               -- stale beyond the silence window. Ranked
--      above the edge's self-report on purpose: if the edge has stopped
--      talking, its last self-report is stale too and cannot be trusted to
--      still describe reality.
--   2. DEGRADED_SELF_REPORTED  -- the edge is heartbeating on schedule and
--      says it is NOT well. Ranked above lag because it is a direct
--      first-party statement, not an inference from a derived metric.
--   3. DEGRADED_LAG            -- inferred from lag columns (inert today).
--   4. HEALTHY                 -- fresh, self-reporting active, no lag breach.
CREATE OR REPLACE VIEW omninode_internal.gateway_link_health_status AS
SELECT
    tenant_id,
    principal_id,
    local_transport_flavor,
    last_seen_at,
    reported_status,
    consecutive_failures,
    lag_messages,
    lag_seconds,
    updated_at,
    EXTRACT(EPOCH FROM (NOW() - last_seen_at)) AS seconds_since_last_seen,
    CASE
        WHEN NOW() - last_seen_at > INTERVAL '60 seconds' THEN 'UNHEALTHY'
        WHEN reported_status <> 'active' THEN 'DEGRADED_SELF_REPORTED'
        WHEN lag_messages IS NOT NULL AND lag_messages > 500 THEN 'DEGRADED_LAG'
        WHEN lag_seconds IS NOT NULL AND lag_seconds > 120 THEN 'DEGRADED_LAG'
        ELSE 'HEALTHY'
    END AS health_status
FROM omninode_internal.gateway_link_health;

COMMENT ON VIEW omninode_internal.gateway_link_health_status IS
    'Read-time health evaluation over gateway_link_health. health_status '
    'flips to UNHEALTHY once a row goes stale beyond '
    'max_silence_window_seconds (60s, node_bus_forwarder_effect/contract.yaml:61) '
    'without the row ever disappearing -- absence of progress is visible as '
    'a stale row, not a missing one. An edge that IS heartbeating on time but '
    'reports itself degraded scores DEGRADED_SELF_REPORTED, never HEALTHY. '
    'Precedence is fixed (stale > self-reported > lag) and documented above '
    'the view body. DEGRADED_LAG arms remain inert until a producer '
    'populates lag_messages/lag_seconds.';
