-- =============================================================================
-- MIGRATION: Create gateway_link_health table + gateway_link_health_status view
-- =============================================================================
-- Ticket: OMN-15570 (Gateway lift Phase 0, item G3)
-- Design source: docs/design/2026-08-08-gateway-node-architecture-lift.md
--   (commit 9422d30b1), §3 Phase 0 row G3; liveness thresholds cited from
--   node_bus_forwarder_effect/contract.yaml:59-63.
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
--   lag data on the heartbeat topic today. G1 (OMN-15741, path-verifying
--   healthcheck) and G2 (OMN-15742, reconnect supervision) have both since
--   merged, but neither added lag telemetry: G2 widened
--   ModelGatewayHeartbeat with status/consecutive_failures/detail, not with
--   lag_messages/lag_seconds. lag_messages/lag_seconds are therefore always
--   NULL today, and gateway_link_health_status only evaluates the
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
--   Hardcoded into the view rather than read from a config table: this is
--   the only tenant profile deployed today (single canary tenant, single
--   contract) and hardcoding the ratchet ties the view directly to the
--   cited contract lines for auditability. Revisit if/when multiple tenant
--   liveness profiles exist.
--
--   OMN-15762 (4th-copy class): this is the 4th independent copy of these
--   three numbers -- contract.yaml, ModelGatewayForwarderConfig defaults,
--   deploy YAML, and now this view -- with no load-time cross-check between
--   them. There is no existing repo pattern for materializing a value into
--   an already-applied SQL view at migration time (the runtime config
--   loader's _materialize_contract_* functions resolve at process start
--   against a live YAML file; a CREATE OR REPLACE VIEW body is baked in
--   once, at migration-apply time, and this table is already live). Rather
--   than leave the drift silent, tests/unit/db/test_migration_100.py pins
--   these three literals against the contract's declared values and fails
--   the suite the moment they diverge -- see OMN-15762 for the tracked
--   de-duplication fix (a single materialized-config source all 4 copies
--   read from).
--
-- IDEMPOTENCY:
--   - CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS are safe to
--     re-run.
--   - CREATE OR REPLACE VIEW is safe to re-run.
--
-- ROLLBACK:
--   docker/migrations/rollback/rollback_100_create_gateway_link_health.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.gateway_link_health (
    tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    local_transport_flavor TEXT NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    lag_messages BIGINT,
    lag_seconds DOUBLE PRECISION,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_gateway_link_health_last_seen_at
    ON public.gateway_link_health (last_seen_at);

COMMENT ON TABLE public.gateway_link_health IS
    'Latest-known-state projection of gateway tenant-edge heartbeats '
    '(OMN-15570, G3). One row per tenant_id, upserted on every heartbeat; '
    'never deleted. Query gateway_link_health_status for live health, not '
    'this table directly -- health is a function of NOW() - last_seen_at, '
    'not a stored column.';

COMMENT ON COLUMN public.gateway_link_health.last_seen_at IS
    'Producer-supplied ModelGatewayHeartbeat.emitted_at -- the freshness '
    'stamp gateway_link_health_status diffs against NOW().';

COMMENT ON COLUMN public.gateway_link_health.lag_messages IS
    'Consumer lag in messages, when a producer supplies it. Always NULL '
    'today -- see migration header SCOPE DISCLOSURE.';

COMMENT ON COLUMN public.gateway_link_health.lag_seconds IS
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
CREATE OR REPLACE VIEW public.gateway_link_health_status AS
SELECT
    tenant_id,
    principal_id,
    local_transport_flavor,
    last_seen_at,
    lag_messages,
    lag_seconds,
    updated_at,
    EXTRACT(EPOCH FROM (NOW() - last_seen_at)) AS seconds_since_last_seen,
    CASE
        WHEN NOW() - last_seen_at > INTERVAL '60 seconds' THEN 'UNHEALTHY'
        WHEN lag_messages IS NOT NULL AND lag_messages > 500 THEN 'DEGRADED_LAG'
        WHEN lag_seconds IS NOT NULL AND lag_seconds > 120 THEN 'DEGRADED_LAG'
        ELSE 'HEALTHY'
    END AS health_status
FROM public.gateway_link_health;

COMMENT ON VIEW public.gateway_link_health_status IS
    'Read-time health evaluation over gateway_link_health. health_status '
    'flips to UNHEALTHY once a row goes stale beyond '
    'max_silence_window_seconds (60s, node_bus_forwarder_effect/contract.yaml:61) '
    'without the row ever disappearing -- absence of progress is visible as '
    'a stale row, not a missing one. DEGRADED_LAG arms are inert until a '
    'producer populates lag_messages/lag_seconds (OMN-15570 scope gap).';
