-- =============================================================================
-- MIGRATION: per-consumer throughput windows + per-topic production tallies
-- =============================================================================
-- Ticket:  OMN-16777 (Phase 1 of epic OMN-16776 — platform observability)
-- Owner:   omnimarket.nodes.node_projection_consumer_flow
-- Version: 1.0.0
--
-- WHY THIS EXISTS
--   Nothing in the platform measured throughput across a seam. Every liveness
--   signal measured connectedness — group membership, process liveness,
--   container health, LAG — and all four outages on 2026-08-23 were consumers
--   that were connected. The canonical case (OMN-16755):
--   node_gateway_link_health_projection_compute was Stable at LAG 0 with
--   current-offset 15,750 while its declared output topic sat at
--   LOG-END-OFFSET 0. 15,750 in, 0 out, every check green.
--
--   These two tables hold the fact that separates a dead consumer from a quiet
--   one: how many envelopes went in, how many came out, how many were dropped,
--   over one heartbeat window.
--
-- WHY THE COUNTERS ARE NULLABLE
--   A row with messages_in = 0 says "observed, and nothing moved" — that is how
--   IDLE is proven. A row with messages_in = NULL says "this window was never
--   observed" (flow_state = 'UNKNOWN'), which is a different fact. Defaulting
--   these to 0 would let a dropped heartbeat read as a quiet one, which is
--   precisely the false-green this whole epic exists to close (OMN-16777 AC5).
--   That is why there is no DEFAULT 0 here.
--
-- WHY (consumer_group, topic, window_start) IS THE KEY
--   Contract-declared in OMN-16777. window_start is producer-assigned event
--   time, so replaying a window reproduces the same row rather than appending a
--   duplicate. ingest_sequence is the producer's monotonic per-process window
--   counter and is the tie-breaker for ordering — never an ingest clock, which
--   would let a redelivered older window overwrite a newer one.
--
--   Known limitation, recorded rather than hidden: two replicas sharing a
--   consumer group emit distinct window_start values (independent drain
--   timestamps), so they land as distinct rows and the read model aggregates
--   across them. A same-microsecond collision between replicas would resolve
--   last-writer-wins; it is possible and vanishingly unlikely, and no data is
--   summed incorrectly in the normal case.
--
-- WHY omninode_internal, EXPLICITLY QUALIFIED
--   omnibase_infra's scripts/ci/check_application_database_sql.py (OMN-15361)
--   settles this twice over: an UNQUALIFIED application relation target is
--   rejected (a bare CREATE resolves against whatever search_path the runner
--   carries, which is how a table lands in a schema nobody declared), and
--   `public` is prohibited outright for application relations. Older node
--   migrations that are bare or public predate that gate; a new one does not
--   get to inherit them.
--
--   ONE physical relation, not the public/omninode_internal pair
--   node_projection_live_events carries (0000 public + 0002
--   omninode_internal). That pair exists only because its write path drifted
--   after public.live_events already existed and OMN-15819 had to reconcile
--   the two. These tables have never existed anywhere, so there is nothing to
--   reconcile: db_io declares omninode_internal, the writer SQL qualifies
--   omninode_internal, and this migration creates omninode_internal — the
--   reader and the write path cannot resolve to different relations.
--
--   No CREATE SCHEMA here: the node-owned migration loop connects to the
--   application database, where omninode_internal already exists and holds
--   every other node-owned table. The flat loop's CREATE SCHEMA is exactly
--   what failed with "permission denied for database omnibase_infra" in
--   OMN-16759 and blocked every staging deploy.
-- =============================================================================

CREATE TABLE IF NOT EXISTS omninode_internal.consumer_flow_windows (
    consumer_group     TEXT        NOT NULL,
    topic              TEXT        NOT NULL,
    window_start       TIMESTAMPTZ NOT NULL,
    window_end         TIMESTAMPTZ NOT NULL,
    node_id            UUID        NOT NULL,
    ingest_sequence    BIGINT      NOT NULL,

    -- NULL means the window was never observed. NOT the same as 0.
    messages_in        BIGINT,
    messages_out       BIGINT,
    messages_dlq       BIGINT,
    handler_errors     BIGINT,

    -- Envelopes the platform published TO `topic` in an overlapping window.
    -- NULL means no producer of this topic is visible on this rail at all
    -- (an external ingress leg), which is why upstream_evidence exists.
    upstream_produced  BIGINT,
    upstream_evidence  TEXT        NOT NULL,

    -- FLOWING | STALLED | STARVED | IDLE | UNKNOWN. Derived in the projection,
    -- never carried on the producing event (envelope purity).
    flow_state         TEXT        NOT NULL,

    -- Event time (the window's own end), not a wall clock: the row is a
    -- statement about the window, so replay reproduces it byte-identically.
    evaluated_at       TIMESTAMPTZ NOT NULL,

    PRIMARY KEY (consumer_group, topic, window_start)
);

-- ---- BEGIN OMN-15376 shape reconciliation: omninode_internal.consumer_flow_windows ----
-- CREATE TABLE IF NOT EXISTS silently no-ops against a drifted pre-existing
-- table. The CREATE INDEX below is not so forgiving: it guards the index NAME,
-- not the COLUMN, so it raises `column "<col>" does not exist` and
-- ON_ERROR_STOP=1 kills the whole migration Job there (the OMN-15376 /
-- OMN-15302 class, one deploy cycle each). The guarded adds below converge a
-- drifted table onto the shape declared above and are no-ops on the
-- fresh-create path. No DROP, no recreate, no TRUNCATE.
--
-- Columns are added NULLABLE. This is the whole correctness point and is not a
-- style choice: PostgreSQL rejects ADD COLUMN ... NOT NULL on a table that
-- already holds rows unless a DEFAULT is supplied, with
--   ERROR:  column "<col>" of relation "<t>" contains null values
-- and ON_ERROR_STOP=1 kills the migration Job there -- the SAME deploy-stopping
-- failure, at the SAME point in the cycle, that the reconciliation exists to
-- prevent. Proven by execution on the .201 dev lane 2026-08-28 against a seeded
-- drifted table (consumer_flow_windows carrying only consumer_group, one row):
-- the NOT NULL spelling exits 3 on `topic`; this nullable spelling exits 0 and
-- converges all 14 columns. The declared NOT NULLs still hold on the
-- fresh-create path, where the CREATE TABLE above establishes them directly.
--
-- No NOT NULL / PRIMARY KEY convergence block: these relations have never
-- physically existed. Read live 2026-08-28 --
-- to_regclass('omninode_internal.consumer_flow_windows') and
-- to_regclass('omninode_internal.topic_produce_windows') are both ABSENT on the
-- .201 dev lane, and onex_application_migration_manifest exists on no lane at
-- all -- so there is no drifted row set to reconcile against.
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS consumer_group TEXT;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS topic TEXT;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS window_start TIMESTAMPTZ;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS window_end TIMESTAMPTZ;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS node_id UUID;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS ingest_sequence BIGINT;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS messages_in BIGINT;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS messages_out BIGINT;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS messages_dlq BIGINT;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS handler_errors BIGINT;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS upstream_produced BIGINT;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS upstream_evidence TEXT;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS flow_state TEXT;
ALTER TABLE omninode_internal.consumer_flow_windows ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMPTZ;
-- ---- END OMN-15376 shape reconciliation: omninode_internal.consumer_flow_windows ----

-- The two queries this table exists to answer: "what is not flowing right now"
-- and "what did consumer X do in the last hour".
CREATE INDEX IF NOT EXISTS idx_consumer_flow_windows_state_time
    ON omninode_internal.consumer_flow_windows (flow_state, window_end DESC);

CREATE INDEX IF NOT EXISTS idx_consumer_flow_windows_group_time
    ON omninode_internal.consumer_flow_windows (consumer_group, window_end DESC);

CREATE TABLE IF NOT EXISTS omninode_internal.topic_produce_windows (
    topic              TEXT        NOT NULL,
    window_start       TIMESTAMPTZ NOT NULL,
    window_end         TIMESTAMPTZ NOT NULL,
    node_id            UUID        NOT NULL,
    ingest_sequence    BIGINT      NOT NULL,
    messages_produced  BIGINT      NOT NULL DEFAULT 0,
    evaluated_at       TIMESTAMPTZ NOT NULL,

    PRIMARY KEY (topic, window_start)
);

-- ---- BEGIN OMN-15376 shape reconciliation: omninode_internal.topic_produce_windows ----
-- Same class, same reasoning, same nullable rule as consumer_flow_windows above.
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS topic TEXT;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS window_start TIMESTAMPTZ;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS window_end TIMESTAMPTZ;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS node_id UUID;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS ingest_sequence BIGINT;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS messages_produced BIGINT DEFAULT 0;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMPTZ;

-- ADD COLUMN IF NOT EXISTS ... DEFAULT is a no-op on a column that already
-- existed without one -- restore the declared default explicitly so a drifted
-- pre-existing column converges too, not only a brand-new one. Only
-- messages_produced declares a DEFAULT; the counters on consumer_flow_windows
-- deliberately have none (NULL means "never observed", which is not 0).
ALTER TABLE omninode_internal.topic_produce_windows ALTER COLUMN messages_produced SET DEFAULT 0;
-- ---- END OMN-15376 shape reconciliation: omninode_internal.topic_produce_windows ----

CREATE INDEX IF NOT EXISTS idx_topic_produce_windows_topic_time
    ON omninode_internal.topic_produce_windows (topic, window_end DESC);
