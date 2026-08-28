-- =============================================================================
-- MIGRATION: additive shape reconciliation for the consumer-flow window tables
-- =============================================================================
-- Ticket:  OMN-16777
-- Owner:   omnimarket.nodes.node_projection_consumer_flow
-- Version: 1.0.0
--
-- WHY THIS FILE EXISTS
--   0000 and 0001 both shipped their guarded adds as
--   `ADD COLUMN IF NOT EXISTS <col> <type> NOT NULL`, which cannot reconcile a
--   drifted table that holds rows -- PostgreSQL raises
--     ERROR:  column "<col>" of relation "<t>" contains null values
--   and ON_ERROR_STOP=1 kills the Job. Correcting that means editing two
--   already-DECLARED migrations, and the OMN-16705 append-only guard's only
--   sanctioned escape is a new-ordinal supersession that lands its successor in
--   the same change. This is that successor, and it is not a placeholder: it
--   re-expresses the corrected reconciliation additively, so a database that
--   already holds these relations converges from its own ordinal rather than
--   depending on the rewritten bytes of either earlier file.
--
-- WHY REWRITING 0000 AND 0001 IS SAFE
--   bootstrap.sql raises 'conflicting migration checksum in canonical node
--   history' only when a RECORDED per-migration content_sha256 stops matching,
--   which requires the file to have been APPLIED. Read live 2026-08-28:
--   to_regclass() for both relations is ABSENT on the .201 dev lane, and
--   onex_application_migration_manifest -- the relation bootstrap.sql joins to
--   raise that exception -- exists on NO .201 lane (dev, stability-test, prod,
--   judge), so the node-migration ledger loop has never run and no checksum is
--   recorded anywhere. The supersession rows carry this evidence.
--
-- Every statement below is idempotent and nullable-by-design; see 0001's header
-- for the full reasoning and the RED/GREEN execution evidence.
-- =============================================================================

-- ---- BEGIN OMN-15376 shape reconciliation: omninode_internal.consumer_flow_windows ----
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

-- ---- BEGIN OMN-15376 shape reconciliation: omninode_internal.topic_produce_windows ----
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS topic TEXT;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS window_start TIMESTAMPTZ;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS window_end TIMESTAMPTZ;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS node_id UUID;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS ingest_sequence BIGINT;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS messages_produced BIGINT DEFAULT 0;
ALTER TABLE omninode_internal.topic_produce_windows ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMPTZ;
ALTER TABLE omninode_internal.topic_produce_windows ALTER COLUMN messages_produced SET DEFAULT 0;
-- ---- END OMN-15376 shape reconciliation: omninode_internal.topic_produce_windows ----
