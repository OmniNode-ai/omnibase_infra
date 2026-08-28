-- =============================================================================
-- MIGRATION: reconcile consumer-flow window columns on drifted deployments
-- =============================================================================
-- Ticket:  OMN-16773
-- Owner:   omnimarket.nodes.node_projection_consumer_flow
-- Version: 1.0.1
--
-- 0000 created the two consumer-flow tables with CREATE TABLE IF NOT EXISTS.
-- A drifted deployment may already have either table with a partial shape; in
-- that case CREATE TABLE is a no-op and later column-dependent statements fail.
-- Keep the applied 0000 bytes frozen and reconcile the declared columns here.
-- =============================================================================

ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS consumer_group TEXT NOT NULL;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS topic TEXT NOT NULL;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS window_start TIMESTAMPTZ NOT NULL;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS window_end TIMESTAMPTZ NOT NULL;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS node_id UUID NOT NULL;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS ingest_sequence BIGINT NOT NULL;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS messages_in BIGINT;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS messages_out BIGINT;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS messages_dlq BIGINT;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS handler_errors BIGINT;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS upstream_produced BIGINT;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS upstream_evidence TEXT NOT NULL;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS flow_state TEXT NOT NULL;
ALTER TABLE omninode_internal.consumer_flow_windows
    ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMPTZ NOT NULL;

ALTER TABLE omninode_internal.topic_produce_windows
    ADD COLUMN IF NOT EXISTS topic TEXT NOT NULL;
ALTER TABLE omninode_internal.topic_produce_windows
    ADD COLUMN IF NOT EXISTS window_start TIMESTAMPTZ NOT NULL;
ALTER TABLE omninode_internal.topic_produce_windows
    ADD COLUMN IF NOT EXISTS window_end TIMESTAMPTZ NOT NULL;
ALTER TABLE omninode_internal.topic_produce_windows
    ADD COLUMN IF NOT EXISTS node_id UUID NOT NULL;
ALTER TABLE omninode_internal.topic_produce_windows
    ADD COLUMN IF NOT EXISTS ingest_sequence BIGINT NOT NULL;
ALTER TABLE omninode_internal.topic_produce_windows
    ADD COLUMN IF NOT EXISTS messages_produced BIGINT NOT NULL DEFAULT 0;
ALTER TABLE omninode_internal.topic_produce_windows
    ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMPTZ NOT NULL;
