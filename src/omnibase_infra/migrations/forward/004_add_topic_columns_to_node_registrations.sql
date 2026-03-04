-- Migration: 004_add_topic_columns_to_registration_projections.sql
-- Purpose: Add subscribe_topics and publish_topics columns to registration_projections
--          to support PostgreSQL-backed ServiceTopicCatalogPostgres (OMN-2746).
-- Author: ONEX Infrastructure Team
-- Date: 2026-02-25
-- Ticket: OMN-2746
--
-- Design Decisions:
--
--   1. JSONB columns store topic string arrays directly in the registrations row.
--      This is the canonical source of truth replacing the former Consul KV path
--      onex/nodes/{id}/event_bus/subscribe_topics.
--
--   2. DEFAULT '[]' ensures existing rows are not broken; the columns start empty
--      and are populated on next registration upsert.
--
--   3. Idempotent via IF NOT EXISTS / ALTER TABLE ADD COLUMN IF NOT EXISTS
--      so re-running the migration is safe.
--
--   4. No NOT NULL constraint on the arrays themselves — the DEFAULT '[]' covers
--      the initial empty state. The application layer enforces non-null via its
--      upsert SQL.

-- =============================================================================
-- ALTER TABLE: registration_projections — add topic columns
-- =============================================================================

ALTER TABLE registration_projections
    ADD COLUMN IF NOT EXISTS subscribe_topics JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS publish_topics   JSONB NOT NULL DEFAULT '[]';

-- =============================================================================
-- INDEXES: registration_projections topic columns
-- =============================================================================

-- GIN index for JSONB containment queries: which nodes subscribe to a given topic?
CREATE INDEX IF NOT EXISTS idx_registration_projections_subscribe_topics
    ON registration_projections USING GIN (subscribe_topics);

CREATE INDEX IF NOT EXISTS idx_registration_projections_publish_topics
    ON registration_projections USING GIN (publish_topics);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON COLUMN registration_projections.subscribe_topics IS
    'JSONB array of topic suffix strings this node subscribes to. '
    'Source of truth for ServiceTopicCatalogPostgres (OMN-2746). '
    'Replaces Consul KV path onex/nodes/{id}/event_bus/subscribe_topics.';

COMMENT ON COLUMN registration_projections.publish_topics IS
    'JSONB array of topic suffix strings this node publishes to. '
    'Source of truth for ServiceTopicCatalogPostgres (OMN-2746). '
    'Replaces Consul KV path onex/nodes/{id}/event_bus/publish_topics.';
