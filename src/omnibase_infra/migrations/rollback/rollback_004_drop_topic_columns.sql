-- Rollback: rollback_004_drop_topic_columns.sql
-- Purpose: Reverse migration 004 — remove topic columns from node_registrations.
-- Ticket: OMN-2746

DROP INDEX IF EXISTS idx_node_registrations_subscribe_topics;
DROP INDEX IF EXISTS idx_node_registrations_publish_topics;

ALTER TABLE node_registrations
    DROP COLUMN IF EXISTS subscribe_topics,
    DROP COLUMN IF EXISTS publish_topics;
