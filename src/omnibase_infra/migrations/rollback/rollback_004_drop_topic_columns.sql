-- Rollback: rollback_004_drop_topic_columns.sql
-- Purpose: Reverse migration 004 — remove topic columns from registration_projections.
-- Ticket: OMN-2746

DROP INDEX IF EXISTS idx_registration_projections_subscribe_topics;
DROP INDEX IF EXISTS idx_registration_projections_publish_topics;

ALTER TABLE registration_projections
    DROP COLUMN IF EXISTS subscribe_topics,
    DROP COLUMN IF EXISTS publish_topics;
