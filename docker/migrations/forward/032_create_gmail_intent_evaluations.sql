-- Migration: 032_create_gmail_intent_evaluations
-- Predecessor: 031_create_llm_call_metrics_and_cost_aggregates
-- Description: Create gmail_intent_evaluations idempotency table (OMN-2792)
-- Created: 2026-02-25
--
-- Purpose: Stores evaluation results for Gmail messages processed by the
-- Gmail intent evaluator. Prevents duplicate evaluations and duplicate Slack
-- posts under Kafka at-least-once delivery.
--
-- The evaluation_id is a deterministic key derived from the Gmail message ID,
-- so re-delivery of the same Kafka event finds an existing row and skips
-- re-evaluation (ON CONFLICT DO NOTHING in the handler).
--
-- Idempotency: CREATE TABLE IF NOT EXISTS; CREATE INDEX IF NOT EXISTS
-- TTL Column: created_at
--
-- Rollback: See rollback/rollback_032_gmail_intent_evaluations.sql

-- ============================================================================
-- TABLE: gmail_intent_evaluations
-- ============================================================================
-- Stores the outcome of each Gmail intent evaluation to support idempotent
-- processing under Kafka at-least-once delivery.
--
-- Design decisions:
--   - evaluation_id is TEXT (not UUID) because it is a deterministic key
--     derived from the Gmail message ID, not a generated UUID.
--   - verdict is TEXT (not an ENUM) to stay flexible as the evaluator evolves
--     without requiring a schema migration for new verdict types.
--   - relevance_score stores the raw float from the LLM evaluator; the handler
--     is responsible for thresholding and routing decisions.
--   - No FK to any session/run table — the table is written by a Kafka consumer
--     that may run before any session is established.
--   - Append-only pattern: rows are never updated; re-delivery is detected by
--     the primary key constraint (ON CONFLICT DO NOTHING).

CREATE TABLE IF NOT EXISTS gmail_intent_evaluations (
    evaluation_id   TEXT PRIMARY KEY,
    verdict         TEXT NOT NULL,
    relevance_score FLOAT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- INDEXES: gmail_intent_evaluations
-- ============================================================================

-- Time-based queries and TTL cleanup (most recent evaluations first)
CREATE INDEX IF NOT EXISTS idx_gmail_intent_evaluations_created_at
    ON gmail_intent_evaluations (created_at);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE gmail_intent_evaluations IS
    'Idempotency table for Gmail intent evaluations (OMN-2792). Each row '
    'records the result of evaluating one Gmail message. Prevents duplicate '
    'evaluations and Slack posts under Kafka at-least-once delivery.';

COMMENT ON COLUMN gmail_intent_evaluations.evaluation_id IS
    'Deterministic primary key derived from Gmail message ID. '
    'Used as the idempotency key: ON CONFLICT (evaluation_id) DO NOTHING.';

COMMENT ON COLUMN gmail_intent_evaluations.verdict IS
    'Evaluator verdict (e.g., "relevant", "irrelevant"). TEXT (not ENUM) '
    'to allow new verdict types without a schema migration.';

COMMENT ON COLUMN gmail_intent_evaluations.relevance_score IS
    'Raw relevance score from the LLM evaluator (0.0–1.0). The handler '
    'applies thresholding logic; this column stores the raw value for auditing.';

COMMENT ON COLUMN gmail_intent_evaluations.created_at IS
    'Timestamp when the evaluation was recorded. Used for TTL cleanup queries.';

-- ============================================================================
-- UPDATE DB_METADATA
-- ============================================================================

UPDATE public.db_metadata
SET schema_version = '032', updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
