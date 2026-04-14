-- =============================================================================
-- MIGRATION: Add quality_score column to routing_outcomes
-- =============================================================================
-- Ticket: OMN-new (Add routing_outcomes.quality_score column)
-- Epic: OMN-7264 (Intelligent Model Router MVP)
-- Version: 1.0.0
--
-- PURPOSE:
--   Adds quality_score DOUBLE PRECISION column to routing_outcomes table.
--   Required by node_platform_readiness._check_quality_score_coverage()
--   which queries WHERE quality_score IS NOT NULL to assess dimension health.
--   Missing column causes platform_readiness dimension to FAIL on every check.
--
-- IDEMPOTENCY:
--   - ALTER TABLE ... ADD COLUMN IF NOT EXISTS is safe to re-run.
--
-- ROLLBACK:
--   docker/migrations/rollback/rollback_066_add_quality_score_to_routing_outcomes.sql
-- =============================================================================

ALTER TABLE public.routing_outcomes
    ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION;

COMMENT ON COLUMN routing_outcomes.quality_score IS
    'Overall quality score for the routing outcome (0.0-1.0). '
    'Populated by node_routing_score_reducer after outcome evaluation. '
    'NULL until score is computed. (OMN-new)';

-- Update migration sentinel
UPDATE public.db_metadata
SET migrations_complete = TRUE,
    schema_version = '066',
    updated_at = NOW()
WHERE id = TRUE AND owner_service = 'omnibase_infra';
