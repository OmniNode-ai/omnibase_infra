-- Rollback: 062_context_enrichment_injection_recorded_tables
-- Description: Drop context enrichment and injection recorded tables (OMN-6158)

DROP TABLE IF EXISTS context_enrichment_events CASCADE;
DROP TABLE IF EXISTS injection_recorded_events CASCADE;
