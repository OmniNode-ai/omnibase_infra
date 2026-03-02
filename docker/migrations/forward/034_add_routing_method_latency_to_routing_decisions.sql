-- OMN-3422: Add routing analytics columns from omniclaude routing events
--
-- routing_method: routing strategy used (SEMANTIC/FALLBACK/RULE_BASED)
--   Used in analytics queries for routing distribution analysis.
--   Sparse index: most rows will have a value; NULL for legacy events only.
--
-- latency_ms: end-to-end routing duration in milliseconds
--   Aggregated only (AVG/P95/MAX); no filter index needed.
--   Only populated by route_via_events_wrapper.py producer.
--
-- Both columns are nullable to allow backfill of existing rows from
-- pre-OMN-3422 events in the DLQ without requiring schema changes.

ALTER TABLE agent_routing_decisions
    ADD COLUMN IF NOT EXISTS routing_method VARCHAR(100),
    ADD COLUMN IF NOT EXISTS latency_ms     INTEGER;

CREATE INDEX IF NOT EXISTS idx_agent_routing_decisions_routing_method
    ON agent_routing_decisions (routing_method)
    WHERE routing_method IS NOT NULL;
