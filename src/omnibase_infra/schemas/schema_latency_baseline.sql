-- Materialized View: latency_baseline
-- Description: Hourly latency baseline for A/B comparison (OMN-1890)
-- Created: 2026-02-04
--
-- Purpose: Pre-computes hourly latency statistics per cohort for dashboard queries.
-- Refreshed hourly (not daily) to catch intra-day drift. Dashboard queries use
-- COALESCE with 7-day rolling fallback when sample_count is insufficient.
--
-- Refresh Strategy:
--   - Scheduled via pg_cron: REFRESH MATERIALIZED VIEW CONCURRENTLY latency_baseline;
--   - Recommended schedule: Every hour at :05 (e.g., 00:05, 01:05, ...)
--   - CONCURRENTLY allows queries during refresh (requires UNIQUE index)
--
-- Usage in Dashboard Queries:
--   SELECT
--     ie.session_id,
--     ie.cohort,
--     ie.user_visible_latency_ms,
--     ie.user_visible_latency_ms - COALESCE(
--       lb.avg_latency_ms,
--       (SELECT AVG(avg_latency_ms) FROM latency_baseline
--        WHERE cohort = ie.cohort AND hour > NOW() - INTERVAL '7 days')
--     ) AS latency_delta_ms
--   FROM injection_effectiveness ie
--   LEFT JOIN latency_baseline lb
--     ON DATE_TRUNC('hour', ie.created_at) = lb.hour
--     AND ie.cohort = lb.cohort;
--
-- Related Tickets:
--   - OMN-1890: Store injection metrics with corrected schema

-- ============================================================================
-- MATERIALIZED VIEW
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS latency_baseline AS
SELECT
    DATE_TRUNC('hour', created_at) AS hour,
    cohort,
    AVG(user_visible_latency_ms) AS avg_latency_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY user_visible_latency_ms) AS p50_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY user_visible_latency_ms) AS p95_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY user_visible_latency_ms) AS p99_latency_ms,
    MIN(user_visible_latency_ms) AS min_latency_ms,
    MAX(user_visible_latency_ms) AS max_latency_ms,
    STDDEV(user_visible_latency_ms) AS stddev_latency_ms,
    COUNT(*) AS sample_count
FROM injection_effectiveness
WHERE user_visible_latency_ms IS NOT NULL
  AND cohort IS NOT NULL
GROUP BY DATE_TRUNC('hour', created_at), cohort;

-- ============================================================================
-- INDEXES FOR MATERIALIZED VIEW
-- ============================================================================

-- Required for REFRESH CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS idx_latency_baseline_hour_cohort
    ON latency_baseline (hour, cohort);

-- Fast lookups by cohort
CREATE INDEX IF NOT EXISTS idx_latency_baseline_cohort
    ON latency_baseline (cohort);

-- Recent hour queries
CREATE INDEX IF NOT EXISTS idx_latency_baseline_hour_desc
    ON latency_baseline (hour DESC);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON MATERIALIZED VIEW latency_baseline IS
    'Hourly latency baseline per cohort for A/B testing comparison (OMN-1890). '
    'Refresh hourly via pg_cron. Use COALESCE with 7-day rolling fallback for '
    'hours with insufficient sample_count.';

COMMENT ON COLUMN latency_baseline.hour IS
    'Truncated hour timestamp for aggregation';
COMMENT ON COLUMN latency_baseline.cohort IS
    'A/B test cohort: control or treatment';
COMMENT ON COLUMN latency_baseline.avg_latency_ms IS
    'Average user-visible latency in milliseconds';
COMMENT ON COLUMN latency_baseline.p50_latency_ms IS
    'Median (50th percentile) latency';
COMMENT ON COLUMN latency_baseline.p95_latency_ms IS
    '95th percentile latency for tail analysis';
COMMENT ON COLUMN latency_baseline.p99_latency_ms IS
    '99th percentile latency for extreme tail analysis';
COMMENT ON COLUMN latency_baseline.sample_count IS
    'Number of sessions in this hour/cohort. Check for reliability (recommended N>=20).';

-- ============================================================================
-- HELPER FUNCTION: Refresh with logging
-- ============================================================================

CREATE OR REPLACE FUNCTION refresh_latency_baseline()
RETURNS void AS $$
DECLARE
    start_time TIMESTAMPTZ;
    end_time TIMESTAMPTZ;
    row_count INTEGER;
BEGIN
    start_time := clock_timestamp();

    -- Use CONCURRENTLY to allow queries during refresh
    REFRESH MATERIALIZED VIEW CONCURRENTLY latency_baseline;

    end_time := clock_timestamp();

    -- Get row count for logging
    SELECT COUNT(*) INTO row_count FROM latency_baseline;

    RAISE NOTICE 'latency_baseline refreshed: % rows in % ms',
        row_count,
        EXTRACT(MILLISECONDS FROM (end_time - start_time))::INTEGER;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_latency_baseline() IS
    'Refresh latency_baseline materialized view with timing log. '
    'Call via pg_cron: SELECT refresh_latency_baseline();';
