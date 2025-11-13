# Database Performance Optimization Guide

## Current Schema Analysis

### Existing Tables and Indexes

#### 1. service_sessions
**Current Indexes:**
- `idx_service_sessions_service_status` (service_name, status) WHERE status = 'active'
- `idx_service_sessions_created_at` (created_at DESC)
- `idx_service_sessions_cleanup` (session_end) WHERE session_end IS NOT NULL

**Optimization Recommendations:**

```sql
-- Add composite index for common queries
CREATE INDEX CONCURRENTLY idx_service_sessions_service_time
ON service_sessions(service_name, created_at DESC)
WHERE status = 'active';

-- Add index for session duration analysis
CREATE INDEX CONCURRENTLY idx_service_sessions_duration
ON service_sessions(service_name, (EXTRACT(EPOCH FROM (session_end - session_start))))
WHERE session_end IS NOT NULL;

-- Optimize instance tracking
CREATE INDEX CONCURRENTLY idx_service_sessions_instance
ON service_sessions(instance_id, status, updated_at DESC)
WHERE instance_id IS NOT NULL;
```

#### 2. hook_events
**Current Indexes:**
- `idx_hook_events_source_action` (source, action)
- `idx_hook_events_processed` (processed, created_at DESC) WHERE NOT processed
- `idx_hook_events_created_at` (created_at DESC)
- `idx_hook_events_retry` (retry_count, created_at) WHERE retry_count > 0

**Optimization Recommendations:**

```sql
-- Optimize event processing pipeline
CREATE INDEX CONCURRENTLY idx_hook_events_processing_queue
ON hook_events(created_at, retry_count)
WHERE NOT processed AND retry_count < 10;

-- Add resource-specific queries
CREATE INDEX CONCURRENTLY idx_hook_events_resource_lookup
ON hook_events(resource, resource_id, created_at DESC);

-- Optimize error analysis
CREATE INDEX CONCURRENTLY idx_hook_events_error_analysis
ON hook_events(source, action, (array_length(processing_errors, 1)))
WHERE array_length(processing_errors, 1) > 0;

-- Add payload-based search (for specific payload queries)
CREATE INDEX CONCURRENTLY idx_hook_events_payload_type
ON hook_events USING GIN((payload->>'type'))
WHERE payload ? 'type';
```

#### 3. event_metrics
**Current Indexes:**
- `idx_event_metrics_event_id` (event_id)
- `idx_event_metrics_created_at` (created_at DESC)
- `idx_event_metrics_performance` (processing_time_ms, kafka_publish_success)

**Optimization Recommendations:**

```sql
-- Optimize performance analysis queries
CREATE INDEX CONCURRENTLY idx_event_metrics_performance_analysis
ON event_metrics(created_at, processing_time_ms)
WHERE kafka_publish_success = true;

-- Add error analysis index
CREATE INDEX CONCURRENTLY idx_event_metrics_errors
ON event_metrics(created_at, error_message)
WHERE kafka_publish_success = false AND error_message IS NOT NULL;

-- Percentile calculations optimization
CREATE INDEX CONCURRENTLY idx_event_metrics_time_buckets
ON event_metrics(
    date_trunc('hour', created_at),
    processing_time_ms,
    kafka_publish_success
);
```

#### 4. security_audit_log
**Current Indexes:**
- `idx_security_audit_event_type` (event_type, created_at DESC)
- `idx_security_audit_failed` (created_at DESC) WHERE NOT success

**Optimization Recommendations:**

```sql
-- Optimize security monitoring
CREATE INDEX CONCURRENTLY idx_security_audit_client_tracking
ON security_audit_log USING GIN((client_info->>'ip_address'))
WHERE client_info ? 'ip_address';

-- Add time-based security analysis
CREATE INDEX CONCURRENTLY idx_security_audit_failure_patterns
ON security_audit_log(
    event_type,
    date_trunc('minute', created_at),
    success
) WHERE NOT success;
```

#### 5. connection_metrics
**Current Indexes:**
- `idx_connection_metrics_recorded_at` (recorded_at DESC)

**Optimization Recommendations:**

```sql
-- Optimize connection monitoring
CREATE INDEX CONCURRENTLY idx_connection_metrics_analysis
ON connection_metrics(
    date_trunc('hour', recorded_at),
    pool_size,
    active_connections
);

-- Add performance trending
CREATE INDEX CONCURRENTLY idx_connection_metrics_trending
ON connection_metrics(recorded_at, avg_query_time_ms)
WHERE avg_query_time_ms IS NOT NULL;
```

## Advanced Optimization Strategies

### 1. Partitioning for Time-Series Data

```sql
-- Partition hook_events by month for better performance
CREATE TABLE hook_events_partitioned (
    LIKE hook_events INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE hook_events_2024_01 PARTITION OF hook_events_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Auto-partition creation function
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name text, start_date date)
RETURNS void AS $$
DECLARE
    partition_name text;
    end_date date;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + interval '1 month';

    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
                    FOR VALUES FROM (%L) TO (%L)',
                    partition_name, table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;
```

### 2. Materialized Views for Analytics

```sql
-- Performance summary materialized view
CREATE MATERIALIZED VIEW performance_summary AS
SELECT
    date_trunc('hour', created_at) as hour,
    count(*) as total_events,
    avg(processing_time_ms) as avg_processing_time,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY processing_time_ms) as p95_processing_time,
    count(*) FILTER (WHERE kafka_publish_success) as successful_events,
    count(*) FILTER (WHERE NOT kafka_publish_success) as failed_events
FROM event_metrics
WHERE created_at >= now() - interval '30 days'
GROUP BY date_trunc('hour', created_at);

CREATE UNIQUE INDEX ON performance_summary (hour);

-- Refresh function
CREATE OR REPLACE FUNCTION refresh_performance_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY performance_summary;
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh every 15 minutes
SELECT cron.schedule('refresh-performance-summary', '*/15 * * * *', 'SELECT refresh_performance_summary();');
```

### 3. Query Optimization Functions

```sql
-- Optimized function for event processing queue
CREATE OR REPLACE FUNCTION get_pending_events(limit_count int DEFAULT 100)
RETURNS TABLE(
    id uuid,
    source varchar,
    action varchar,
    resource varchar,
    resource_id varchar,
    payload jsonb,
    retry_count int
) AS $$
BEGIN
    RETURN QUERY
    SELECT he.id, he.source, he.action, he.resource, he.resource_id, he.payload, he.retry_count
    FROM hook_events he
    WHERE NOT he.processed
      AND he.retry_count < 10
      AND he.created_at > now() - interval '1 hour'
    ORDER BY he.created_at, he.retry_count
    LIMIT limit_count
    FOR UPDATE SKIP LOCKED;
END;
$$ LANGUAGE plpgsql;

-- Performance metrics aggregation
CREATE OR REPLACE FUNCTION get_performance_metrics(
    start_time timestamp with time zone,
    end_time timestamp with time zone
)
RETURNS TABLE(
    metric_name text,
    metric_value numeric,
    metric_unit text
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        'avg_processing_time'::text,
        round(avg(processing_time_ms)::numeric, 2),
        'ms'::text
    FROM event_metrics
    WHERE created_at BETWEEN start_time AND end_time
      AND kafka_publish_success = true

    UNION ALL

    SELECT
        'p95_processing_time'::text,
        round(percentile_cont(0.95) WITHIN GROUP (ORDER BY processing_time_ms)::numeric, 2),
        'ms'::text
    FROM event_metrics
    WHERE created_at BETWEEN start_time AND end_time
      AND kafka_publish_success = true

    UNION ALL

    SELECT
        'error_rate'::text,
        round(
            (count(*) FILTER (WHERE NOT kafka_publish_success)::numeric /
             count(*)::numeric * 100), 2
        ),
        'percent'::text
    FROM event_metrics
    WHERE created_at BETWEEN start_time AND end_time;
END;
$$ LANGUAGE plpgsql;
```

### 4. Connection Pool Optimization

```sql
-- Connection pool monitoring
CREATE OR REPLACE FUNCTION monitor_connection_pool()
RETURNS TABLE(
    metric text,
    value numeric,
    status text
) AS $$
DECLARE
    active_connections int;
    total_connections int;
    max_connections int;
    utilization_percent numeric;
BEGIN
    -- Get current connection statistics
    SELECT count(*)::int INTO active_connections
    FROM pg_stat_activity
    WHERE state = 'active' AND application_name LIKE 'omninode_bridge%';

    SELECT count(*)::int INTO total_connections
    FROM pg_stat_activity
    WHERE application_name LIKE 'omninode_bridge%';

    SELECT setting::int INTO max_connections
    FROM pg_settings
    WHERE name = 'max_connections';

    utilization_percent := (total_connections::numeric / max_connections::numeric) * 100;

    -- Return metrics
    RETURN QUERY VALUES
        ('active_connections'::text, active_connections::numeric,
         CASE WHEN active_connections > 50 THEN 'warning' ELSE 'ok' END::text),
        ('total_connections'::text, total_connections::numeric,
         CASE WHEN total_connections > max_connections * 0.8 THEN 'warning' ELSE 'ok' END::text),
        ('utilization_percent'::text, round(utilization_percent, 2),
         CASE
            WHEN utilization_percent > 90 THEN 'critical'
            WHEN utilization_percent > 70 THEN 'warning'
            ELSE 'ok'
         END::text);
END;
$$ LANGUAGE plpgsql;
```

## Query Pattern Optimizations

### 1. Batch Operations

```sql
-- Batch event insertion with conflict handling
CREATE OR REPLACE FUNCTION batch_insert_hook_events(events jsonb)
RETURNS int AS $$
DECLARE
    inserted_count int := 0;
    event_record jsonb;
BEGIN
    FOR event_record IN SELECT jsonb_array_elements(events)
    LOOP
        INSERT INTO hook_events (
            id, source, action, resource, resource_id,
            payload, metadata, processed, processing_errors, retry_count
        ) VALUES (
            (event_record->>'id')::uuid,
            event_record->>'source',
            event_record->>'action',
            event_record->>'resource',
            event_record->>'resource_id',
            event_record->'payload',
            event_record->'metadata',
            (event_record->>'processed')::boolean,
            ARRAY(SELECT jsonb_array_elements_text(event_record->'processing_errors')),
            (event_record->>'retry_count')::int
        )
        ON CONFLICT (id) DO UPDATE SET
            updated_at = NOW(),
            retry_count = EXCLUDED.retry_count,
            processing_errors = EXCLUDED.processing_errors;

        inserted_count := inserted_count + 1;
    END LOOP;

    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;
```

### 2. Efficient Cleanup Operations

```sql
-- Optimized cleanup with batching
CREATE OR REPLACE FUNCTION cleanup_old_data_optimized(batch_size int DEFAULT 1000)
RETURNS TABLE(
    table_name text,
    deleted_count bigint,
    execution_time_ms numeric
) AS $$
DECLARE
    start_time timestamp;
    end_time timestamp;
    deleted_rows bigint;
BEGIN
    -- Hook events cleanup
    start_time := clock_timestamp();

    WITH deleted AS (
        DELETE FROM hook_events
        WHERE id IN (
            SELECT id FROM hook_events
            WHERE processed = TRUE
              AND created_at < NOW() - INTERVAL '30 days'
            LIMIT batch_size
        )
        RETURNING 1
    )
    SELECT count(*) INTO deleted_rows FROM deleted;

    end_time := clock_timestamp();

    RETURN QUERY VALUES (
        'hook_events'::text,
        deleted_rows,
        EXTRACT(MILLISECONDS FROM (end_time - start_time))::numeric
    );

    -- Event metrics cleanup
    start_time := clock_timestamp();

    WITH deleted AS (
        DELETE FROM event_metrics
        WHERE id IN (
            SELECT id FROM event_metrics
            WHERE created_at < NOW() - INTERVAL '90 days'
            LIMIT batch_size
        )
        RETURNING 1
    )
    SELECT count(*) INTO deleted_rows FROM deleted;

    end_time := clock_timestamp();

    RETURN QUERY VALUES (
        'event_metrics'::text,
        deleted_rows,
        EXTRACT(MILLISECONDS FROM (end_time - start_time))::numeric
    );
END;
$$ LANGUAGE plpgsql;
```

## Performance Monitoring Queries

### 1. Index Usage Analysis

```sql
-- Check index usage and efficiency
SELECT
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan,
    CASE
        WHEN idx_scan = 0 THEN 'UNUSED'
        WHEN idx_tup_read = 0 THEN 'WRITE_ONLY'
        ELSE 'ACTIVE'
    END as index_status,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
```

### 2. Query Performance Analysis

```sql
-- Identify slow queries
SELECT
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    stddev_exec_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE query LIKE '%hook_events%' OR query LIKE '%event_metrics%'
ORDER BY mean_exec_time DESC
LIMIT 10;
```

### 3. Table Statistics

```sql
-- Table size and activity analysis
SELECT
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    CASE
        WHEN n_live_tup > 0
        THEN round((n_dead_tup::numeric / n_live_tup::numeric) * 100, 2)
        ELSE 0
    END as dead_tuple_percent,
    pg_size_pretty(pg_total_relation_size(relid)) as total_size,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(relid) DESC;
```

## Implementation Recommendations

### 1. Immediate Actions (High Impact, Low Effort)
- Add the recommended composite indexes using `CREATE INDEX CONCURRENTLY`
- Implement the performance monitoring queries
- Set up automated index usage analysis

### 2. Medium Term (Medium Impact, Medium Effort)
- Implement table partitioning for time-series data
- Create materialized views for common analytics queries
- Optimize batch operations

### 3. Long Term (High Impact, High Effort)
- Implement comprehensive query optimization framework
- Set up automated performance tuning
- Consider read replicas for analytics workloads

### 4. Monitoring and Maintenance
- Regular VACUUM and ANALYZE operations
- Monitor index usage and remove unused indexes
- Track query performance trends
- Automated cleanup of old data

This optimization guide provides a comprehensive approach to improving database performance while maintaining data integrity and system reliability.
