-- OmniNode Bridge - Database Schema
-- O.N.E. v0.1 Compliant PostgreSQL Schema
--
-- This schema defines the core tables for metadata stamping with:
-- - O.N.E. v0.1 protocol compliance
-- - Namespace support for multi-tenancy
-- - Performance-optimized indexes
-- - Partitioned metrics tables
-- - Intelligence data storage

-- ============================================================================
-- EXTENSIONS
-- ============================================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable query performance statistics
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ============================================================================
-- METADATA STAMPS TABLE
-- ============================================================================

-- Metadata stamps with O.N.E. v0.1 compliance fields
CREATE TABLE IF NOT EXISTS metadata_stamps (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_hash VARCHAR(64) NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL CHECK (file_size >= 0),
    content_type VARCHAR(255),

    -- Stamp data (JSONB for flexibility)
    stamp_data JSONB NOT NULL,
    protocol_version VARCHAR(10) NOT NULL DEFAULT '1.0',

    -- O.N.E. v0.1 Compliance fields
    intelligence_data JSONB DEFAULT '{}',
    version INTEGER DEFAULT 1,
    op_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    namespace VARCHAR(255) NOT NULL DEFAULT 'omninode.services.metadata',
    metadata_version VARCHAR(10) NOT NULL DEFAULT '0.1',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- PROTOCOL HANDLERS TABLE
-- ============================================================================

-- Protocol handlers with enhanced configuration
CREATE TABLE IF NOT EXISTS protocol_handlers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    handler_type VARCHAR(100) NOT NULL UNIQUE,
    file_extensions TEXT[] NOT NULL,
    handler_config JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- HASH METRICS TABLE (PARTITIONED)
-- ============================================================================

-- Partitioned hash metrics for performance monitoring
-- This table is partitioned by timestamp to optimize time-series queries
CREATE TABLE IF NOT EXISTS hash_metrics (
    id UUID DEFAULT uuid_generate_v4(),
    operation_type VARCHAR(50) NOT NULL,
    execution_time_ms INTEGER NOT NULL CHECK (execution_time_ms >= 0),
    file_size_bytes BIGINT CHECK (file_size_bytes >= 0),
    cpu_usage_percent DECIMAL(5,2) CHECK (cpu_usage_percent >= 0 AND cpu_usage_percent <= 100),
    memory_usage_mb INTEGER CHECK (memory_usage_mb >= 0),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Create initial partition (current month)
-- Note: Additional partitions should be created as needed
-- Example: CREATE TABLE hash_metrics_2025_10 PARTITION OF hash_metrics
--          FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

-- ============================================================================
-- PERFORMANCE-OPTIMIZED INDEXES
-- ============================================================================

-- Primary lookup index (file_hash)
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_file_hash
    ON metadata_stamps(file_hash);

-- Namespace filtering index
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_namespace
    ON metadata_stamps(namespace);

-- Operation ID lookup index
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_op_id
    ON metadata_stamps(op_id);

-- GIN index for JSONB intelligence_data (fast JSON queries)
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_intelligence_data_gin
    ON metadata_stamps USING GIN(intelligence_data);

-- Temporal index for time-based queries
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_created_at
    ON metadata_stamps(created_at DESC);

-- Hash metrics temporal index
CREATE INDEX IF NOT EXISTS idx_hash_metrics_timestamp
    ON hash_metrics(timestamp DESC);

-- Protocol handler type lookup index
CREATE INDEX IF NOT EXISTS idx_protocol_handlers_type
    ON protocol_handlers(handler_type);

-- ============================================================================
-- UPDATE TRIGGER
-- ============================================================================

-- Automatically update updated_at timestamp on row modification
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to metadata_stamps table
CREATE TRIGGER update_metadata_stamps_updated_at
    BEFORE UPDATE ON metadata_stamps
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to protocol_handlers table
CREATE TRIGGER update_protocol_handlers_updated_at
    BEFORE UPDATE ON protocol_handlers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- EVENT LOGS TABLE (Autonomous Code Generation Event Tracing)
-- ============================================================================

-- Event logs for autonomous code generation event tracing and debugging
-- Supports 13 Kafka topics with 9 event types from PR #25
CREATE TABLE IF NOT EXISTS event_logs (
    -- Primary identification
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Session and correlation tracking
    session_id UUID NOT NULL,
    correlation_id UUID,  -- Nullable for status events

    -- Event classification
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN ('request', 'response', 'status', 'error')),
    topic VARCHAR(255) NOT NULL,

    -- Temporal tracking
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,

    -- Event status and performance
    status VARCHAR(50) NOT NULL CHECK (status IN ('sent', 'received', 'failed', 'processing', 'completed')),
    processing_time_ms INTEGER CHECK (processing_time_ms IS NULL OR processing_time_ms >= 0),

    -- Event data (JSONB for flexibility)
    payload JSONB NOT NULL,
    metadata JSONB DEFAULT '{}' NOT NULL
);

-- Event logs indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_event_logs_session_id
    ON event_logs(session_id);

CREATE INDEX IF NOT EXISTS idx_event_logs_correlation_id
    ON event_logs(correlation_id);

CREATE INDEX IF NOT EXISTS idx_event_logs_timestamp
    ON event_logs(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_event_logs_event_type
    ON event_logs(event_type);

CREATE INDEX IF NOT EXISTS idx_event_logs_topic
    ON event_logs(topic);

-- Composite index for optimal session + time queries
CREATE INDEX IF NOT EXISTS idx_event_logs_session_timestamp
    ON event_logs(session_id, timestamp DESC);

-- GIN indexes for JSONB payload and metadata queries
CREATE INDEX IF NOT EXISTS idx_event_logs_payload_gin
    ON event_logs USING GIN(payload);

CREATE INDEX IF NOT EXISTS idx_event_logs_metadata_gin
    ON event_logs USING GIN(metadata);

-- Event Logs Table Comments:
-- Purpose: Store events from autonomous code generation infrastructure
-- Query Patterns:
--   1. trace_session_events(): Query by session_id with time range
--   2. get_session_metrics(): Aggregate by event_type, topic, calculate percentiles
--   3. find_correlated_events(): Query by correlation_id for request/response matching
-- Performance Targets:
--   - trace_session_events(): < 50ms for 1000 events
--   - get_session_metrics(): < 100ms for metrics calculation
--   - find_correlated_events(): < 20ms for correlation lookup
-- Related Files:
--   - src/omninode_bridge/events/codegen_schemas.py (9 event schemas)
--   - src/omninode_bridge/dashboard/codegen_event_tracer.py (tracer implementation)

-- ============================================================================
-- HELPER VIEWS
-- ============================================================================

-- View for recent stamp activity (last 24 hours)
CREATE OR REPLACE VIEW recent_stamp_activity AS
SELECT
    namespace,
    COUNT(*) as stamp_count,
    AVG((stamp_data->>'execution_time_ms')::numeric) as avg_execution_time_ms,
    MAX(created_at) as last_stamp_time
FROM metadata_stamps
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY namespace
ORDER BY stamp_count DESC;

-- View for namespace statistics
CREATE OR REPLACE VIEW namespace_statistics AS
SELECT
    namespace,
    COUNT(*) as total_stamps,
    COUNT(DISTINCT file_hash) as unique_hashes,
    SUM(file_size) as total_size_bytes,
    AVG(file_size) as avg_size_bytes,
    MIN(created_at) as first_stamp,
    MAX(created_at) as last_stamp
FROM metadata_stamps
GROUP BY namespace;

-- View for event logs session summary (last 24 hours)
CREATE OR REPLACE VIEW event_logs_session_summary AS
SELECT
    session_id,
    COUNT(*) as total_events,
    COUNT(DISTINCT correlation_id) as unique_correlations,
    MIN(timestamp) as session_start,
    MAX(timestamp) as session_end,
    EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) * 1000 as session_duration_ms,
    COUNT(CASE WHEN status IN ('sent', 'received', 'completed') THEN 1 END) as successful_events,
    COUNT(CASE WHEN status = 'failed' OR event_type = 'error' THEN 1 END) as failed_events,
    AVG(processing_time_ms) FILTER (WHERE processing_time_ms IS NOT NULL) as avg_processing_time_ms
FROM event_logs
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY session_id
ORDER BY session_start DESC;

-- View for event logs topic performance (last 24 hours)
CREATE OR REPLACE VIEW event_logs_topic_performance AS
SELECT
    topic,
    COUNT(*) as event_count,
    AVG(processing_time_ms) FILTER (WHERE processing_time_ms IS NOT NULL) as avg_processing_time_ms,
    MAX(processing_time_ms) as max_processing_time_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_time_ms)
        FILTER (WHERE processing_time_ms IS NOT NULL) as p95_processing_time_ms,
    COUNT(CASE WHEN status = 'failed' OR event_type = 'error' THEN 1 END) as failed_count,
    COUNT(CASE WHEN status IN ('sent', 'received', 'completed') THEN 1 END) as success_count
FROM event_logs
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY topic
ORDER BY avg_processing_time_ms DESC NULLS LAST;

-- ============================================================================
-- SAMPLE DATA (OPTIONAL - FOR DEVELOPMENT ONLY)
-- ============================================================================

-- Insert sample protocol handlers
-- Uncomment for development environments
/*
INSERT INTO protocol_handlers (handler_type, file_extensions, handler_config) VALUES
    ('image_handler', ARRAY['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'],
     '{"max_size_mb": 10, "extract_exif": true}'::jsonb),
    ('document_handler', ARRAY['.pdf', '.doc', '.docx', '.txt', '.md'],
     '{"max_size_mb": 50, "extract_text": true}'::jsonb),
    ('audio_handler', ARRAY['.mp3', '.wav', '.flac', '.aac', '.ogg'],
     '{"max_size_mb": 100, "extract_metadata": true}'::jsonb),
    ('video_handler', ARRAY['.mp4', '.avi', '.mkv', '.mov', '.webm'],
     '{"max_size_mb": 500, "extract_metadata": true}'::jsonb),
    ('archive_handler', ARRAY['.zip', '.tar', '.gz', '.rar', '.7z'],
     '{"max_size_mb": 1000, "scan_contents": false}'::jsonb)
ON CONFLICT (handler_type) DO NOTHING;
*/

-- ============================================================================
-- MAINTENANCE FUNCTIONS
-- ============================================================================

-- Function to create monthly hash_metrics partitions
CREATE OR REPLACE FUNCTION create_hash_metrics_partition(partition_date DATE)
RETURNS void AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    partition_name := 'hash_metrics_' || TO_CHAR(partition_date, 'YYYY_MM');
    start_date := DATE_TRUNC('month', partition_date);
    end_date := start_date + INTERVAL '1 month';

    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF hash_metrics FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date
    );
END;
$$ LANGUAGE plpgsql;

-- Function to clean old metrics (retention: 90 days)
CREATE OR REPLACE FUNCTION cleanup_old_metrics()
RETURNS void AS $$
BEGIN
    DELETE FROM hash_metrics
    WHERE timestamp < NOW() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERFORMANCE TUNING COMMENTS
-- ============================================================================

-- For production deployments, consider:
-- 1. Adjust shared_buffers to 25% of RAM
-- 2. Set effective_cache_size to 50% of RAM
-- 3. Increase work_mem for complex queries (16MB - 64MB)
-- 4. Enable parallel query execution for large tables
-- 5. Configure connection pooling (20-50 connections for typical workload)
-- 6. Monitor pg_stat_statements for slow queries
-- 7. Regularly run VACUUM ANALYZE on high-churn tables
-- 8. Consider table partitioning for tables > 100GB
-- 9. Use connection poolers (PgBouncer, PgPool-II) for high concurrency
-- 10. Implement read replicas for read-heavy workloads

-- ============================================================================
-- SECURITY CONSIDERATIONS
-- ============================================================================

-- For production deployments, ensure:
-- 1. Row-level security (RLS) policies for namespace isolation
-- 2. SSL/TLS for all database connections
-- 3. Strong password policies
-- 4. Regular security audits
-- 5. Principle of least privilege for application users
-- 6. Audit logging for sensitive operations
-- 7. Encryption at rest for sensitive data
-- 8. Network segmentation and firewall rules
-- 9. Regular PostgreSQL version updates
-- 10. Backup encryption and secure storage

-- ============================================================================
-- BACKUP AND RECOVERY
-- ============================================================================

-- Recommended backup strategy:
-- 1. Daily full backups with pg_dump or pg_basebackup
-- 2. WAL archiving for point-in-time recovery
-- 3. Test restore procedures regularly
-- 4. Store backups in multiple geographic locations
-- 5. Retention: 7 daily, 4 weekly, 12 monthly backups
-- 6. Document recovery procedures (RTO: 1 hour, RPO: 5 minutes)

-- Example backup command:
-- pg_dump -h localhost -U postgres -d metadata_stamping_prod -F c -f backup_$(date +%Y%m%d).dump

-- Example restore command:
-- pg_restore -h localhost -U postgres -d metadata_stamping_prod -c backup_20251014.dump
