-- MetadataStampingService Phase 1 Database Schema
-- High-performance schema with optimized indexing for sub-5ms operations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Metadata stamps table with optimized indexing and compliance fields
CREATE TABLE IF NOT EXISTS metadata_stamps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_hash VARCHAR(64) NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL CHECK (file_size >= 0),
    content_type VARCHAR(255),
    stamp_data JSONB NOT NULL,
    protocol_version VARCHAR(10) NOT NULL DEFAULT '1.0',
    -- Compliance fields from omnibase_3 and ai-dev patterns
    intelligence_data JSONB DEFAULT '{}',
    version INTEGER DEFAULT 1,
    op_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    namespace VARCHAR(255) NOT NULL DEFAULT 'omninode.services.metadata',
    metadata_version VARCHAR(10) NOT NULL DEFAULT '0.1',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

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

-- Hash generation metrics with partitioning support (base table)
CREATE TABLE IF NOT EXISTS hash_metrics (
    id UUID DEFAULT uuid_generate_v4(),
    operation_type VARCHAR(50) NOT NULL,
    execution_time_ms INTEGER NOT NULL CHECK (execution_time_ms >= 0),
    file_size_bytes BIGINT CHECK (file_size_bytes >= 0),
    cpu_usage_percent DECIMAL(5,2) CHECK (cpu_usage_percent >= 0 AND cpu_usage_percent <= 100),
    memory_usage_mb INTEGER CHECK (memory_usage_mb >= 0),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Create monthly partition for current month (example)
CREATE TABLE IF NOT EXISTS hash_metrics_y2025m09 PARTITION OF hash_metrics
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

-- Create monthly partition for next month
CREATE TABLE IF NOT EXISTS hash_metrics_y2025m10 PARTITION OF hash_metrics
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

-- Performance-optimized indexes
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_file_hash ON metadata_stamps(file_hash);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_file_path ON metadata_stamps USING GIN(to_tsvector('english', file_path));
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_created_at ON metadata_stamps(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_file_size ON metadata_stamps(file_size) WHERE file_size > 1024*1024; -- Large files only
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_content_type ON metadata_stamps(content_type) WHERE content_type IS NOT NULL;

-- JSONB indexes for stamp_data queries
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_stamp_data_gin ON metadata_stamps USING GIN(stamp_data);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_protocol_version ON metadata_stamps(protocol_version);

-- Compliance field indexes
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_namespace ON metadata_stamps(namespace);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_op_id ON metadata_stamps(op_id);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_version ON metadata_stamps(version);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_metadata_version ON metadata_stamps(metadata_version);
CREATE INDEX IF NOT EXISTS idx_metadata_stamps_intelligence_data_gin ON metadata_stamps USING GIN(intelligence_data);

-- Protocol handlers indexes
CREATE INDEX IF NOT EXISTS idx_protocol_handlers_type ON protocol_handlers(handler_type);
CREATE INDEX IF NOT EXISTS idx_protocol_handlers_active ON protocol_handlers(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_protocol_handlers_extensions ON protocol_handlers USING GIN(file_extensions);

-- Hash metrics indexes with partition awareness
CREATE INDEX IF NOT EXISTS idx_hash_metrics_timestamp ON hash_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_hash_metrics_operation_type ON hash_metrics(operation_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_hash_metrics_execution_time ON hash_metrics(execution_time_ms) WHERE execution_time_ms > 2; -- Performance violations

-- Triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_metadata_stamps_updated_at BEFORE UPDATE
    ON metadata_stamps FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_protocol_handlers_updated_at BEFORE UPDATE
    ON protocol_handlers FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Database performance tuning settings (run as superuser)
-- ALTER SYSTEM SET shared_buffers = '256MB';
-- ALTER SYSTEM SET effective_cache_size = '1GB';
-- ALTER SYSTEM SET work_mem = '16MB';
-- ALTER SYSTEM SET maintenance_work_mem = '256MB';
-- ALTER SYSTEM SET checkpoint_completion_target = 0.9;
-- ALTER SYSTEM SET wal_buffers = '16MB';
-- ALTER SYSTEM SET default_statistics_target = 100;
