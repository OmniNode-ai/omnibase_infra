-- Migration: 013_create_llm_metrics_tables
-- Description: Create tables for LLM generation metrics, history, and learned patterns
-- Dependencies: None (standalone tables)
-- Created: 2025-10-31
-- Purpose: Support debug intelligence storage for LLM-based code generation

-- LLM Generation Metrics Table
-- Tracks token usage, costs, performance, and success/failure for each generation
CREATE TABLE IF NOT EXISTS llm_generation_metrics (
    metric_id UUID PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    correlation_id UUID,
    node_type VARCHAR(50) NOT NULL,
    model_tier VARCHAR(20) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    prompt_tokens INTEGER NOT NULL CHECK (prompt_tokens >= 0),
    completion_tokens INTEGER NOT NULL CHECK (completion_tokens >= 0),
    total_tokens INTEGER NOT NULL CHECK (total_tokens >= 0),
    latency_ms DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),
    cost_usd DOUBLE PRECISION NOT NULL CHECK (cost_usd >= 0),
    success BOOLEAN NOT NULL,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Performance indexes for metrics
CREATE INDEX IF NOT EXISTS idx_llm_metrics_session_id
    ON llm_generation_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_correlation_id
    ON llm_generation_metrics(correlation_id);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_node_type
    ON llm_generation_metrics(node_type);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_model_name
    ON llm_generation_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_created_at
    ON llm_generation_metrics(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_success
    ON llm_generation_metrics(success);

-- Compound indexes for common queries
CREATE INDEX IF NOT EXISTS idx_llm_metrics_model_created
    ON llm_generation_metrics(model_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_node_created
    ON llm_generation_metrics(node_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_session_created
    ON llm_generation_metrics(session_id, created_at DESC);

-- JSONB index for metadata queries
CREATE INDEX IF NOT EXISTS idx_llm_metrics_metadata
    ON llm_generation_metrics USING GIN (metadata);

-- Comments for documentation
COMMENT ON TABLE llm_generation_metrics IS 'LLM generation metrics for token usage, cost, and performance tracking';
COMMENT ON COLUMN llm_generation_metrics.session_id IS 'Session identifier for grouping related metrics';
COMMENT ON COLUMN llm_generation_metrics.correlation_id IS 'ONEX correlation ID for distributed tracing';
COMMENT ON COLUMN llm_generation_metrics.node_type IS 'Type of node being generated (effect, compute, reducer, orchestrator)';
COMMENT ON COLUMN llm_generation_metrics.model_tier IS 'Model tier classification (tier_1, tier_2, tier_3)';
COMMENT ON COLUMN llm_generation_metrics.model_name IS 'Specific model used for generation';
COMMENT ON COLUMN llm_generation_metrics.total_tokens IS 'Total tokens used (prompt + completion)';
COMMENT ON COLUMN llm_generation_metrics.latency_ms IS 'Generation latency in milliseconds';
COMMENT ON COLUMN llm_generation_metrics.cost_usd IS 'Generation cost in USD';

-- LLM Generation History Table
-- Stores actual prompts and outputs for debugging and pattern learning
CREATE TABLE IF NOT EXISTS llm_generation_history (
    history_id UUID PRIMARY KEY,
    metric_id UUID NOT NULL REFERENCES llm_generation_metrics(metric_id) ON DELETE CASCADE,
    prompt_text TEXT NOT NULL,
    generated_text TEXT NOT NULL,
    quality_score DOUBLE PRECISION CHECK (quality_score >= 0 AND quality_score <= 1),
    validation_passed BOOLEAN NOT NULL,
    validation_errors JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Performance indexes for history
CREATE INDEX IF NOT EXISTS idx_llm_history_metric_id
    ON llm_generation_history(metric_id);
CREATE INDEX IF NOT EXISTS idx_llm_history_quality_score
    ON llm_generation_history(quality_score DESC) WHERE quality_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_llm_history_validation_passed
    ON llm_generation_history(validation_passed);
CREATE INDEX IF NOT EXISTS idx_llm_history_created_at
    ON llm_generation_history(created_at DESC);

-- JSONB index for validation errors
CREATE INDEX IF NOT EXISTS idx_llm_history_validation_errors
    ON llm_generation_history USING GIN (validation_errors);

-- Comments for documentation
COMMENT ON TABLE llm_generation_history IS 'Historical record of LLM generations with prompts and outputs';
COMMENT ON COLUMN llm_generation_history.metric_id IS 'Foreign key to associated generation metric';
COMMENT ON COLUMN llm_generation_history.prompt_text IS 'Full prompt text sent to LLM';
COMMENT ON COLUMN llm_generation_history.generated_text IS 'Full generated text from LLM';
COMMENT ON COLUMN llm_generation_history.quality_score IS 'Quality assessment score (0-1)';
COMMENT ON COLUMN llm_generation_history.validation_passed IS 'Whether generated code passed validation';
COMMENT ON COLUMN llm_generation_history.validation_errors IS 'Validation error details as JSON';

-- LLM Patterns Table
-- Stores learned patterns from successful generations
CREATE TABLE IF NOT EXISTS llm_patterns (
    pattern_id UUID PRIMARY KEY,
    pattern_type VARCHAR(100) NOT NULL,
    node_type VARCHAR(50),
    pattern_data JSONB NOT NULL,
    usage_count INTEGER NOT NULL DEFAULT 0 CHECK (usage_count >= 0),
    avg_quality_score DOUBLE PRECISION CHECK (avg_quality_score >= 0 AND avg_quality_score <= 1),
    success_rate DOUBLE PRECISION CHECK (success_rate >= 0 AND success_rate <= 1),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Performance indexes for patterns
CREATE INDEX IF NOT EXISTS idx_llm_patterns_pattern_type
    ON llm_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_llm_patterns_node_type
    ON llm_patterns(node_type);
CREATE INDEX IF NOT EXISTS idx_llm_patterns_quality_score
    ON llm_patterns(avg_quality_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_llm_patterns_usage_count
    ON llm_patterns(usage_count DESC);
CREATE INDEX IF NOT EXISTS idx_llm_patterns_updated_at
    ON llm_patterns(updated_at DESC);

-- Compound indexes for pattern queries
CREATE INDEX IF NOT EXISTS idx_llm_patterns_type_quality
    ON llm_patterns(pattern_type, avg_quality_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_llm_patterns_node_quality
    ON llm_patterns(node_type, avg_quality_score DESC NULLS LAST);

-- JSONB indexes for pattern data and metadata
CREATE INDEX IF NOT EXISTS idx_llm_patterns_pattern_data
    ON llm_patterns USING GIN (pattern_data);
CREATE INDEX IF NOT EXISTS idx_llm_patterns_metadata
    ON llm_patterns USING GIN (metadata);

-- Comments for documentation
COMMENT ON TABLE llm_patterns IS 'Learned patterns from successful LLM generations';
COMMENT ON COLUMN llm_patterns.pattern_type IS 'Type of pattern (prompt_template, code_structure, etc.)';
COMMENT ON COLUMN llm_patterns.node_type IS 'Node type this pattern applies to (optional)';
COMMENT ON COLUMN llm_patterns.pattern_data IS 'Pattern data as JSON (template, structure, etc.)';
COMMENT ON COLUMN llm_patterns.usage_count IS 'Number of times pattern has been used';
COMMENT ON COLUMN llm_patterns.avg_quality_score IS 'Average quality score across uses';
COMMENT ON COLUMN llm_patterns.success_rate IS 'Success rate (successful uses / total uses)';

-- Create trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_llm_patterns_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_llm_patterns_updated_at
    BEFORE UPDATE ON llm_patterns
    FOR EACH ROW
    EXECUTE FUNCTION update_llm_patterns_updated_at();

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Migration 013_create_llm_metrics_tables completed successfully';
    RAISE NOTICE 'Created tables: llm_generation_metrics, llm_generation_history, llm_patterns';
    RAISE NOTICE 'Created indexes for performance optimization';
    RAISE NOTICE 'Created trigger for automatic updated_at timestamp';
END $$;
