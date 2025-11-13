-- Migration: 010_create_llm_metrics_tables
-- Description: Create LLM generation metrics tables for cost tracking, performance monitoring, and learning
-- Dependencies: None (standalone metrics tables)
-- Created: 2025-10-31

-- LLM generation metrics (primary metrics table)
CREATE TABLE IF NOT EXISTS llm_generation_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID NOT NULL,
    session_id UUID,  -- Links to codegen session

    -- Context
    node_type VARCHAR(50) NOT NULL,  -- effect, compute, reducer, orchestrator
    service_name VARCHAR(255) NOT NULL,
    method_name VARCHAR(255) NOT NULL,
    operation_type VARCHAR(100),  -- method_implementation, requirement_extraction, etc.

    -- LLM details
    tier VARCHAR(50) NOT NULL,  -- local, cloud_fast, cloud_premium
    model_used VARCHAR(100) NOT NULL,

    -- Token usage
    tokens_input INTEGER NOT NULL CHECK (tokens_input >= 0),
    tokens_output INTEGER NOT NULL CHECK (tokens_output >= 0),
    tokens_total INTEGER NOT NULL CHECK (tokens_total >= 0),

    -- Performance
    latency_ms FLOAT NOT NULL CHECK (latency_ms >= 0),
    cost_usd NUMERIC(10, 6) NOT NULL CHECK (cost_usd >= 0),

    -- Quality
    syntax_valid BOOLEAN NOT NULL DEFAULT FALSE,
    onex_compliant BOOLEAN NOT NULL DEFAULT FALSE,
    has_type_hints BOOLEAN NOT NULL DEFAULT FALSE,
    security_issues JSONB DEFAULT '[]'::jsonb,

    -- Metadata
    finish_reason VARCHAR(50),  -- stop, length, error
    truncated BOOLEAN DEFAULT FALSE,
    prompt_length INTEGER,
    response_length INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance indexes for llm_generation_metrics
CREATE INDEX IF NOT EXISTS idx_llm_metrics_correlation
    ON llm_generation_metrics(correlation_id);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_session
    ON llm_generation_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_node_type
    ON llm_generation_metrics(node_type);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_model
    ON llm_generation_metrics(model_used);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_created
    ON llm_generation_metrics(created_at DESC);

-- Compound index for common queries (cost analysis by model and tier)
CREATE INDEX IF NOT EXISTS idx_llm_metrics_tier_model
    ON llm_generation_metrics(tier, model_used);

-- Comments for documentation
COMMENT ON TABLE llm_generation_metrics IS 'Tracks LLM generation metrics for cost analysis, performance monitoring, and quality scoring';
COMMENT ON COLUMN llm_generation_metrics.correlation_id IS 'Correlation ID for tracking related operations';
COMMENT ON COLUMN llm_generation_metrics.session_id IS 'Links to codegen session for multi-step tracking';
COMMENT ON COLUMN llm_generation_metrics.node_type IS 'Type of ONEX node (effect, compute, reducer, orchestrator)';
COMMENT ON COLUMN llm_generation_metrics.tier IS 'Model tier (local, cloud_fast, cloud_premium)';
COMMENT ON COLUMN llm_generation_metrics.model_used IS 'Specific model used (e.g., glm-4.5, ollama/codellama:7b)';
COMMENT ON COLUMN llm_generation_metrics.cost_usd IS 'Cost in USD for this generation';
COMMENT ON COLUMN llm_generation_metrics.security_issues IS 'Array of detected security issues as JSON';

-- LLM context windows (model capabilities)
CREATE TABLE IF NOT EXISTS llm_context_windows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tier VARCHAR(50) NOT NULL UNIQUE,
    model_name VARCHAR(100) NOT NULL,
    max_tokens INTEGER NOT NULL CHECK (max_tokens > 0),
    cost_per_1m_input_usd NUMERIC(10, 4),
    cost_per_1m_output_usd NUMERIC(10, 4),
    supports_streaming BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for active model lookups
CREATE INDEX IF NOT EXISTS idx_llm_context_windows_active
    ON llm_context_windows(is_active);

-- Comments for documentation
COMMENT ON TABLE llm_context_windows IS 'Model capabilities and pricing for LLM tiers';
COMMENT ON COLUMN llm_context_windows.tier IS 'Model tier (local, cloud_fast, cloud_premium) - unique identifier';
COMMENT ON COLUMN llm_context_windows.max_tokens IS 'Maximum context window size in tokens';
COMMENT ON COLUMN llm_context_windows.cost_per_1m_input_usd IS 'Cost per 1 million input tokens in USD';
COMMENT ON COLUMN llm_context_windows.cost_per_1m_output_usd IS 'Cost per 1 million output tokens in USD';

-- Insert default context windows
INSERT INTO llm_context_windows (tier, model_name, max_tokens, cost_per_1m_input_usd, cost_per_1m_output_usd, supports_streaming)
VALUES
    ('cloud_fast', 'glm-4.5', 128000, 0.20, 0.20, TRUE),
    ('local', 'ollama/codellama:7b', 16384, 0.00, 0.00, TRUE),
    ('cloud_premium', 'glm-4.6', 128000, 0.30, 0.30, TRUE)
ON CONFLICT (tier) DO NOTHING;

-- LLM generation history (for learning and feedback)
CREATE TABLE IF NOT EXISTS llm_generation_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID NOT NULL,

    -- Context
    node_type VARCHAR(50) NOT NULL,
    method_name VARCHAR(255) NOT NULL,

    -- Prompt and response
    prompt_text TEXT NOT NULL,
    generated_code TEXT NOT NULL,

    -- Quality scores
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    validation_passed BOOLEAN NOT NULL,

    -- Feedback (for learning)
    user_accepted BOOLEAN,
    user_feedback TEXT,
    manual_edits_required BOOLEAN,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance indexes for llm_generation_history
CREATE INDEX IF NOT EXISTS idx_llm_history_correlation
    ON llm_generation_history(correlation_id);
CREATE INDEX IF NOT EXISTS idx_llm_history_quality
    ON llm_generation_history(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_llm_history_node_type
    ON llm_generation_history(node_type);

-- Compound index for analyzing successful patterns
CREATE INDEX IF NOT EXISTS idx_llm_history_validation_quality
    ON llm_generation_history(validation_passed, quality_score DESC);

-- Comments for documentation
COMMENT ON TABLE llm_generation_history IS 'Stores LLM generation history for learning and pattern extraction';
COMMENT ON COLUMN llm_generation_history.prompt_text IS 'Full prompt text sent to LLM';
COMMENT ON COLUMN llm_generation_history.generated_code IS 'Generated code or business logic';
COMMENT ON COLUMN llm_generation_history.quality_score IS 'Quality score (0.0-1.0) from QualityValidator';
COMMENT ON COLUMN llm_generation_history.user_accepted IS 'Whether user accepted the generated code';
COMMENT ON COLUMN llm_generation_history.manual_edits_required IS 'Whether manual edits were needed';

-- LLM patterns table (for future learning and optimization)
CREATE TABLE IF NOT EXISTS llm_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_type VARCHAR(50) NOT NULL,  -- e.g., high_quality_prompt, common_error, successful_implementation
    node_type VARCHAR(50),
    pattern_data JSONB NOT NULL,
    usage_count INTEGER DEFAULT 0,
    avg_quality_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE
);

-- Performance indexes for llm_patterns
CREATE INDEX IF NOT EXISTS idx_llm_patterns_type
    ON llm_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_llm_patterns_node_type
    ON llm_patterns(node_type);
CREATE INDEX IF NOT EXISTS idx_llm_patterns_quality
    ON llm_patterns(avg_quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_llm_patterns_usage
    ON llm_patterns(usage_count DESC);

-- Comments for documentation
COMMENT ON TABLE llm_patterns IS 'Stores learned patterns from LLM generations for future optimization';
COMMENT ON COLUMN llm_patterns.pattern_type IS 'Type of pattern (high_quality_prompt, common_error, successful_implementation)';
COMMENT ON COLUMN llm_patterns.pattern_data IS 'Pattern details as JSON (prompt templates, code patterns, error patterns)';
COMMENT ON COLUMN llm_patterns.usage_count IS 'Number of times this pattern has been used/referenced';
COMMENT ON COLUMN llm_patterns.avg_quality_score IS 'Average quality score when this pattern is used';
