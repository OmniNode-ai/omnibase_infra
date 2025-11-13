"""
SQL Query Constants for LLM Metrics Storage.

All queries use asyncpg parameterized format ($1, $2, etc.) for safety
and performance.
"""

# === LLM Generation Metrics Queries ===

INSERT_GENERATION_METRIC = """
    INSERT INTO llm_generation_metrics (
        metric_id,
        session_id,
        correlation_id,
        node_type,
        model_tier,
        model_name,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        latency_ms,
        cost_usd,
        success,
        error_message,
        metadata,
        created_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
    RETURNING metric_id, created_at
"""

SELECT_METRIC_BY_ID = """
    SELECT
        metric_id,
        session_id,
        correlation_id,
        node_type,
        model_tier,
        model_name,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        latency_ms,
        cost_usd,
        success,
        error_message,
        metadata,
        created_at
    FROM llm_generation_metrics
    WHERE metric_id = $1
"""

SELECT_METRICS_BY_SESSION = """
    SELECT
        metric_id,
        session_id,
        correlation_id,
        node_type,
        model_tier,
        model_name,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        latency_ms,
        cost_usd,
        success,
        error_message,
        metadata,
        created_at
    FROM llm_generation_metrics
    WHERE session_id = $1
    ORDER BY created_at DESC
"""

SELECT_AVERAGE_METRICS = """
    SELECT
        COUNT(*) as total_generations,
        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_generations,
        SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_generations,
        AVG(latency_ms) as avg_latency_ms,
        AVG(prompt_tokens) as avg_prompt_tokens,
        AVG(completion_tokens) as avg_completion_tokens,
        AVG(total_tokens) as avg_total_tokens,
        AVG(cost_usd) as avg_cost_usd,
        SUM(total_tokens) as total_tokens,
        SUM(cost_usd) as total_cost_usd
    FROM llm_generation_metrics
    WHERE model_name = $1
      AND created_at >= NOW() - INTERVAL '1 day' * $2
"""

SELECT_METRICS_BY_NODE_TYPE = """
    SELECT
        metric_id,
        session_id,
        correlation_id,
        node_type,
        model_tier,
        model_name,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        latency_ms,
        cost_usd,
        success,
        error_message,
        metadata,
        created_at
    FROM llm_generation_metrics
    WHERE node_type = $1
      AND created_at >= NOW() - INTERVAL '1 day' * $2
    ORDER BY created_at DESC
    LIMIT $3
"""

# === LLM Generation History Queries ===

INSERT_GENERATION_HISTORY = """
    INSERT INTO llm_generation_history (
        history_id,
        metric_id,
        prompt_text,
        generated_text,
        quality_score,
        validation_passed,
        validation_errors,
        created_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    RETURNING history_id, created_at
"""

SELECT_HISTORY_BY_METRIC = """
    SELECT
        history_id,
        metric_id,
        prompt_text,
        generated_text,
        quality_score,
        validation_passed,
        validation_errors,
        created_at
    FROM llm_generation_history
    WHERE metric_id = $1
"""

SELECT_HISTORY_BY_ID = """
    SELECT
        history_id,
        metric_id,
        prompt_text,
        generated_text,
        quality_score,
        validation_passed,
        validation_errors,
        created_at
    FROM llm_generation_history
    WHERE history_id = $1
"""

SELECT_HIGH_QUALITY_HISTORY = """
    SELECT
        history_id,
        metric_id,
        prompt_text,
        generated_text,
        quality_score,
        validation_passed,
        validation_errors,
        created_at
    FROM llm_generation_history
    WHERE quality_score >= $1
      AND validation_passed = true
    ORDER BY quality_score DESC
    LIMIT $2
"""

# === LLM Pattern Queries ===

INSERT_PATTERN = """
    INSERT INTO llm_patterns (
        pattern_id,
        pattern_type,
        node_type,
        pattern_data,
        usage_count,
        avg_quality_score,
        success_rate,
        metadata,
        created_at,
        updated_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    ON CONFLICT (pattern_id)
    DO UPDATE SET
        usage_count = llm_patterns.usage_count + 1,
        avg_quality_score = $6,
        success_rate = $7,
        metadata = $8,
        updated_at = $10
    RETURNING pattern_id, created_at, updated_at
"""

SELECT_PATTERN_BY_ID = """
    SELECT
        pattern_id,
        pattern_type,
        node_type,
        pattern_data,
        usage_count,
        avg_quality_score,
        success_rate,
        metadata,
        created_at,
        updated_at
    FROM llm_patterns
    WHERE pattern_id = $1
"""

SELECT_BEST_PATTERNS = """
    SELECT
        pattern_id,
        pattern_type,
        node_type,
        pattern_data,
        usage_count,
        avg_quality_score,
        success_rate,
        metadata,
        created_at,
        updated_at
    FROM llm_patterns
    WHERE pattern_type = $1
      AND avg_quality_score IS NOT NULL
    ORDER BY avg_quality_score DESC, usage_count DESC
    LIMIT $2
"""

SELECT_PATTERNS_BY_NODE_TYPE = """
    SELECT
        pattern_id,
        pattern_type,
        node_type,
        pattern_data,
        usage_count,
        avg_quality_score,
        success_rate,
        metadata,
        created_at,
        updated_at
    FROM llm_patterns
    WHERE node_type = $1
    ORDER BY avg_quality_score DESC NULLS LAST, usage_count DESC
    LIMIT $2
"""

UPDATE_PATTERN_USAGE = """
    UPDATE llm_patterns
    SET usage_count = usage_count + 1,
        updated_at = NOW()
    WHERE pattern_id = $1
    RETURNING pattern_id, usage_count
"""

UPDATE_PATTERN_METRICS = """
    UPDATE llm_patterns
    SET avg_quality_score = $2,
        success_rate = $3,
        updated_at = NOW()
    WHERE pattern_id = $1
    RETURNING pattern_id, avg_quality_score, success_rate
"""

# === Aggregate/Analytics Queries ===

SELECT_SESSION_SUMMARY = """
    SELECT
        session_id,
        COUNT(*) as total_generations,
        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_generations,
        SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_generations,
        SUM(total_tokens) as total_tokens,
        SUM(cost_usd) as total_cost_usd,
        AVG(latency_ms) as avg_latency_ms,
        MIN(created_at) as period_start,
        MAX(created_at) as period_end
    FROM llm_generation_metrics
    WHERE session_id = $1
    GROUP BY session_id
"""

SELECT_MODEL_PERFORMANCE = """
    SELECT
        model_name,
        model_tier,
        COUNT(*) as total_generations,
        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_generations,
        AVG(latency_ms) as avg_latency_ms,
        AVG(cost_usd) as avg_cost_usd,
        SUM(total_tokens) as total_tokens
    FROM llm_generation_metrics
    WHERE created_at >= NOW() - INTERVAL '1 day' * $1
    GROUP BY model_name, model_tier
    ORDER BY total_generations DESC
"""

SELECT_NODE_TYPE_PERFORMANCE = """
    SELECT
        node_type,
        COUNT(*) as total_generations,
        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_generations,
        AVG(latency_ms) as avg_latency_ms,
        AVG(total_tokens) as avg_total_tokens
    FROM llm_generation_metrics
    WHERE created_at >= NOW() - INTERVAL '1 day' * $1
    GROUP BY node_type
    ORDER BY total_generations DESC
"""

SELECT_DAILY_SUMMARY = """
    SELECT
        DATE(created_at) as date,
        COUNT(*) as total_generations,
        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_generations,
        SUM(total_tokens) as total_tokens,
        SUM(cost_usd) as total_cost_usd,
        AVG(latency_ms) as avg_latency_ms
    FROM llm_generation_metrics
    WHERE created_at >= NOW() - INTERVAL '1 day' * $1
    GROUP BY DATE(created_at)
    ORDER BY date DESC
"""

# === Cleanup Queries (for maintenance) ===

DELETE_OLD_METRICS = """
    DELETE FROM llm_generation_metrics
    WHERE created_at < NOW() - INTERVAL '1 day' * $1
"""

DELETE_OLD_HISTORY = """
    DELETE FROM llm_generation_history
    WHERE created_at < NOW() - INTERVAL '1 day' * $1
"""
