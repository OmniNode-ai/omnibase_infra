"""
Unit tests for LLMMetricsStore.

Tests the async storage layer for LLM metrics with mocked database connections.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from omnibase_core import EnumCoreErrorCode, ModelOnexError

from omninode_bridge.intelligence.llm_metrics_store import LLMMetricsStore
from omninode_bridge.intelligence.models import (
    LLMGenerationHistory,
    LLMGenerationMetric,
    LLMPattern,
    MetricsSummary,
)


@pytest.fixture
def mock_db_pool():
    """Create a mock asyncpg pool."""
    pool = AsyncMock()
    pool.get_size = MagicMock(return_value=10)
    pool.get_idle_size = MagicMock(return_value=5)
    return pool


@pytest.fixture
def mock_connection():
    """Create a mock database connection."""
    conn = AsyncMock()
    return conn


@pytest.fixture
def llm_metrics_store(mock_db_pool):
    """Create LLMMetricsStore with mocked pool."""
    return LLMMetricsStore(mock_db_pool)


@pytest.fixture
def sample_metric():
    """Create a sample LLM generation metric."""
    return LLMGenerationMetric(
        session_id="sess_test_123",
        node_type="effect",
        model_tier="tier_2",
        model_name="claude-sonnet-4",
        prompt_tokens=1500,
        completion_tokens=800,
        total_tokens=2300,
        latency_ms=3500.0,
        cost_usd=0.0345,
        success=True,
    )


@pytest.fixture
def sample_history():
    """Create a sample generation history."""
    return LLMGenerationHistory(
        metric_id=uuid4(),
        prompt_text="Generate a Python Effect node...",
        generated_text="class NodeMyEffect:\n    async def execute_effect(self)...",
        quality_score=0.92,
        validation_passed=True,
    )


@pytest.fixture
def sample_pattern():
    """Create a sample learned pattern."""
    return LLMPattern(
        pattern_type="prompt_template",
        node_type="effect",
        pattern_data={
            "template": "Generate a Python Effect node that...",
            "variables": ["node_name", "operation_type"],
        },
        usage_count=15,
        avg_quality_score=0.93,
        success_rate=0.95,
    )


class TestLLMMetricsStoreInit:
    """Test LLMMetricsStore initialization."""

    def test_init_success(self, mock_db_pool):
        """Test successful initialization."""
        store = LLMMetricsStore(mock_db_pool)
        assert store.pool == mock_db_pool
        assert store._metrics_cache["total_operations"] == 0

    def test_init_none_pool(self):
        """Test initialization with None pool raises ValueError."""
        with pytest.raises(ValueError, match="db_pool cannot be None"):
            LLMMetricsStore(None)


class TestStoreGenerationMetric:
    """Test storing generation metrics."""

    @pytest.mark.asyncio
    async def test_store_metric_success(
        self, llm_metrics_store, mock_db_pool, mock_connection, sample_metric
    ):
        """Test successful metric storage."""
        # Setup mock
        mock_db_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_db_pool.release = AsyncMock()
        mock_connection.fetchrow = AsyncMock(
            return_value={
                "metric_id": sample_metric.metric_id,
                "created_at": datetime.utcnow(),
            }
        )

        # Execute
        result = await llm_metrics_store.store_generation_metric(sample_metric)

        # Verify
        assert result == str(sample_metric.metric_id)
        mock_connection.fetchrow.assert_called_once()
        mock_db_pool.release.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_metric_none_input(self, llm_metrics_store):
        """Test storing None metric raises ValueError."""
        with pytest.raises(ValueError, match="metric cannot be None"):
            await llm_metrics_store.store_generation_metric(None)

    @pytest.mark.asyncio
    async def test_store_metric_db_returns_none(
        self, llm_metrics_store, mock_db_pool, mock_connection, sample_metric
    ):
        """Test handling when database returns None."""
        # Setup mock to return None
        mock_db_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_db_pool.release = AsyncMock()
        mock_connection.fetchrow = AsyncMock(return_value=None)

        # Execute and verify
        with pytest.raises(ModelOnexError) as exc_info:
            await llm_metrics_store.store_generation_metric(sample_metric)

        assert exc_info.value.error_code == EnumCoreErrorCode.DATABASE_OPERATION_ERROR
        assert "returned None" in exc_info.value.message


class TestStoreGenerationHistory:
    """Test storing generation history."""

    @pytest.mark.asyncio
    async def test_store_history_success(
        self, llm_metrics_store, mock_db_pool, mock_connection, sample_history
    ):
        """Test successful history storage."""
        # Setup mock
        mock_db_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_db_pool.release = AsyncMock()
        mock_connection.fetchrow = AsyncMock(
            return_value={
                "history_id": sample_history.history_id,
                "created_at": datetime.utcnow(),
            }
        )

        # Execute
        result = await llm_metrics_store.store_generation_history(sample_history)

        # Verify
        assert result == str(sample_history.history_id)
        mock_connection.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_history_none_input(self, llm_metrics_store):
        """Test storing None history raises ValueError."""
        with pytest.raises(ValueError, match="history cannot be None"):
            await llm_metrics_store.store_generation_history(None)


class TestGetMetricsBySession:
    """Test retrieving metrics by session."""

    @pytest.mark.asyncio
    async def test_get_metrics_success(
        self, llm_metrics_store, mock_db_pool, mock_connection, sample_metric
    ):
        """Test successful metrics retrieval."""
        # Setup mock
        mock_db_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_db_pool.release = AsyncMock()
        mock_connection.fetch = AsyncMock(
            return_value=[dict(sample_metric.model_dump())]
        )

        # Execute
        result = await llm_metrics_store.get_metrics_by_session("sess_test_123")

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], LLMGenerationMetric)
        assert result[0].session_id == "sess_test_123"

    @pytest.mark.asyncio
    async def test_get_metrics_empty_session(self, llm_metrics_store):
        """Test getting metrics with empty session_id raises ValueError."""
        with pytest.raises(ValueError, match="session_id cannot be None or empty"):
            await llm_metrics_store.get_metrics_by_session("")


class TestGetAverageMetrics:
    """Test getting average metrics."""

    @pytest.mark.asyncio
    async def test_get_average_metrics_success(
        self, llm_metrics_store, mock_db_pool, mock_connection
    ):
        """Test successful average metrics retrieval."""
        # Setup mock
        mock_db_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_db_pool.release = AsyncMock()
        mock_connection.fetchrow = AsyncMock(
            return_value={
                "total_generations": 150,
                "successful_generations": 142,
                "failed_generations": 8,
                "avg_latency_ms": 3200.0,
                "avg_prompt_tokens": 1500.0,
                "avg_completion_tokens": 800.0,
                "avg_total_tokens": 2300.0,
                "avg_cost_usd": 0.034,
                "total_tokens": 345000,
                "total_cost_usd": 5.1,
            }
        )

        # Execute
        result = await llm_metrics_store.get_average_metrics("claude-sonnet-4", days=7)

        # Verify
        assert result["total_generations"] == 150
        assert result["avg_latency_ms"] == 3200.0

    @pytest.mark.asyncio
    async def test_get_average_metrics_no_data(
        self, llm_metrics_store, mock_db_pool, mock_connection
    ):
        """Test average metrics when no data exists."""
        # Setup mock to return None
        mock_db_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_db_pool.release = AsyncMock()
        mock_connection.fetchrow = AsyncMock(return_value=None)

        # Execute
        result = await llm_metrics_store.get_average_metrics("claude-sonnet-4", days=7)

        # Verify empty metrics returned
        assert result["total_generations"] == 0
        assert result["total_cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_get_average_metrics_invalid_days(self, llm_metrics_store):
        """Test getting average metrics with invalid days raises ValueError."""
        with pytest.raises(ValueError, match="days must be >= 1"):
            await llm_metrics_store.get_average_metrics("claude-sonnet-4", days=0)


class TestStoreLearnedPattern:
    """Test storing learned patterns."""

    @pytest.mark.asyncio
    async def test_store_pattern_success(
        self, llm_metrics_store, mock_db_pool, mock_connection, sample_pattern
    ):
        """Test successful pattern storage."""
        # Setup mock
        mock_db_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_db_pool.release = AsyncMock()
        mock_connection.fetchrow = AsyncMock(
            return_value={
                "pattern_id": sample_pattern.pattern_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        )

        # Execute
        result = await llm_metrics_store.store_learned_pattern(sample_pattern)

        # Verify
        assert result == str(sample_pattern.pattern_id)
        mock_connection.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_pattern_none_input(self, llm_metrics_store):
        """Test storing None pattern raises ValueError."""
        with pytest.raises(ValueError, match="pattern cannot be None"):
            await llm_metrics_store.store_learned_pattern(None)


class TestGetBestPatterns:
    """Test retrieving best patterns."""

    @pytest.mark.asyncio
    async def test_get_best_patterns_success(
        self, llm_metrics_store, mock_db_pool, mock_connection, sample_pattern
    ):
        """Test successful best patterns retrieval."""
        # Setup mock
        mock_db_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_db_pool.release = AsyncMock()
        mock_connection.fetch = AsyncMock(
            return_value=[dict(sample_pattern.model_dump())]
        )

        # Execute
        result = await llm_metrics_store.get_best_patterns("prompt_template", limit=10)

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], LLMPattern)
        assert result[0].pattern_type == "prompt_template"

    @pytest.mark.asyncio
    async def test_get_best_patterns_invalid_limit(self, llm_metrics_store):
        """Test getting patterns with invalid limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await llm_metrics_store.get_best_patterns("prompt_template", limit=0)


class TestGetSessionSummary:
    """Test getting session summary."""

    @pytest.mark.asyncio
    async def test_get_session_summary_success(
        self, llm_metrics_store, mock_db_pool, mock_connection
    ):
        """Test successful session summary retrieval."""
        # Setup mock
        mock_db_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_db_pool.release = AsyncMock()
        mock_connection.fetchrow = AsyncMock(
            return_value={
                "session_id": "sess_test_123",
                "total_generations": 150,
                "successful_generations": 142,
                "failed_generations": 8,
                "total_tokens": 345000,
                "total_cost_usd": 5.1,
                "avg_latency_ms": 3200.0,
                "period_start": datetime.utcnow(),
                "period_end": datetime.utcnow(),
            }
        )

        # Execute
        result = await llm_metrics_store.get_session_summary("sess_test_123")

        # Verify
        assert result is not None
        assert isinstance(result, MetricsSummary)
        assert result.total_generations == 150
        assert result.success_rate > 0.9

    @pytest.mark.asyncio
    async def test_get_session_summary_not_found(
        self, llm_metrics_store, mock_db_pool, mock_connection
    ):
        """Test session summary when session not found."""
        # Setup mock to return None
        mock_db_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_db_pool.release = AsyncMock()
        mock_connection.fetchrow = AsyncMock(return_value=None)

        # Execute
        result = await llm_metrics_store.get_session_summary("sess_nonexistent")

        # Verify
        assert result is None


class TestHealthCheck:
    """Test health check."""

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, llm_metrics_store, mock_db_pool, mock_connection
    ):
        """Test successful health check."""
        # Setup mock
        mock_db_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_db_pool.release = AsyncMock()
        mock_connection.fetchval = AsyncMock(return_value=1)

        # Execute
        result = await llm_metrics_store.health_check()

        # Verify
        assert result["status"] == "healthy"
        assert "response_time_ms" in result
        assert result["pool_size"] == 10
        assert result["pool_idle"] == 5

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self, llm_metrics_store, mock_db_pool, mock_connection
    ):
        """Test health check when database fails."""
        # Setup mock to raise exception
        mock_db_pool.acquire = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        # Execute
        result = await llm_metrics_store.health_check()

        # Verify
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert "Database connection failed" in result["error"]
