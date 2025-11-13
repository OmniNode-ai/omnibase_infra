"""
LLM Metrics Store - Async PostgreSQL Storage for Debug Intelligence.

High-performance async storage layer for LLM generation metrics, history,
and learned patterns with comprehensive error handling and monitoring.

ONEX v2.0 Compliance:
- Async/await throughout
- Circuit breaker pattern for resilience
- Comprehensive error handling
- Performance monitoring
- Type-safe operations with Pydantic models
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import asyncpg
from omnibase_core import EnumCoreErrorCode, ModelOnexError

from omninode_bridge.intelligence import queries
from omninode_bridge.intelligence.models import (
    LLMGenerationHistory,
    LLMGenerationMetric,
    LLMPattern,
    MetricsSummary,
)

logger = logging.getLogger(__name__)

# Alias for compatibility
OnexError = ModelOnexError


class LLMMetricsStoreError(Exception):
    """Base exception for LLM metrics store operations."""

    pass


class LLMMetricsStore:
    """
    Async storage for LLM generation metrics and debug intelligence.

    Provides high-performance PostgreSQL-backed storage for:
    - LLM generation metrics (tokens, cost, latency, success/failure)
    - Generation history (prompts, outputs, quality scores)
    - Learned patterns (templates, structures, best practices)

    Usage:
        >>> store = LLMMetricsStore(db_pool)
        >>> metric = LLMGenerationMetric(
        ...     session_id="sess_123",
        ...     node_type="effect",
        ...     model_tier="tier_2",
        ...     model_name="claude-sonnet-4",
        ...     prompt_tokens=1500,
        ...     completion_tokens=800,
        ...     total_tokens=2300,
        ...     latency_ms=3500.0,
        ...     cost_usd=0.0345,
        ...     success=True
        ... )
        >>> metric_id = await store.store_generation_metric(metric)
    """

    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize LLM metrics store.

        Args:
            db_pool: asyncpg connection pool (must be pre-initialized)

        Raises:
            ValueError: If db_pool is None
        """
        if db_pool is None:
            raise ValueError("db_pool cannot be None")

        self.pool = db_pool
        self._metrics_cache: dict[str, Any] = {
            "total_operations": 0,
            "failed_operations": 0,
            "avg_operation_time_ms": 0.0,
        }

        logger.info("LLMMetricsStore initialized with existing connection pool")

    @asynccontextmanager
    async def _acquire_connection(self):
        """
        Acquire database connection with error handling.

        Yields:
            Database connection

        Raises:
            OnexError: If connection acquisition fails
        """
        connection = None
        start_time = time.perf_counter()

        try:
            connection = await self.pool.acquire()
            if connection is None:
                raise OnexError(
                    error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                    message="Failed to acquire database connection - pool returned None",
                    context={"operation": "acquire_connection"},
                )
            yield connection

        except OnexError:
            # Re-raise OnexError exceptions without wrapping
            raise

        except asyncpg.PostgresError as e:
            self._metrics_cache["failed_operations"] += 1
            logger.error(f"PostgreSQL connection error: {e}")
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_CONNECTION_ERROR,
                message=f"Database connection error: {e}",
                context={"operation": "acquire_connection"},
                original_error=e,
            )

        except Exception as e:
            self._metrics_cache["failed_operations"] += 1
            logger.error(f"Unexpected connection error: {e}")
            raise OnexError(
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                message=f"Unexpected connection error: {e}",
                context={"operation": "acquire_connection"},
                original_error=e,
            )

        finally:
            if connection:
                try:
                    await self.pool.release(connection)
                except Exception as e:
                    logger.warning(f"Failed to release connection: {e}")

            # Update metrics
            execution_time = (time.perf_counter() - start_time) * 1000
            self._metrics_cache["total_operations"] += 1
            count = self._metrics_cache["total_operations"]
            current_avg = self._metrics_cache["avg_operation_time_ms"]
            self._metrics_cache["avg_operation_time_ms"] = (
                (current_avg * (count - 1)) + execution_time
            ) / count

    async def store_generation_metric(
        self,
        metric: LLMGenerationMetric,
    ) -> str:
        """
        Store LLM generation metric.

        Args:
            metric: LLM generation metric to store

        Returns:
            Metric ID (UUID as string)

        Raises:
            OnexError: If storage operation fails
            ValueError: If metric is None or invalid
        """
        if metric is None:
            raise ValueError("metric cannot be None")

        start_time = time.perf_counter()

        try:
            async with self._acquire_connection() as conn:
                # Convert metadata dict to JSON string for asyncpg
                metadata_json = json.dumps(metric.metadata) if metric.metadata else "{}"

                result = await conn.fetchrow(
                    queries.INSERT_GENERATION_METRIC,
                    metric.metric_id,
                    metric.session_id,
                    metric.correlation_id,
                    metric.node_type,
                    metric.model_tier,
                    metric.model_name,
                    metric.prompt_tokens,
                    metric.completion_tokens,
                    metric.total_tokens,
                    metric.latency_ms,
                    metric.cost_usd,
                    metric.success,
                    metric.error_message,
                    metadata_json,
                    metric.created_at,
                )

                if result is None:
                    raise OnexError(
                        error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                        message="Failed to store generation metric - database returned None",
                        context={
                            "operation": "store_generation_metric",
                            "session_id": metric.session_id,
                            "node_type": metric.node_type,
                        },
                    )

                execution_time = (time.perf_counter() - start_time) * 1000
                logger.debug(
                    f"Stored generation metric: {result['metric_id']} "
                    f"(session={metric.session_id}, tokens={metric.total_tokens}, "
                    f"cost=${metric.cost_usd:.4f}, time={execution_time:.2f}ms)"
                )

                return str(result["metric_id"])

        except OnexError:
            raise
        except asyncpg.PostgresError as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Database error storing generation metric: {e}",
                context={
                    "operation": "store_generation_metric",
                    "session_id": metric.session_id,
                    "execution_time_ms": execution_time,
                },
                original_error=e,
            )
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            raise OnexError(
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                message=f"Unexpected error storing generation metric: {e}",
                context={
                    "operation": "store_generation_metric",
                    "session_id": metric.session_id,
                    "execution_time_ms": execution_time,
                },
                original_error=e,
            )

    async def store_generation_history(
        self,
        history: LLMGenerationHistory,
    ) -> str:
        """
        Store generation history with prompt and output.

        Args:
            history: Generation history to store

        Returns:
            History ID (UUID as string)

        Raises:
            OnexError: If storage operation fails
            ValueError: If history is None or invalid
        """
        if history is None:
            raise ValueError("history cannot be None")

        start_time = time.perf_counter()

        try:
            async with self._acquire_connection() as conn:
                # Convert validation_errors dict to JSON string for asyncpg
                validation_errors_json = (
                    json.dumps(history.validation_errors)
                    if history.validation_errors
                    else None
                )

                result = await conn.fetchrow(
                    queries.INSERT_GENERATION_HISTORY,
                    history.history_id,
                    history.metric_id,
                    history.prompt_text,
                    history.generated_text,
                    history.quality_score,
                    history.validation_passed,
                    validation_errors_json,
                    history.created_at,
                )

                if result is None:
                    raise OnexError(
                        error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                        message="Failed to store generation history - database returned None",
                        context={
                            "operation": "store_generation_history",
                            "metric_id": str(history.metric_id),
                        },
                    )

                execution_time = (time.perf_counter() - start_time) * 1000
                logger.debug(
                    f"Stored generation history: {result['history_id']} "
                    f"(metric={history.metric_id}, quality={history.quality_score}, "
                    f"time={execution_time:.2f}ms)"
                )

                return str(result["history_id"])

        except OnexError:
            raise
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Failed to store generation history: {e}",
                context={
                    "operation": "store_generation_history",
                    "metric_id": str(history.metric_id),
                    "execution_time_ms": execution_time,
                },
                original_error=e,
            )

    async def get_metrics_by_session(
        self,
        session_id: str,
    ) -> list[LLMGenerationMetric]:
        """
        Get all metrics for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of generation metrics (may be empty)

        Raises:
            OnexError: If query fails
            ValueError: If session_id is None or empty
        """
        if not session_id:
            raise ValueError("session_id cannot be None or empty")

        try:
            async with self._acquire_connection() as conn:
                rows = await conn.fetch(queries.SELECT_METRICS_BY_SESSION, session_id)

                if rows is None:
                    raise OnexError(
                        error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                        message="Query returned None",
                        context={
                            "operation": "get_metrics_by_session",
                            "session_id": session_id,
                        },
                    )

                # Convert rows to dicts and parse JSON fields
                metrics = []
                for row in rows:
                    row_dict = dict(row)
                    # Parse JSON string back to dict
                    if isinstance(row_dict.get("metadata"), str):
                        row_dict["metadata"] = json.loads(row_dict["metadata"])
                    metrics.append(LLMGenerationMetric(**row_dict))

                logger.debug(
                    f"Retrieved {len(metrics)} metrics for session {session_id}"
                )
                return metrics

        except OnexError:
            raise
        except Exception as e:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Failed to get metrics by session: {e}",
                context={
                    "operation": "get_metrics_by_session",
                    "session_id": session_id,
                },
                original_error=e,
            )

    async def get_average_metrics(
        self,
        model_name: str,
        days: int = 7,
    ) -> dict[str, float]:
        """
        Get average metrics for a model over N days.

        Args:
            model_name: Model name to query
            days: Number of days to look back (default: 7)

        Returns:
            Dictionary with average metrics

        Raises:
            OnexError: If query fails
            ValueError: If model_name is None or days < 1
        """
        if not model_name:
            raise ValueError("model_name cannot be None or empty")
        if days < 1:
            raise ValueError("days must be >= 1")

        try:
            async with self._acquire_connection() as conn:
                row = await conn.fetchrow(
                    queries.SELECT_AVERAGE_METRICS,
                    model_name,
                    days,
                )

                if row is None:
                    # No data found - return empty metrics
                    return {
                        "total_generations": 0,
                        "successful_generations": 0,
                        "failed_generations": 0,
                        "avg_latency_ms": 0.0,
                        "avg_prompt_tokens": 0.0,
                        "avg_completion_tokens": 0.0,
                        "avg_total_tokens": 0.0,
                        "avg_cost_usd": 0.0,
                        "total_tokens": 0,
                        "total_cost_usd": 0.0,
                    }

                metrics = dict(row)
                logger.debug(
                    f"Retrieved average metrics for {model_name} "
                    f"({days} days): {metrics['total_generations']} generations"
                )
                return metrics

        except OnexError:
            raise
        except Exception as e:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Failed to get average metrics: {e}",
                context={
                    "operation": "get_average_metrics",
                    "model_name": model_name,
                    "days": days,
                },
                original_error=e,
            )

    async def store_learned_pattern(
        self,
        pattern: LLMPattern,
    ) -> str:
        """
        Store a learned pattern (upsert).

        If pattern already exists, updates usage count and metrics.

        Args:
            pattern: Pattern to store

        Returns:
            Pattern ID (UUID as string)

        Raises:
            OnexError: If storage operation fails
            ValueError: If pattern is None or invalid
        """
        if pattern is None:
            raise ValueError("pattern cannot be None")

        start_time = time.perf_counter()

        try:
            async with self._acquire_connection() as conn:
                # Convert pattern_data and metadata dicts to JSON strings for asyncpg
                pattern_data_json = json.dumps(pattern.pattern_data)
                metadata_json = (
                    json.dumps(pattern.metadata) if pattern.metadata else "{}"
                )

                result = await conn.fetchrow(
                    queries.INSERT_PATTERN,
                    pattern.pattern_id,
                    pattern.pattern_type,
                    pattern.node_type,
                    pattern_data_json,
                    pattern.usage_count,
                    pattern.avg_quality_score,
                    pattern.success_rate,
                    metadata_json,
                    pattern.created_at,
                    pattern.updated_at,
                )

                if result is None:
                    raise OnexError(
                        error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                        message="Failed to store pattern - database returned None",
                        context={
                            "operation": "store_learned_pattern",
                            "pattern_type": pattern.pattern_type,
                        },
                    )

                execution_time = (time.perf_counter() - start_time) * 1000
                logger.debug(
                    f"Stored pattern: {result['pattern_id']} "
                    f"(type={pattern.pattern_type}, usage={pattern.usage_count}, "
                    f"time={execution_time:.2f}ms)"
                )

                return str(result["pattern_id"])

        except OnexError:
            raise
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Failed to store learned pattern: {e}",
                context={
                    "operation": "store_learned_pattern",
                    "pattern_type": pattern.pattern_type,
                    "execution_time_ms": execution_time,
                },
                original_error=e,
            )

    async def get_best_patterns(
        self,
        pattern_type: str,
        limit: int = 10,
    ) -> list[LLMPattern]:
        """
        Get best patterns by quality score.

        Args:
            pattern_type: Type of pattern to retrieve
            limit: Maximum number of patterns to return (default: 10)

        Returns:
            List of patterns ordered by quality score (may be empty)

        Raises:
            OnexError: If query fails
            ValueError: If pattern_type is None or limit < 1
        """
        if not pattern_type:
            raise ValueError("pattern_type cannot be None or empty")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        try:
            async with self._acquire_connection() as conn:
                rows = await conn.fetch(
                    queries.SELECT_BEST_PATTERNS,
                    pattern_type,
                    limit,
                )

                if rows is None:
                    raise OnexError(
                        error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                        message="Query returned None",
                        context={
                            "operation": "get_best_patterns",
                            "pattern_type": pattern_type,
                        },
                    )

                # Convert rows to dicts and parse JSON fields
                patterns = []
                for row in rows:
                    row_dict = dict(row)
                    # Parse JSON strings back to dicts
                    if isinstance(row_dict.get("pattern_data"), str):
                        row_dict["pattern_data"] = json.loads(row_dict["pattern_data"])
                    if isinstance(row_dict.get("metadata"), str):
                        row_dict["metadata"] = json.loads(row_dict["metadata"])
                    patterns.append(LLMPattern(**row_dict))

                logger.debug(
                    f"Retrieved {len(patterns)} best patterns for type {pattern_type}"
                )
                return patterns

        except OnexError:
            raise
        except Exception as e:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Failed to get best patterns: {e}",
                context={
                    "operation": "get_best_patterns",
                    "pattern_type": pattern_type,
                },
                original_error=e,
            )

    async def get_session_summary(
        self,
        session_id: str,
    ) -> Optional[MetricsSummary]:
        """
        Get aggregated summary for a session.

        Args:
            session_id: Session identifier

        Returns:
            Metrics summary or None if session not found

        Raises:
            OnexError: If query fails
            ValueError: If session_id is None or empty
        """
        if not session_id:
            raise ValueError("session_id cannot be None or empty")

        try:
            async with self._acquire_connection() as conn:
                row = await conn.fetchrow(queries.SELECT_SESSION_SUMMARY, session_id)

                if row is None:
                    logger.debug(f"No summary found for session {session_id}")
                    return None

                data = dict(row)
                success_rate = (
                    data["successful_generations"] / data["total_generations"]
                    if data["total_generations"] > 0
                    else 0.0
                )

                summary = MetricsSummary(
                    total_generations=data["total_generations"],
                    successful_generations=data["successful_generations"],
                    failed_generations=data["failed_generations"],
                    total_tokens=data["total_tokens"],
                    total_cost_usd=data["total_cost_usd"],
                    avg_latency_ms=data["avg_latency_ms"],
                    success_rate=success_rate,
                    period_start=data["period_start"],
                    period_end=data["period_end"],
                )

                logger.debug(f"Retrieved session summary for {session_id}")
                return summary

        except OnexError:
            raise
        except Exception as e:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Failed to get session summary: {e}",
                context={
                    "operation": "get_session_summary",
                    "session_id": session_id,
                },
                original_error=e,
            )

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on metrics store.

        Returns:
            Health check results

        Raises:
            OnexError: If health check fails critically
        """
        try:
            start_time = time.perf_counter()

            async with self._acquire_connection() as conn:
                result = await conn.fetchval("SELECT 1")

            response_time = (time.perf_counter() - start_time) * 1000

            if result != 1:
                raise OnexError(
                    error_code=EnumCoreErrorCode.DATABASE_OPERATION_ERROR,
                    message=f"Health check returned unexpected value: {result}",
                    context={"operation": "health_check"},
                )

            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "pool_size": self.pool.get_size(),
                "pool_idle": self.pool.get_idle_size(),
                "metrics": self._metrics_cache,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_type": type(e).__name__,
                "metrics": self._metrics_cache,
            }
