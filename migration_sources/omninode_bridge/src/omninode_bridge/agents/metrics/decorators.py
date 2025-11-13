"""
Decorators and context managers for zero-boilerplate metrics.

Provides @timed, @counted decorators and timing() context manager.
"""

import asyncio
import functools
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Optional, TypeVar

from omninode_bridge.agents.metrics.collector import MetricsCollector

# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None

T = TypeVar("T")


def set_metrics_collector(collector: MetricsCollector) -> None:
    """
    Set global metrics collector instance.

    Args:
        collector: MetricsCollector instance
    """
    global _metrics_collector
    _metrics_collector = collector


def get_metrics_collector() -> Optional[MetricsCollector]:
    """
    Get global metrics collector instance.

    Returns:
        MetricsCollector instance or None
    """
    return _metrics_collector


def timed(
    metric_name: str,
    tags: Optional[dict[str, str]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically time function execution.

    Args:
        metric_name: Metric name (e.g., "operation_time_ms")
        tags: Optional tags for filtering/grouping

    Returns:
        Decorator function

    Usage:
        @timed("parse_contract_time_ms", {"type": "yaml"})
        async def parse_contract(yaml_content: str):
            # Implementation
            pass

    Performance: <1ms overhead
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            collector = get_metrics_collector()
            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000

                if collector:
                    # Fire and forget (non-blocking)
                    asyncio.create_task(
                        collector.record_timing(metric_name, duration_ms, tags=tags)
                    )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            collector = get_metrics_collector()
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000

                if collector:
                    # Fire and forget (non-blocking)
                    asyncio.create_task(
                        collector.record_timing(metric_name, duration_ms, tags=tags)
                    )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def counted(
    metric_name: str,
    tags: Optional[dict[str, str]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically count function calls.

    Args:
        metric_name: Metric name (e.g., "operation_count")
        tags: Optional tags for filtering/grouping

    Returns:
        Decorator function

    Usage:
        @counted("cache_hit_count", {"cache": "template"})
        async def get_from_cache(key: str):
            # Implementation
            pass

    Performance: <0.5ms overhead
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            collector = get_metrics_collector()

            # Record count (fire and forget)
            if collector:
                asyncio.create_task(
                    collector.record_counter(metric_name, count=1, tags=tags)
                )

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            collector = get_metrics_collector()

            # Record count (fire and forget)
            if collector:
                asyncio.create_task(
                    collector.record_counter(metric_name, count=1, tags=tags)
                )

            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


@asynccontextmanager
async def timing(
    metric_name: str,
    tags: Optional[dict[str, str]] = None,
    collector: Optional[MetricsCollector] = None,
):
    """
    Context manager for timing code blocks.

    Args:
        metric_name: Metric name
        tags: Optional tags
        collector: Optional collector (uses global if not provided)

    Yields:
        None

    Usage:
        async with timing("parse_time_ms", {"type": "yaml"}):
            contract = await parse_yaml(content)

    Performance: <1ms overhead
    """
    if collector is None:
        collector = get_metrics_collector()

    start_time = time.perf_counter()

    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000

        if collector:
            # Fire and forget (non-blocking)
            asyncio.create_task(
                collector.record_timing(metric_name, duration_ms, tags=tags)
            )
