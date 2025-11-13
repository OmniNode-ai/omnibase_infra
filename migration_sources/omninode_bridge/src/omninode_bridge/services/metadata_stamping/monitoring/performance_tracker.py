"""Performance tracking decorators and context managers.

Provides easy integration with existing service components to automatically
track performance metrics without modifying core business logic.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional, TypeVar

from .metrics_collector import MetricsCollector, OperationType, PerformanceMetric

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class PerformanceTracker:
    """Performance tracker with decorators and context managers."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance tracker.

        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector

    def track_operation(
        self, operation_type: OperationType, metadata: Optional[dict[str, Any]] = None
    ):
        """Decorator to track operation performance.

        Args:
            operation_type: Type of operation being tracked
            metadata: Additional metadata to record

        Returns:
            Decorated function
        """

        def decorator(func: F) -> F:
            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self.metrics_collector.record_operation(
                        *args,
                        operation_type=operation_type,
                        operation_func=func,
                        metadata=metadata,
                        **kwargs,
                    )

                return async_wrapper
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return asyncio.run(
                        self.metrics_collector.record_operation(
                            *args,
                            operation_type=operation_type,
                            operation_func=func,
                            metadata=metadata,
                            **kwargs,
                        )
                    )

                return sync_wrapper

        return decorator

    @asynccontextmanager
    async def track_context(
        self, operation_type: OperationType, metadata: Optional[dict[str, Any]] = None
    ):
        """Context manager to track operation performance.

        Args:
            operation_type: Type of operation being tracked
            metadata: Additional metadata to record

        Yields:
            Context with timing information
        """
        start_time = time.perf_counter()
        success = True
        error_type = None
        context = {"start_time": start_time}

        try:
            yield context
        except Exception as e:
            success = False
            error_type = type(e).__name__
            logger.error(f"Operation {operation_type.value} failed in context: {e}")
            raise
        finally:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            context["execution_time_ms"] = execution_time_ms

            metric = PerformanceMetric(
                operation_type=operation_type,
                execution_time_ms=execution_time_ms,
                timestamp=datetime.now(),
                success=success,
                error_type=error_type,
                metadata=metadata or {},
            )

            self.metrics_collector.record_metric(metric)

    def track_database_operation(self, operation_subtype: str = "query"):
        """Decorator specifically for database operations.

        Args:
            operation_subtype: Subtype of database operation

        Returns:
            Decorated function
        """
        operation_mapping = {
            "query": OperationType.DATABASE_QUERY,
            "select": OperationType.DATABASE_QUERY,
            "insert": OperationType.DATABASE_INSERT,
            "update": OperationType.DATABASE_UPDATE,
            "delete": OperationType.DATABASE_UPDATE,
            "batch": OperationType.DATABASE_BATCH,
        }

        operation_type = operation_mapping.get(
            operation_subtype.lower(), OperationType.DATABASE_QUERY
        )

        return self.track_operation(
            operation_type=operation_type,
            metadata={"operation_subtype": operation_subtype},
        )

    def track_hash_generation(self, file_size_bytes: Optional[int] = None):
        """Decorator specifically for hash generation operations.

        Args:
            file_size_bytes: Size of file being hashed

        Returns:
            Decorated function
        """
        metadata = {}
        if file_size_bytes is not None:
            metadata["file_size_bytes"] = file_size_bytes

        return self.track_operation(
            operation_type=OperationType.HASH_GENERATION, metadata=metadata
        )

    def track_batch_processing(self, batch_size: Optional[int] = None):
        """Decorator specifically for batch processing operations.

        Args:
            batch_size: Number of items in batch

        Returns:
            Decorated function
        """
        metadata = {}
        if batch_size is not None:
            metadata["batch_size"] = batch_size

        return self.track_operation(
            operation_type=OperationType.BATCH_PROCESSING, metadata=metadata
        )

    def track_api_request(self, endpoint: str, method: str = "GET"):
        """Decorator specifically for API request tracking.

        Args:
            endpoint: API endpoint being called
            method: HTTP method

        Returns:
            Decorated function
        """
        return self.track_operation(
            operation_type=OperationType.API_REQUEST,
            metadata={"endpoint": endpoint, "method": method},
        )

    def create_batch_tracker(
        self, operation_type: OperationType = OperationType.BATCH_PROCESSING
    ):
        """Create a batch operation tracker.

        Args:
            operation_type: Type of batch operation

        Returns:
            Batch tracker instance
        """
        return BatchTracker(self.metrics_collector, operation_type)


class BatchTracker:
    """Specialized tracker for batch operations to calculate throughput."""

    def __init__(
        self, metrics_collector: MetricsCollector, operation_type: OperationType
    ):
        """Initialize batch tracker.

        Args:
            metrics_collector: Metrics collector instance
            operation_type: Type of batch operation
        """
        self.metrics_collector = metrics_collector
        self.operation_type = operation_type
        self.start_time: Optional[float] = None
        self.items_processed = 0
        self.batch_metadata: dict[str, Any] = {}

    def start_batch(self, expected_items: Optional[int] = None, **metadata):
        """Start tracking a batch operation.

        Args:
            expected_items: Expected number of items to process
            **metadata: Additional metadata for the batch
        """
        self.start_time = time.perf_counter()
        self.items_processed = 0
        self.batch_metadata = {"expected_items": expected_items, **metadata}

    def add_item(self, count: int = 1):
        """Add processed items to the batch.

        Args:
            count: Number of items processed
        """
        self.items_processed += count

    async def finish_batch(
        self, success: bool = True, error_type: Optional[str] = None
    ):
        """Finish tracking the batch operation.

        Args:
            success: Whether the batch operation succeeded
            error_type: Type of error if operation failed
        """
        if self.start_time is None:
            logger.warning("Batch tracker finished without being started")
            return

        execution_time_ms = (time.perf_counter() - self.start_time) * 1000

        # Calculate throughput
        throughput = 0
        if execution_time_ms > 0:
            throughput = (
                self.items_processed / execution_time_ms
            ) * 1000  # items per second

        metadata = {
            **self.batch_metadata,
            "items_processed": self.items_processed,
            "throughput_per_second": throughput,
        }

        metric = PerformanceMetric(
            operation_type=self.operation_type,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now(),
            success=success,
            error_type=error_type,
            metadata=metadata,
        )

        self.metrics_collector.record_metric(metric)

        # Reset for potential reuse
        self.start_time = None
        self.items_processed = 0
        self.batch_metadata = {}

    @asynccontextmanager
    async def track_batch_context(
        self, expected_items: Optional[int] = None, **metadata
    ):
        """Context manager for tracking batch operations.

        Args:
            expected_items: Expected number of items to process
            **metadata: Additional metadata for the batch

        Yields:
            Batch tracker instance
        """
        self.start_batch(expected_items, **metadata)
        success = True
        error_type = None

        try:
            yield self
        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise
        finally:
            await self.finish_batch(success, error_type)


# Global performance tracker instance (initialized by service)
_global_tracker: Optional[PerformanceTracker] = None


def set_global_tracker(tracker: PerformanceTracker):
    """Set the global performance tracker.

    Args:
        tracker: Performance tracker instance
    """
    global _global_tracker
    _global_tracker = tracker


def get_global_tracker() -> Optional[PerformanceTracker]:
    """Get the global performance tracker.

    Returns:
        Performance tracker instance or None if not set
    """
    return _global_tracker


# Convenience decorators that use the global tracker
def track_database(operation_subtype: str = "query"):
    """Convenience decorator for database operations using global tracker.

    Args:
        operation_subtype: Subtype of database operation

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        if _global_tracker is None:
            logger.warning("Global performance tracker not set, skipping tracking")
            return func
        return _global_tracker.track_database_operation(operation_subtype)(func)

    return decorator


def track_hash():
    """Convenience decorator for hash generation using global tracker.

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        if _global_tracker is None:
            logger.warning("Global performance tracker not set, skipping tracking")
            return func
        return _global_tracker.track_hash_generation()(func)

    return decorator


def track_api(endpoint: str, method: str = "GET"):
    """Convenience decorator for API requests using global tracker.

    Args:
        endpoint: API endpoint being called
        method: HTTP method

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        if _global_tracker is None:
            logger.warning("Global performance tracker not set, skipping tracking")
            return func
        return _global_tracker.track_api_request(endpoint, method)(func)

    return decorator
