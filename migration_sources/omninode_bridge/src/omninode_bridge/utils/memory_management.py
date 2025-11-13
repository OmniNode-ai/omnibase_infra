"""Memory management utilities for OmniNode Bridge workflow processing.

Addresses performance issues with large workflow data processing by implementing:
- Streaming data processing for large workflows
- Memory-efficient pagination
- Safe dictionary access patterns
- Resource cleanup utilities
"""

import asyncio
import gc
import logging
import weakref
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """Memory usage metrics for monitoring."""

    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    timestamp: datetime


class MemoryMonitor:
    """Monitor memory usage and provide warnings for potential issues."""

    def __init__(
        self,
        warning_threshold_percent: float = 80.0,
        critical_threshold_percent: float = 90.0,
    ):
        self.warning_threshold = warning_threshold_percent
        self.critical_threshold = critical_threshold_percent
        self._process = psutil.Process()

    def get_current_memory(self) -> MemoryMetrics:
        """Get current memory usage metrics."""
        memory_info = self._process.memory_info()
        system_memory = psutil.virtual_memory()

        return MemoryMetrics(
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=self._process.memory_percent(),
            available_mb=system_memory.available / 1024 / 1024,
            timestamp=datetime.now(UTC),
        )

    def check_memory_pressure(self) -> dict[str, Any]:
        """Check if system is under memory pressure."""
        metrics = self.get_current_memory()

        status = "normal"
        if metrics.percent >= self.critical_threshold:
            status = "critical"
        elif metrics.percent >= self.warning_threshold:
            status = "warning"

        return {
            "status": status,
            "metrics": metrics,
            "recommendations": self._get_recommendations(status, metrics),
        }

    def _get_recommendations(self, status: str, metrics: MemoryMetrics) -> list[str]:
        """Get memory optimization recommendations."""
        recommendations = []

        if status == "critical":
            recommendations.extend(
                [
                    "Force garbage collection immediately",
                    "Enable workflow result streaming",
                    "Reduce workflow batch sizes",
                    "Consider workflow pagination",
                ],
            )
        elif status == "warning":
            recommendations.extend(
                [
                    "Enable periodic garbage collection",
                    "Monitor large workflow processing",
                    "Consider result streaming for large datasets",
                ],
            )

        return recommendations


class WorkflowDataStreamer:
    """Stream large workflow data to avoid memory exhaustion."""

    def __init__(
        self,
        chunk_size: int = 1000,
        memory_monitor: MemoryMonitor | None = None,
    ):
        self.chunk_size = chunk_size
        self.memory_monitor = memory_monitor or MemoryMonitor()

    async def stream_workflow_results(
        self,
        workflow_data: list[dict[str, Any]],
        max_memory_mb: float | None = None,
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        """Stream workflow results in chunks to manage memory usage.

        Args:
            workflow_data: Large list of workflow result data
            max_memory_mb: Maximum memory usage threshold

        Yields:
            Chunks of workflow data
        """
        if not workflow_data:
            return

        processed_count = 0
        total_count = len(workflow_data)

        logger.info(
            f"Starting workflow data streaming: {total_count} items, chunk_size={self.chunk_size}",
        )

        for i in range(0, total_count, self.chunk_size):
            # Check memory pressure before processing chunk
            memory_status = self.memory_monitor.check_memory_pressure()

            if memory_status["status"] == "critical":
                logger.warning(
                    f"Critical memory pressure detected at item {i}/{total_count}. "
                    f"Memory: {memory_status['metrics'].rss_mb:.1f}MB "
                    f"({memory_status['metrics'].percent:.1f}%)",
                )
                # Force garbage collection before continuing
                gc.collect()
                await asyncio.sleep(0.1)  # Brief pause to allow memory cleanup

            chunk = workflow_data[i : i + self.chunk_size]
            processed_count += len(chunk)

            logger.debug(
                f"Streaming chunk {i//self.chunk_size + 1}: {len(chunk)} items",
            )
            yield chunk

            # Periodic memory monitoring
            if processed_count % (self.chunk_size * 5) == 0:
                current_memory = self.memory_monitor.get_current_memory()
                logger.info(
                    f"Processed {processed_count}/{total_count} items. "
                    f"Memory: {current_memory.rss_mb:.1f}MB ({current_memory.percent:.1f}%)",
                )

    async def paginated_workflow_processing(
        self,
        workflow_data: list[dict[str, Any]],
        page_size: int = 100,
        max_concurrent_pages: int = 3,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Process workflow data with pagination and concurrency limits.

        Args:
            workflow_data: Workflow data to process
            page_size: Number of items per page
            max_concurrent_pages: Maximum concurrent pages to process

        Yields:
            Individual processed workflow items
        """
        total_pages = (len(workflow_data) + page_size - 1) // page_size
        semaphore = asyncio.Semaphore(max_concurrent_pages)

        async def process_page(page_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
            async with semaphore:
                # Simulate processing with memory monitoring
                memory_status = self.memory_monitor.check_memory_pressure()
                if memory_status["status"] != "normal":
                    logger.warning(
                        f"Memory pressure during page processing: {memory_status['status']}",
                    )

                # Process page data (placeholder for actual workflow processing)
                await asyncio.sleep(0.01)  # Simulate processing time
                return page_data

        logger.info(
            f"Starting paginated processing: {total_pages} pages, page_size={page_size}",
        )

        # Process pages in batches
        for page_start in range(0, len(workflow_data), page_size):
            page_end = min(page_start + page_size, len(workflow_data))
            page_data = workflow_data[page_start:page_end]

            processed_page = await process_page(page_data)

            for item in processed_page:
                yield item


class SafeDictAccessor:
    """Utility for safe dictionary access to prevent KeyError exceptions."""

    @staticmethod
    def safe_get(
        data: dict[str, Any],
        key: str,
        default: Any = None,
        required: bool = False,
    ) -> Any:
        """Safely get a value from a dictionary.

        Args:
            data: Dictionary to access
            key: Key to retrieve
            default: Default value if key not found
            required: Whether the key is required (raises ValueError if missing)

        Returns:
            Value from dictionary or default

        Raises:
            ValueError: If required=True and key is missing
        """
        if key in data:
            return data[key]

        if required:
            raise ValueError(f"Required key '{key}' not found in data")

        return default

    @staticmethod
    def safe_nested_get(
        data: dict[str, Any],
        keys: list[str],
        default: Any = None,
    ) -> Any:
        """Safely get a nested value from a dictionary.

        Args:
            data: Dictionary to access
            keys: List of keys for nested access (e.g., ['workflow', 'metadata', 'status'])
            default: Default value if any key in path is missing

        Returns:
            Nested value or default
        """
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    @staticmethod
    def safe_update(
        target: dict[str, Any],
        source: dict[str, Any],
        allowed_keys: list[str] | None = None,
    ) -> None:
        """Safely update a dictionary with another dictionary.

        Args:
            target: Dictionary to update
            source: Source dictionary with updates
            allowed_keys: If provided, only these keys are allowed to be updated
        """
        for key, value in source.items():
            if allowed_keys is None or key in allowed_keys:
                target[key] = value
            else:
                logger.warning(f"Ignoring update for disallowed key: {key}")


@asynccontextmanager
async def managed_workflow_processing(
    memory_limit_mb: float | None = None,
    auto_gc: bool = True,
    gc_interval_seconds: float = 30.0,
):
    """Context manager for workflow processing with automatic memory management.

    Args:
        memory_limit_mb: Optional memory limit in MB
        auto_gc: Whether to run automatic garbage collection
        gc_interval_seconds: Interval between garbage collection runs
    """
    memory_monitor = MemoryMonitor()
    gc_task = None

    try:
        # Start automatic garbage collection if enabled
        if auto_gc:

            async def gc_loop():
                while True:
                    await asyncio.sleep(gc_interval_seconds)
                    memory_status = memory_monitor.check_memory_pressure()

                    if memory_status["status"] in ["warning", "critical"]:
                        logger.info(
                            f"Running garbage collection due to {memory_status['status']} memory pressure",
                        )
                        collected = gc.collect()
                        logger.info(f"Garbage collection freed {collected} objects")

            gc_task = asyncio.create_task(gc_loop())

        # Check initial memory state
        initial_memory = memory_monitor.get_current_memory()
        logger.info(
            f"Starting workflow processing with {initial_memory.rss_mb:.1f}MB memory usage",
        )

        yield {
            "memory_monitor": memory_monitor,
            "streamer": WorkflowDataStreamer(memory_monitor=memory_monitor),
            "safe_dict": SafeDictAccessor(),
        }

    finally:
        # Cleanup
        if gc_task:
            gc_task.cancel()
            try:
                await gc_task
            except asyncio.CancelledError:
                pass

        # Final memory report
        final_memory = memory_monitor.get_current_memory()
        logger.info(
            f"Workflow processing completed with {final_memory.rss_mb:.1f}MB memory usage",
        )


class ResourceTracker:
    """Track and manage resource usage for workflow processing."""

    def __init__(self):
        self._resources: dict[str, weakref.WeakSet] = {}
        self._resource_counts: dict[str, int] = {}

    def register_resource(self, resource_type: str, resource: Any) -> None:
        """Register a resource for tracking."""
        if resource_type not in self._resources:
            self._resources[resource_type] = weakref.WeakSet()
            self._resource_counts[resource_type] = 0

        self._resources[resource_type].add(resource)
        self._resource_counts[resource_type] += 1

    def get_resource_stats(self) -> dict[str, Any]:
        """Get current resource usage statistics."""
        stats = {}

        for resource_type, weak_set in self._resources.items():
            active_count = len(weak_set)
            total_created = self._resource_counts[resource_type]

            stats[resource_type] = {
                "active": active_count,
                "total_created": total_created,
                "potential_leaks": max(
                    0,
                    active_count - (total_created * 0.1),
                ),  # More than 10% still active might indicate leaks
            }

        return stats

    def cleanup_stale_resources(self) -> dict[str, int]:
        """Force cleanup of stale resources."""
        cleanup_stats = {}

        for resource_type, weak_set in self._resources.items():
            initial_count = len(weak_set)

            # WeakSet automatically removes dead references
            # Force garbage collection to clean up unreferenced objects
            gc.collect()

            final_count = len(weak_set)
            cleanup_stats[resource_type] = initial_count - final_count

        return cleanup_stats


# Global resource tracker instance
resource_tracker = ResourceTracker()
