#!/usr/bin/env python3
"""
Metrics Collection Pattern Generator for ONEX v2.0 Code Generation.

Generates comprehensive metrics tracking code for operation, resource, and business metrics.
Designed to integrate with MixinMetrics from omnibase_core.

Performance Requirements:
- <1ms overhead per operation
- Efficient aggregation using collections.deque
- Periodic publishing (every 100 operations or 60 seconds)
- Memory-efficient storage with bounded queues
- Thread-safe with asyncio locks

ONEX v2.0 Compliance:
- Suffix-based naming conventions
- Type-safe with proper annotations
- Event-driven metrics publishing
- Comprehensive observability

Example Usage:
    >>> from omninode_bridge.codegen.patterns.metrics import generate_metrics_initialization
    >>> init_code = generate_metrics_initialization(["orchestration", "validation"])
    >>> print(init_code)

Author: Code Generation System
Last Updated: 2025-11-05
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# === Configuration Models ===


@dataclass
class MetricsConfiguration:
    """
    Configuration for metrics collection patterns.

    Attributes:
        operations: List of operation names to track
        service_name: Name of service for metrics publishing
        publish_interval_ops: Publish metrics every N operations (default: 100)
        publish_interval_seconds: Publish metrics every N seconds (default: 60)
        max_duration_samples: Maximum duration samples to keep (default: 1000)
        enable_resource_metrics: Enable resource (CPU/memory) monitoring
        enable_business_metrics: Enable business/KPI metrics
        percentiles: Percentiles to calculate (default: [50, 95, 99])
    """

    operations: list[str] = field(default_factory=list)
    service_name: str = "node_service"
    publish_interval_ops: int = 100
    publish_interval_seconds: int = 60
    max_duration_samples: int = 1000
    enable_resource_metrics: bool = True
    enable_business_metrics: bool = True
    percentiles: list[int] = field(default_factory=lambda: [50, 95, 99])


# === Code Generation Functions ===


def generate_metrics_initialization(
    operations: list[str],
    max_duration_samples: int = 1000,
    enable_resource_metrics: bool = True,
    enable_business_metrics: bool = True,
    publish_interval_seconds: int = 60,
    publish_interval_ops: int = 100,
) -> str:
    """
    Generate metrics tracking initialization code.

    Produces code for __init__ method to initialize metrics data structures.
    Uses collections.deque for efficient bounded queues.

    Args:
        operations: List of operation names to track (e.g., ["orchestration", "validation"])
        max_duration_samples: Maximum duration samples to keep (default: 1000)
        enable_resource_metrics: Include resource metrics initialization
        enable_business_metrics: Include business metrics initialization
        publish_interval_seconds: Publish metrics every N seconds (default: 60)
        publish_interval_ops: Publish metrics every N operations (default: 100)

    Returns:
        Python code string for metrics initialization

    Example:
        >>> code = generate_metrics_initialization(["orchestration", "validation"])
        >>> print(code)
    """
    # Input validation
    if not isinstance(operations, list):
        raise TypeError(
            f"operations must be a list, got: {type(operations).__name__}. "
            f"Example: ['orchestration', 'validation', 'aggregation']"
        )

    if not operations:
        raise ValueError(
            "operations must contain at least one operation, got empty list. "
            "Example: ['orchestration', 'validation', 'aggregation']"
        )

    for op in operations:
        if not isinstance(op, str) or not op:
            raise ValueError(
                f"All operations must be non-empty strings, got invalid operation: {op!r}. "
                f"Valid examples: 'orchestration', 'validation', 'query'"
            )

    if not isinstance(max_duration_samples, int):
        raise TypeError(
            f"max_duration_samples must be an integer, got: {type(max_duration_samples).__name__}. "
            f"Valid examples: 1000, 5000, 10000"
        )

    if max_duration_samples < 1:
        raise ValueError(
            f"max_duration_samples must be at least 1, got: {max_duration_samples}. "
            f"Valid examples: 1000, 5000, 10000"
        )

    if not isinstance(enable_resource_metrics, bool):
        raise TypeError(
            f"enable_resource_metrics must be a boolean, got: {type(enable_resource_metrics).__name__}"
        )

    if not isinstance(enable_business_metrics, bool):
        raise TypeError(
            f"enable_business_metrics must be a boolean, got: {type(enable_business_metrics).__name__}"
        )

    if not isinstance(publish_interval_seconds, int):
        raise TypeError(
            f"publish_interval_seconds must be an integer, got: {type(publish_interval_seconds).__name__}. "
            f"Valid examples: 30, 60, 120"
        )

    if publish_interval_seconds < 1:
        raise ValueError(
            f"publish_interval_seconds must be at least 1, got: {publish_interval_seconds}. "
            f"Valid examples: 30, 60, 120"
        )

    if not isinstance(publish_interval_ops, int):
        raise TypeError(
            f"publish_interval_ops must be an integer, got: {type(publish_interval_ops).__name__}. "
            f"Valid examples: 50, 100, 500"
        )

    if publish_interval_ops < 1:
        raise ValueError(
            f"publish_interval_ops must be at least 1, got: {publish_interval_ops}. "
            f"Valid examples: 50, 100, 500"
        )

    operations_init = "\n".join(
        [
            f'            "{op}": {{'
            f'"count": 0, '
            f'"duration_ms": deque(maxlen={max_duration_samples}), '
            f'"errors": 0, '
            f'"last_error": None'
            f"}},"
            for op in operations
        ]
    )

    resource_section = ""
    if enable_resource_metrics:
        resource_section = """
        self._metrics["resources"] = {
            "memory_mb": 0,
            "cpu_percent": 0.0,
            "active_connections": 0,
            "queue_depth": 0,
        }
        self._last_resource_check = time.time()
        self._resource_check_interval = 30  # seconds"""

    business_section = ""
    if enable_business_metrics:
        business_section = """
        self._metrics["business"] = {
            "items_processed": 0,
            "throughput_per_second": 0.0,
            "custom_kpis": {},
        }"""

    return f"""# Metrics initialization (imported in __init__)
from collections import deque
import asyncio
import time
import statistics
import psutil
import os

# Initialize metrics tracking
self._metrics = {{
    "operations": {{
{operations_init}
    }},
}}
self._metrics_lock = asyncio.Lock()
self._last_publish = time.time()
self._publish_interval_seconds = {publish_interval_seconds}
self._publish_interval_ops = {publish_interval_ops}
{resource_section}
{business_section}"""


def generate_operation_metrics_tracking(
    percentiles: Optional[list[int]] = None,
) -> str:
    """
    Generate operation-level metrics tracking code.

    Produces async method to track operation execution metrics:
    - Execution count
    - Duration tracking with percentiles
    - Error tracking with last error details
    - Automatic periodic publishing

    Args:
        percentiles: Percentiles to calculate (default: [50, 95, 99])

    Returns:
        Python code string for operation metrics tracking method

    Performance:
        - <1ms overhead per operation
        - Efficient deque operations (O(1) append)
        - Lock-free fast path for reads
    """
    if percentiles is None:
        percentiles = [50, 95, 99]

    # Generate guarded percentile calculation code
    # Initialize all percentile variables (8 spaces indent - inside async with block)
    percentile_vars = "\n        ".join([f"p{p} = 0.0" for p in percentiles])
    # Calculate percentile assignments (12 spaces indent - inside if block)
    percentile_assignments = "\n            ".join(
        [
            f"p{p} = sorted_times[int(len(sorted_times) * {p / 100.0})]"
            for p in percentiles
        ]
    )
    # Build dictionary entries (12 spaces indent - inside return dict)
    percentile_dict_entries = "\n            ".join(
        [f'"p{p}": p{p},' for p in percentiles]
    )

    return f'''async def _track_operation_metrics(
    self,
    operation: str,
    duration_ms: float,
    success: bool,
    error_details: Optional[str] = None,
) -> None:
    """
    Track operation-level metrics with <1ms overhead.

    Args:
        operation: Operation name (must exist in self._metrics["operations"])
        duration_ms: Operation duration in milliseconds
        success: Whether operation succeeded
        error_details: Error details if operation failed

    Performance:
        - <1ms overhead per call
        - Efficient deque operations
        - Async lock for thread safety
        - Periodic publishing (every 100 ops or 60s)
    """
    if operation not in self._metrics["operations"]:
        logger.warning(f"Unknown operation for metrics: {{operation}}")
        return

    start = time.perf_counter()

    async with self._metrics_lock:
        metrics = self._metrics["operations"][operation]
        metrics["count"] += 1
        metrics["duration_ms"].append(duration_ms)

        if not success:
            metrics["errors"] += 1
            metrics["last_error"] = {{
                "timestamp": time.time(),
                "details": error_details,
            }}

        # Check if we should publish metrics
        should_publish = (
            metrics["count"] % self._publish_interval_ops == 0
            or (time.time() - self._last_publish) >= self._publish_interval_seconds
        )

        if should_publish:
            await self._publish_metrics_event(operation, metrics)
            self._last_publish = time.time()

    # Track overhead (should be <1ms)
    overhead_ms = (time.perf_counter() - start) * 1000
    if overhead_ms > 1.0:
        logger.warning(
            f"Metrics tracking overhead exceeded 1ms: {{overhead_ms:.2f}}ms for {{operation}}"
        )


async def _calculate_operation_statistics(
    self,
    operation: str,
) -> dict[str, float]:
    """
    Calculate operation statistics from tracked metrics.

    Args:
        operation: Operation name

    Returns:
        Statistics dictionary with avg, min, max, and percentiles
    """
    async with self._metrics_lock:
        metrics = self._metrics["operations"][operation]
        durations = list(metrics["duration_ms"])

        if not durations:
            return {{
                "count": 0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "errors": 0,
            }}

        # Guard against empty or small samples for percentile calculation
        sorted_times = sorted(durations)
        {percentile_vars}

        if len(sorted_times) == 1:
            # Single sample - use that value for all percentiles
            {percentile_assignments}
        elif len(sorted_times) > 1:
            # Normal case - calculate percentiles
            {percentile_assignments}

        return {{
            "count": metrics["count"],
            "avg": statistics.mean(durations),
            "min": min(durations),
            "max": max(durations),
            {percentile_dict_entries}
            "errors": metrics["errors"],
            "error_rate": metrics["errors"] / metrics["count"] if metrics["count"] > 0 else 0.0,
        }}'''


def generate_resource_metrics_collection() -> str:
    """
    Generate resource monitoring code.

    Produces async method to collect resource metrics:
    - Memory usage (RSS and available)
    - CPU usage percentage
    - Active connections (if applicable)
    - Queue depth tracking

    Returns:
        Python code string for resource metrics collection method

    Performance:
        - Throttled collection (every 30 seconds)
        - Non-blocking via asyncio.to_thread
        - Minimal overhead on hot path
    """
    return '''async def _collect_resource_metrics(self) -> None:
    """
    Collect resource usage metrics.

    Collects:
    - Memory usage (RSS, available)
    - CPU percentage
    - Active connections (if tracked)
    - Queue depth (if tracked)

    Performance:
        - Throttled to every 30 seconds
        - Non-blocking collection via asyncio.to_thread
        - Minimal hot-path overhead
    """
    current_time = time.time()

    # Throttle resource collection (expensive operations)
    if current_time - self._last_resource_check < self._resource_check_interval:
        return

    try:
        # Use asyncio.to_thread for CPU-intensive psutil calls
        process = await asyncio.to_thread(psutil.Process, os.getpid())
        memory_info = await asyncio.to_thread(process.memory_info)
        cpu_percent = await asyncio.to_thread(process.cpu_percent, 0.1)

        async with self._metrics_lock:
            self._metrics["resources"]["memory_mb"] = memory_info.rss / 1024 / 1024
            self._metrics["resources"]["cpu_percent"] = cpu_percent

            # Track connections if available
            if hasattr(self, "_connection_pool"):
                pool = self._connection_pool
                self._metrics["resources"]["active_connections"] = (
                    pool.get_size() - pool.get_idle_size()
                    if hasattr(pool, "get_size")
                    else 0
                )

            # Track queue depth if available
            if hasattr(self, "_task_queue"):
                self._metrics["resources"]["queue_depth"] = self._task_queue.qsize()

        self._last_resource_check = current_time

    except Exception as e:
        logger.warning(f"Failed to collect resource metrics: {e}")


async def _get_resource_metrics(self) -> dict[str, float]:
    """
    Get current resource metrics snapshot.

    Returns:
        Dictionary of resource metrics
    """
    async with self._metrics_lock:
        return dict(self._metrics["resources"])'''


def generate_business_metrics_tracking() -> str:
    """
    Generate business/KPI metrics tracking code.

    Produces methods to track business-specific metrics:
    - Items processed counters
    - Throughput rate calculation
    - Custom KPI tracking
    - Aggregation over time windows

    Returns:
        Python code string for business metrics tracking methods
    """
    return '''async def _track_business_metric(
    self,
    metric_name: str,
    value: float,
    increment: bool = False,
) -> None:
    """
    Track business/KPI metrics.

    Args:
        metric_name: Name of the metric
        value: Metric value
        increment: If True, increment existing value; if False, set value

    Example:
        >>> await self._track_business_metric("items_processed", 1, increment=True)
        >>> await self._track_business_metric("average_quality_score", 0.95)
    """
    async with self._metrics_lock:
        if metric_name in ["items_processed"]:
            if increment:
                self._metrics["business"][metric_name] += value
            else:
                self._metrics["business"][metric_name] = value
        else:
            # Custom KPIs
            if increment:
                current = self._metrics["business"]["custom_kpis"].get(metric_name, 0)
                self._metrics["business"]["custom_kpis"][metric_name] = current + value
            else:
                self._metrics["business"]["custom_kpis"][metric_name] = value


async def _calculate_throughput(self) -> float:
    """
    Calculate current throughput rate (items/second).

    Returns:
        Throughput rate in items per second
    """
    async with self._metrics_lock:
        items = self._metrics["business"]["items_processed"]
        # Calculate based on total operation count and time
        total_ops = sum(
            m["count"] for m in self._metrics["operations"].values()
        )
        if total_ops == 0:
            return 0.0

        # Estimate based on operation rates
        avg_duration_ms = 0.0
        count = 0
        for metrics in self._metrics["operations"].values():
            if metrics["duration_ms"]:
                avg_duration_ms += statistics.mean(metrics["duration_ms"])
                count += 1

        if count > 0 and avg_duration_ms > 0:
            avg_duration_seconds = (avg_duration_ms / count) / 1000
            return 1.0 / avg_duration_seconds if avg_duration_seconds > 0 else 0.0

        return 0.0


async def _get_business_metrics(self) -> dict[str, Any]:
    """
    Get current business metrics snapshot.

    Returns:
        Dictionary of business metrics and KPIs
    """
    throughput = await self._calculate_throughput()

    async with self._metrics_lock:
        return {
            "items_processed": self._metrics["business"]["items_processed"],
            "throughput_per_second": throughput,
            "custom_kpis": dict(self._metrics["business"]["custom_kpis"]),
        }'''


def generate_metrics_publishing(
    service_name: str,
    kafka_topic: Optional[str] = None,
) -> str:
    """
    Generate metrics publishing code for Kafka events.

    Produces async method to publish metrics to Kafka using OnexEnvelopeV1 format.

    Args:
        service_name: Service name for event metadata
        kafka_topic: Kafka topic for metrics (default: "{service_name}.metrics.v1")

    Returns:
        Python code string for metrics publishing method
    """
    # Input validation
    if not service_name or not isinstance(service_name, str):
        raise ValueError(
            f"service_name must be a non-empty string, got: {service_name!r}. "
            f"Valid examples: 'orchestrator', 'reducer', 'metadata_stamping'"
        )

    if kafka_topic is not None:
        if not isinstance(kafka_topic, str) or not kafka_topic:
            raise ValueError(
                f"kafka_topic must be a non-empty string or None, got: {kafka_topic!r}. "
                f"Valid examples: 'metrics.v1', 'orchestrator.metrics.v1'"
            )

    if kafka_topic is None:
        kafka_topic = f"{service_name}.metrics.v1"

    return f'''async def _publish_metrics_event(
    self,
    operation: str,
    metrics: dict[str, Any],
) -> None:
    """
    Publish metrics event to Kafka.

    Publishes metrics using OnexEnvelopeV1 format to Kafka topic.

    Args:
        operation: Operation name
        metrics: Metrics data dictionary

    Performance:
        - Async/non-blocking
        - Fire-and-forget (no blocking on publish)
        - Error handling with fallback
    """
    try:
        # Calculate statistics
        stats = await self._calculate_operation_statistics(operation)

        # Create metrics event payload
        payload = {{
            "service_name": "{service_name}",
            "operation": operation,
            "timestamp": time.time(),
            "statistics": stats,
            "metrics": metrics,
        }}

        # Publish to Kafka (if available)
        if hasattr(self, "_kafka_producer") and self._kafka_producer:
            try:
                # Non-blocking publish
                await self._kafka_producer.send(
                    topic="{kafka_topic}",
                    value=payload,
                )
            except Exception as e:
                logger.warning(f"Failed to publish metrics to Kafka: {{e}}")

        # Log metrics for observability
        logger.info(
            f"Metrics [{{operation}}]: "
            f"count={{stats['count']}}, "
            f"avg={{stats['avg']:.2f}}ms, "
            f"p95={{stats.get('p95', 0):.2f}}ms, "
            f"errors={{stats['errors']}}"
        )

    except Exception as e:
        logger.error(f"Failed to publish metrics event: {{e}}")


async def _publish_all_metrics(self) -> None:
    """
    Publish all metrics (operations, resources, business) as a single event.

    Used for periodic comprehensive metrics reporting.
    """
    try:
        # Collect all metrics
        operations_stats = {{}}
        for operation in self._metrics["operations"].keys():
            operations_stats[operation] = await self._calculate_operation_statistics(operation)

        resource_metrics = await self._get_resource_metrics()
        business_metrics = await self._get_business_metrics()

        payload = {{
            "service_name": "{service_name}",
            "timestamp": time.time(),
            "operations": operations_stats,
            "resources": resource_metrics,
            "business": business_metrics,
        }}

        # Publish comprehensive metrics
        if hasattr(self, "_kafka_producer") and self._kafka_producer:
            await self._kafka_producer.send(
                topic="{kafka_topic}.comprehensive",
                value=payload,
            )

        logger.debug(f"Published comprehensive metrics for {service_name}")

    except Exception as e:
        logger.error(f"Failed to publish comprehensive metrics: {{e}}")'''


def generate_metrics_decorator() -> str:
    """
    Generate decorator for automatic metrics tracking.

    Produces decorator that wraps async methods to automatically track:
    - Execution duration
    - Success/failure status
    - Error details

    Returns:
        Python code string for metrics tracking decorator

    Example:
        >>> @track_metrics("orchestration")
        >>> async def orchestrate_workflow(self, request):
        >>>     # method implementation
    """
    return '''def track_metrics(operation_name: str):
    """
    Decorator to automatically track metrics for async methods.

    Args:
        operation_name: Name of operation for metrics tracking

    Usage:
        @track_metrics("orchestration")
        async def orchestrate_workflow(self, request):
            # method implementation

    Performance:
        - <1ms overhead
        - Automatic error tracking
        - Duration measurement
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()
            success = False
            error_details = None

            try:
                result = await func(self, *args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_details = f"{type(e).__name__}: {str(e)}"
                raise
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Track metrics (non-blocking)
                if hasattr(self, "_track_operation_metrics"):
                    asyncio.create_task(
                        self._track_operation_metrics(
                            operation_name,
                            duration_ms,
                            success,
                            error_details,
                        )
                    )

        return wrapper
    return decorator'''


def generate_complete_metrics_class(
    config: MetricsConfiguration,
) -> str:
    """
    Generate complete metrics tracking code as a mixable class.

    Produces a complete class with all metrics tracking functionality
    that can be mixed into node implementations.

    Args:
        config: Metrics configuration

    Returns:
        Python code string for complete metrics tracking class
    """
    init_code = generate_metrics_initialization(
        operations=config.operations,
        max_duration_samples=config.max_duration_samples,
        enable_resource_metrics=config.enable_resource_metrics,
        enable_business_metrics=config.enable_business_metrics,
        publish_interval_seconds=config.publish_interval_seconds,
        publish_interval_ops=config.publish_interval_ops,
    )

    operation_code = generate_operation_metrics_tracking(percentiles=config.percentiles)
    resource_code = generate_resource_metrics_collection()
    business_code = generate_business_metrics_tracking()
    publishing_code = generate_metrics_publishing(
        service_name=config.service_name,
        kafka_topic=f"{config.service_name}.metrics.v1",
    )
    decorator_code = generate_metrics_decorator()

    return f'''"""
Generated Metrics Tracking Mixin.

Auto-generated metrics tracking with:
- Operation metrics (duration, errors, counts)
- Resource metrics (CPU, memory, connections)
- Business metrics (throughput, KPIs)
- Periodic Kafka publishing

Generated for: {config.service_name}
Operations tracked: {", ".join(config.operations)}
"""

import asyncio
import functools
import logging
import os
import statistics
import time
from collections import deque
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


class MetricsTrackingMixin:
    """
    Metrics tracking mixin for ONEX nodes.

    Provides comprehensive metrics collection with:
    - <1ms overhead per operation
    - Efficient bounded queues (deque)
    - Periodic publishing to Kafka
    - Resource monitoring
    - Business KPI tracking
    """

    def _initialize_metrics(self) -> None:
        """Initialize metrics tracking data structures."""
{_indent(init_code, 2)}

{_indent(operation_code, 1)}

{_indent(resource_code, 1)}

{_indent(business_code, 1)}

{_indent(publishing_code, 1)}


# Decorator for automatic metrics tracking
{decorator_code}
'''


def _indent(code: str, levels: int) -> str:
    """Indent code block by specified levels (4 spaces per level)."""
    indent = "    " * levels
    return "\n".join(
        indent + line if line.strip() else line for line in code.split("\n")
    )


# === Example Usage ===


def generate_example_usage() -> str:
    """
    Generate example usage of metrics patterns.

    Returns:
        Python code string showing example usage
    """
    return '''# Example: Using metrics patterns in a node

from omnibase_core.nodes.node_effect import NodeEffect
from omninode_bridge.codegen.patterns.metrics import (
    MetricsConfiguration,
    generate_complete_metrics_class,
    generate_metrics_initialization,
)

# Generate metrics tracking code
config = MetricsConfiguration(
    operations=["orchestration", "validation", "transformation"],
    service_name="my_orchestrator",
    publish_interval_ops=100,
    publish_interval_seconds=60,
)

metrics_class_code = generate_complete_metrics_class(config)

# Or use individual generators for custom integration
class NodeMyOrchestrator(NodeEffect):
    """My orchestrator with metrics tracking."""

    def __init__(self, config: dict):
        super().__init__(config)

        # Initialize metrics (generated code)
        from collections import deque
        import asyncio
        import time

        self._metrics = {
            "operations": {
                "orchestration": {"count": 0, "duration_ms": deque(maxlen=1000), "errors": 0},
                "validation": {"count": 0, "duration_ms": deque(maxlen=1000), "errors": 0},
            },
        }
        self._metrics_lock = asyncio.Lock()
        self._last_publish = time.time()

    @track_metrics("orchestration")
    async def orchestrate(self, request):
        """Orchestrate workflow with automatic metrics tracking."""
        # Implementation here
        pass
'''


if __name__ == "__main__":
    # Example: Generate metrics patterns
    print("=== Metrics Pattern Generator ===\n")

    # Example 1: Initialization code
    print("1. Metrics Initialization:\n")
    init_code = generate_metrics_initialization(
        operations=["orchestration", "validation", "transformation"]
    )
    print(init_code)
    print("\n" + "=" * 80 + "\n")

    # Example 2: Complete metrics class
    print("2. Complete Metrics Class:\n")
    config = MetricsConfiguration(
        operations=["orchestration", "validation"],
        service_name="example_service",
    )
    complete_class = generate_complete_metrics_class(config)
    print(complete_class[:500] + "...\n(truncated)")
    print("\n" + "=" * 80 + "\n")

    # Example 3: Usage example
    print("3. Example Usage:\n")
    print(generate_example_usage())
