# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Performance timing utilities for E2E registration tests.

This module provides utilities for measuring and asserting performance thresholds
in the ONEX 2-way registration pattern E2E tests (OMN-892).

Performance Thresholds (calibrated for remote infrastructure at 192.168.86.200):
    - Introspection broadcast latency: <200ms (includes network round-trip)
    - Registry processing latency: <300ms (network + database operations)
    - Dual registration time: <1000ms (multiple network calls)
    - Heartbeat overhead: <150ms (single network call)
    - Heartbeat interval: 30s (+/- 5s tolerance)

Note:
    Original OMN-892 requirements assumed local infrastructure with minimal latency.
    These thresholds account for production-like distributed deployments where
    network round-trip adds ~20-50ms per operation.

Example:
    Basic timing assertion:

    >>> async with timed_operation("introspection_broadcast", threshold_ms=200) as timing:
    ...     await node.broadcast_introspection()
    >>> timing.assert_passed()

    Using the performance collector:

    >>> collector = PerformanceCollector()
    >>> async with collector.time("op1", threshold_ms=200):
    ...     await do_op1()
    >>> async with collector.time("op2", threshold_ms=300):
    ...     await do_op2()
    >>> collector.assert_all_passed()
    >>> collector.print_summary()
"""

from __future__ import annotations

import statistics
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from omnibase_infra.models.registration import ModelNodeHeartbeatEvent

# Type variables for generic functions
T = TypeVar("T")
P = ParamSpec("P")


class PerformanceThresholds:
    """Performance thresholds for ONEX 2-way registration pattern (OMN-892).

    These thresholds define the maximum acceptable latencies for various
    operations in the ONEX 2-way registration pattern.

    Note:
        These thresholds are calibrated for **remote infrastructure** at 192.168.86.200.
        Network round-trip latency (~20-50ms) is factored into each threshold.
        Original OMN-892 requirements assumed local infrastructure; these values
        provide realistic margins for production-like distributed deployments.

    Attributes:
        INTROSPECTION_BROADCAST_MS: Max latency for introspection broadcast (200ms).
            Includes network round-trip to Kafka broker.
        REGISTRY_PROCESSING_MS: Max latency for registry to process registration (300ms).
            Includes network latency + database operations.
        DUAL_REGISTRATION_MS: Max total time for dual registration flow (1000ms).
            Multiple network calls: introspection broadcast + registry processing + DB writes.
        HEARTBEAT_OVERHEAD_MS: Max overhead for heartbeat emission (150ms).
            Single network call with serialization overhead.
        HEARTBEAT_INTERVAL_SECONDS: Expected interval between heartbeats (30s).
        HEARTBEAT_TOLERANCE_SECONDS: Allowed deviation from expected interval (5s).
    """

    INTROSPECTION_BROADCAST_MS: float = 200.0
    REGISTRY_PROCESSING_MS: float = 300.0
    DUAL_REGISTRATION_MS: float = 1000.0
    HEARTBEAT_OVERHEAD_MS: float = 150.0
    HEARTBEAT_INTERVAL_SECONDS: float = 30.0
    HEARTBEAT_TOLERANCE_SECONDS: float = 5.0


@dataclass
class TimingResult:
    """Result of a timed operation.

    Attributes:
        operation: Name of the timed operation.
        elapsed_ms: Actual elapsed time in milliseconds.
        threshold_ms: Optional threshold in milliseconds.
        start_time: Operation start timestamp (monotonic).
        end_time: Operation end timestamp (monotonic).

    Example:
        >>> result = TimingResult(
        ...     operation="introspection",
        ...     elapsed_ms=35.5,
        ...     threshold_ms=50.0,
        ... )
        >>> result.passed
        True
        >>> result.assert_passed()  # No error raised
    """

    operation: str
    elapsed_ms: float
    threshold_ms: float | None = None
    start_time: float = field(default=0.0, repr=False)
    end_time: float = field(default=0.0, repr=False)

    @property
    def passed(self) -> bool:
        """Check if elapsed time is under threshold.

        Returns:
            True if no threshold set or elapsed time is under threshold.
        """
        if self.threshold_ms is None:
            return True
        return self.elapsed_ms < self.threshold_ms

    @property
    def margin_ms(self) -> float | None:
        """Get margin between elapsed time and threshold.

        Returns:
            Positive value if under threshold (margin remaining),
            negative value if over threshold (amount exceeded),
            None if no threshold set.
        """
        if self.threshold_ms is None:
            return None
        return self.threshold_ms - self.elapsed_ms

    def assert_passed(self) -> None:
        """Assert that the threshold was not exceeded.

        Raises:
            AssertionError: If elapsed time exceeds threshold, with detailed message
                showing operation name, actual time, threshold, and amount exceeded.
        """
        if not self.passed:
            exceeded_by = self.elapsed_ms - (self.threshold_ms or 0)
            raise AssertionError(
                f"Performance threshold exceeded for '{self.operation}': "
                f"{self.elapsed_ms:.2f}ms > {self.threshold_ms}ms "
                f"(exceeded by {exceeded_by:.2f}ms)"
            )

    def __str__(self) -> str:
        """Format timing result for display."""
        status = "PASS" if self.passed else "FAIL"
        threshold_str = f"/{self.threshold_ms:.1f}ms" if self.threshold_ms else ""
        return f"[{status}] {self.operation}: {self.elapsed_ms:.2f}ms{threshold_str}"


@asynccontextmanager
async def timed_operation(
    operation: str,
    threshold_ms: float | None = None,
) -> AsyncGenerator[TimingResult, None]:
    """Context manager that times an async operation.

    Creates a TimingResult that is populated with timing data when the
    context exits. The result object is yielded before the operation runs,
    allowing post-operation assertions.

    Args:
        operation: Name of the operation being timed (for error messages).
        threshold_ms: Optional threshold in milliseconds. If provided,
            assert_passed() will check against this value.

    Yields:
        TimingResult with elapsed_ms populated after context exits.

    Example:
        >>> async with timed_operation("introspection_broadcast", threshold_ms=50) as timing:
        ...     await node.broadcast_introspection()
        >>> print(f"Took {timing.elapsed_ms:.2f}ms")
        >>> timing.assert_passed()

    Note:
        The timing uses time.perf_counter() for high-resolution measurement.
        The result object's elapsed_ms is 0.0 until the context exits.
    """
    # Create result with placeholder values
    result = TimingResult(
        operation=operation,
        elapsed_ms=0.0,
        threshold_ms=threshold_ms,
    )

    start = time.perf_counter()
    result.start_time = start

    try:
        yield result
    finally:
        end = time.perf_counter()
        result.end_time = end
        # Update elapsed time - convert to milliseconds
        # Using object.__setattr__ because dataclass might be frozen in future
        object.__setattr__(result, "elapsed_ms", (end - start) * 1000)


def assert_performance(
    threshold_ms: float,
    operation: str | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator that asserts async function completes within threshold.

    Wraps an async function to measure execution time and raise AssertionError
    if the threshold is exceeded.

    Args:
        threshold_ms: Maximum allowed execution time in milliseconds.
        operation: Optional operation name for error messages.
            Defaults to the decorated function's name.

    Returns:
        Decorator function that wraps async functions with timing assertion.

    Example:
        >>> @assert_performance(threshold_ms=50, operation="introspection")
        ... async def test_introspection_speed():
        ...     await node.broadcast()

        >>> @assert_performance(threshold_ms=100)  # Uses function name
        ... async def test_registry_processing():
        ...     await registry.process()

    Raises:
        AssertionError: If decorated function exceeds threshold.
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        op_name = operation or func.__name__

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with timed_operation(op_name, threshold_ms) as timing:
                result = await func(*args, **kwargs)
            timing.assert_passed()
            return result

        return wrapper

    return decorator


class PerformanceCollector:
    """Collects timing results across multiple operations.

    Provides a way to measure multiple operations and aggregate results
    for summary reporting and batch assertions.

    Attributes:
        results: List of TimingResult objects collected.

    Example:
        >>> collector = PerformanceCollector()
        >>> async with collector.time("introspection", threshold_ms=50):
        ...     await node.broadcast_introspection()
        >>> async with collector.time("registry_process", threshold_ms=100):
        ...     await registry.process()
        >>> collector.assert_all_passed()
        >>> collector.print_summary()

        Output:
            Performance Summary
            ===================
            [PASS] introspection: 35.50ms/50.0ms
            [PASS] registry_process: 78.20ms/100.0ms

            Total operations: 2
            Passed: 2, Failed: 0
            Total time: 113.70ms
    """

    def __init__(self) -> None:
        """Initialize empty collector."""
        self.results: list[TimingResult] = []

    @asynccontextmanager
    async def time(
        self,
        operation: str,
        threshold_ms: float | None = None,
    ) -> AsyncGenerator[TimingResult, None]:
        """Time an operation and collect the result.

        Args:
            operation: Name of the operation being timed.
            threshold_ms: Optional threshold in milliseconds.

        Yields:
            TimingResult that is automatically added to results list.

        Example:
            >>> async with collector.time("my_operation", threshold_ms=50) as timing:
            ...     await do_operation()
            >>> # timing is now in collector.results
        """
        async with timed_operation(operation, threshold_ms) as result:
            yield result
        self.results.append(result)

    def add_result(self, result: TimingResult) -> None:
        """Manually add a timing result.

        Args:
            result: TimingResult to add to collection.
        """
        self.results.append(result)

    def clear(self) -> None:
        """Clear all collected results."""
        self.results.clear()

    @property
    def all_passed(self) -> bool:
        """Check if all operations passed their thresholds.

        Returns:
            True if all results passed or have no threshold.
        """
        return all(r.passed for r in self.results)

    @property
    def failed_results(self) -> list[TimingResult]:
        """Get list of results that failed their thresholds.

        Returns:
            List of TimingResult objects that exceeded their threshold.
        """
        return [r for r in self.results if not r.passed]

    @property
    def passed_results(self) -> list[TimingResult]:
        """Get list of results that passed their thresholds.

        Returns:
            List of TimingResult objects within their threshold.
        """
        return [r for r in self.results if r.passed]

    def assert_all_passed(self) -> None:
        """Assert all timed operations passed their thresholds.

        Raises:
            AssertionError: If any operation exceeded its threshold, with
                detailed message listing all failures.
        """
        failed = self.failed_results
        if failed:
            failure_details = "\n".join(f"  - {r!s}" for r in failed)
            raise AssertionError(
                f"{len(failed)} operation(s) exceeded performance threshold:\n"
                f"{failure_details}"
            )

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for all collected results.

        Returns:
            Dictionary containing:
                - total_operations: Number of operations timed
                - passed_count: Number that passed threshold
                - failed_count: Number that exceeded threshold
                - total_time_ms: Sum of all elapsed times
                - min_time_ms: Minimum elapsed time
                - max_time_ms: Maximum elapsed time
                - avg_time_ms: Average elapsed time
                - results: List of result dictionaries
        """
        if not self.results:
            return {
                "total_operations": 0,
                "passed_count": 0,
                "failed_count": 0,
                "total_time_ms": 0.0,
                "min_time_ms": 0.0,
                "max_time_ms": 0.0,
                "avg_time_ms": 0.0,
                "results": [],
            }

        elapsed_times = [r.elapsed_ms for r in self.results]
        return {
            "total_operations": len(self.results),
            "passed_count": len(self.passed_results),
            "failed_count": len(self.failed_results),
            "total_time_ms": sum(elapsed_times),
            "min_time_ms": min(elapsed_times),
            "max_time_ms": max(elapsed_times),
            "avg_time_ms": statistics.mean(elapsed_times),
            "results": [
                {
                    "operation": r.operation,
                    "elapsed_ms": r.elapsed_ms,
                    "threshold_ms": r.threshold_ms,
                    "passed": r.passed,
                }
                for r in self.results
            ],
        }

    def print_summary(self) -> None:
        """Print formatted summary to stdout."""
        summary = self.get_summary()

        print("\nPerformance Summary")
        print("=" * 40)

        for result in self.results:
            print(str(result))

        print()
        print(f"Total operations: {summary['total_operations']}")
        print(f"Passed: {summary['passed_count']}, Failed: {summary['failed_count']}")
        print(f"Total time: {summary['total_time_ms']:.2f}ms")

        if summary["total_operations"] > 1:
            print(
                f"Range: {summary['min_time_ms']:.2f}ms - {summary['max_time_ms']:.2f}ms"
            )
            print(f"Average: {summary['avg_time_ms']:.2f}ms")


async def measure_latency[T](
    operation: Callable[[], Awaitable[T]],
    iterations: int = 1,
) -> tuple[T, float]:
    """Measure latency of an async operation.

    Executes the operation the specified number of times and returns
    the result from the last iteration along with the average latency.

    Args:
        operation: Async callable to measure (no arguments).
        iterations: Number of times to execute operation. Default 1.

    Returns:
        Tuple of (last_result, average_latency_ms).

    Example:
        >>> result, latency = await measure_latency(
        ...     lambda: node.broadcast_introspection(),
        ...     iterations=5,
        ... )
        >>> print(f"Result: {result}, Avg latency: {latency:.2f}ms")

    Raises:
        ValueError: If iterations < 1.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    total_time = 0.0
    result: T | None = None

    for _ in range(iterations):
        start = time.perf_counter()
        result = await operation()
        end = time.perf_counter()
        total_time += (end - start) * 1000

    avg_latency = total_time / iterations
    # result is guaranteed to be set since iterations >= 1
    return result, avg_latency  # type: ignore[return-value]


async def measure_latency_percentiles[T](
    operation: Callable[[], Awaitable[T]],
    iterations: int = 10,
) -> dict[str, float]:
    """Measure latency percentiles of an async operation.

    Executes the operation multiple times and calculates percentile statistics.
    Useful for understanding latency distribution and tail latencies.

    Args:
        operation: Async callable to measure (no arguments).
        iterations: Number of iterations. Default 10. Minimum 1.

    Returns:
        Dictionary with percentile values in milliseconds:
            - min: Minimum latency
            - p50: 50th percentile (median)
            - p95: 95th percentile
            - p99: 99th percentile
            - max: Maximum latency
            - avg: Average latency
            - std_dev: Standard deviation (0.0 if iterations < 2)
            - iterations: Number of iterations performed

    Example:
        >>> stats = await measure_latency_percentiles(
        ...     lambda: node.broadcast_introspection(),
        ...     iterations=100,
        ... )
        >>> print(f"p50: {stats['p50']:.2f}ms, p99: {stats['p99']:.2f}ms")

    Raises:
        ValueError: If iterations < 1.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    latencies: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter()
        await operation()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies.sort()

    def percentile(data: list[float], p: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (data[c] - data[f]) * (k - f)

    return {
        "min": min(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "max": max(latencies),
        "avg": statistics.mean(latencies),
        "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "iterations": iterations,
    }


def verify_heartbeat_interval(
    events: list[ModelNodeHeartbeatEvent],
    expected_interval_seconds: float = PerformanceThresholds.HEARTBEAT_INTERVAL_SECONDS,
    tolerance_seconds: float = PerformanceThresholds.HEARTBEAT_TOLERANCE_SECONDS,
) -> bool:
    """Verify heartbeat events are spaced correctly.

    Checks that the intervals between consecutive heartbeat events
    are within the expected tolerance.

    Args:
        events: List of heartbeat events to verify (must be sorted by timestamp).
        expected_interval_seconds: Expected interval between heartbeats.
            Default is PerformanceThresholds.HEARTBEAT_INTERVAL_SECONDS (30s).
        tolerance_seconds: Allowed deviation from expected interval.
            Default is PerformanceThresholds.HEARTBEAT_TOLERANCE_SECONDS (5s).

    Returns:
        True if all intervals are within tolerance of expected interval.
        Returns True if fewer than 2 events (no interval to check).

    Example:
        >>> events = [heartbeat1, heartbeat2, heartbeat3]  # 30s apart
        >>> verify_heartbeat_interval(events)
        True
        >>> verify_heartbeat_interval(events, expected_interval_seconds=60)
        False  # Intervals are 30s, expected 60s

    Note:
        Events must have timestamps. The function compares consecutive
        event timestamps to determine intervals.
    """
    if len(events) < 2:
        return True

    min_interval = expected_interval_seconds - tolerance_seconds
    max_interval = expected_interval_seconds + tolerance_seconds

    for i in range(1, len(events)):
        prev_time = events[i - 1].timestamp
        curr_time = events[i].timestamp
        interval = (curr_time - prev_time).total_seconds()

        if not (min_interval <= interval <= max_interval):
            return False

    return True


def calculate_heartbeat_stats(
    events: list[ModelNodeHeartbeatEvent],
) -> dict[str, float]:
    """Calculate heartbeat interval statistics.

    Analyzes the intervals between consecutive heartbeat events
    and returns statistical measures.

    Args:
        events: List of heartbeat events (must be sorted by timestamp).

    Returns:
        Dictionary with interval statistics:
            - min_interval_s: Minimum interval between heartbeats
            - max_interval_s: Maximum interval between heartbeats
            - avg_interval_s: Average interval between heartbeats
            - std_dev_s: Standard deviation of intervals
            - count: Number of intervals analyzed

        All values are 0.0 if fewer than 2 events.

    Example:
        >>> stats = calculate_heartbeat_stats(heartbeat_events)
        >>> print(f"Avg interval: {stats['avg_interval_s']:.1f}s")
        >>> print(f"Interval range: {stats['min_interval_s']:.1f}s - {stats['max_interval_s']:.1f}s")
    """
    if len(events) < 2:
        return {
            "min_interval_s": 0.0,
            "max_interval_s": 0.0,
            "avg_interval_s": 0.0,
            "std_dev_s": 0.0,
            "count": 0,
        }

    intervals: list[float] = []
    for i in range(1, len(events)):
        prev_time = events[i - 1].timestamp
        curr_time = events[i].timestamp
        intervals.append((curr_time - prev_time).total_seconds())

    return {
        "min_interval_s": min(intervals),
        "max_interval_s": max(intervals),
        "avg_interval_s": statistics.mean(intervals),
        "std_dev_s": statistics.stdev(intervals) if len(intervals) > 1 else 0.0,
        "count": len(intervals),
    }


def assert_heartbeat_interval(
    events: list[ModelNodeHeartbeatEvent],
    expected_interval_seconds: float = PerformanceThresholds.HEARTBEAT_INTERVAL_SECONDS,
    tolerance_seconds: float = PerformanceThresholds.HEARTBEAT_TOLERANCE_SECONDS,
) -> None:
    """Assert heartbeat intervals are within expected tolerance.

    Combines verify_heartbeat_interval with assertion for cleaner test code.

    Args:
        events: List of heartbeat events to verify.
        expected_interval_seconds: Expected interval between heartbeats.
        tolerance_seconds: Allowed deviation from expected interval.

    Raises:
        AssertionError: If any interval is outside tolerance, with details
            about the actual vs expected intervals.

    Example:
        >>> assert_heartbeat_interval(heartbeat_events)  # Uses defaults
        >>> assert_heartbeat_interval(
        ...     heartbeat_events,
        ...     expected_interval_seconds=60,
        ...     tolerance_seconds=10,
        ... )
    """
    if len(events) < 2:
        return

    stats = calculate_heartbeat_stats(events)
    min_expected = expected_interval_seconds - tolerance_seconds
    max_expected = expected_interval_seconds + tolerance_seconds

    if not verify_heartbeat_interval(
        events, expected_interval_seconds, tolerance_seconds
    ):
        raise AssertionError(
            f"Heartbeat interval out of tolerance:\n"
            f"  Expected: {expected_interval_seconds:.1f}s +/- {tolerance_seconds:.1f}s "
            f"({min_expected:.1f}s - {max_expected:.1f}s)\n"
            f"  Actual range: {stats['min_interval_s']:.2f}s - {stats['max_interval_s']:.2f}s\n"
            f"  Actual avg: {stats['avg_interval_s']:.2f}s\n"
            f"  Intervals checked: {stats['count']}"
        )


__all__ = [
    # Core classes
    "PerformanceCollector",
    "PerformanceThresholds",
    "TimingResult",
    # Context managers and decorators
    "assert_performance",
    "timed_operation",
    # Latency measurement
    "measure_latency",
    "measure_latency_percentiles",
    # Heartbeat verification
    "assert_heartbeat_interval",
    "calculate_heartbeat_stats",
    "verify_heartbeat_interval",
]
