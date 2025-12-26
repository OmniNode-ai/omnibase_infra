# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Statistical utilities for performance testing.

This module provides statistically rigorous utilities for performance tests,
including proper warmup handling, multiple-run aggregation, confidence intervals,
and memory baseline tracking.

Statistical Best Practices:
    - Use at least 10 runs for reliable statistics (30+ for tight confidence)
    - Discard first 1-3 runs as warmup for JIT effects
    - Report median and percentiles rather than mean (more robust to outliers)
    - Use confidence intervals for threshold comparisons
    - Track memory baselines using tracemalloc for accuracy

Usage:
    >>> from tests.helpers.statistics_utils import (
    ...     PerformanceStats,
    ...     MemoryTracker,
    ...     run_with_warmup,
    ... )
    >>>
    >>> # Run multiple iterations with warmup
    >>> timings = await run_with_warmup(
    ...     operation=lambda: my_function(),
    ...     iterations=30,
    ...     warmup_iterations=3,
    ... )
    >>>
    >>> # Calculate statistics
    >>> stats = PerformanceStats.from_samples(timings)
    >>> assert stats.p95 < 5.0, f"P95 latency {stats.p95:.3f}s exceeds threshold"
    >>>
    >>> # Memory tracking
    >>> tracker = MemoryTracker()
    >>> tracker.start()
    >>> # ... do work ...
    >>> tracker.snapshot("after_batch_1")
    >>> growth = tracker.get_growth_mb("after_batch_1")

Related:
    - OMN-955: Chaos and replay testing
    - test_replay_performance.py: Performance test suite
"""

from __future__ import annotations

import gc
import math
import statistics
import time
import tracemalloc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")


# =============================================================================
# Performance Statistics
# =============================================================================


@dataclass(frozen=True)
class PerformanceStats:
    """Immutable container for performance statistics.

    Provides comprehensive statistical analysis of timing samples including
    central tendency, dispersion, and percentile measures.

    All time values are in seconds.

    Attributes:
        count: Number of samples.
        mean: Arithmetic mean of samples.
        median: Median (50th percentile) of samples.
        std_dev: Sample standard deviation.
        min_val: Minimum sample value.
        max_val: Maximum sample value.
        p50: 50th percentile (same as median).
        p75: 75th percentile.
        p90: 90th percentile.
        p95: 95th percentile.
        p99: 99th percentile.
        coefficient_of_variation: std_dev / mean (relative variability).
    """

    count: int
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    coefficient_of_variation: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> PerformanceStats:
        """Create statistics from a list of timing samples.

        Args:
            samples: List of timing values in seconds.

        Returns:
            PerformanceStats with calculated statistics.

        Raises:
            ValueError: If samples list is empty or has fewer than 2 items.
        """
        if len(samples) < 2:
            raise ValueError(
                f"At least 2 samples required for statistics, got {len(samples)}"
            )

        sorted_samples = sorted(samples)
        count = len(sorted_samples)
        mean = statistics.mean(sorted_samples)
        median = statistics.median(sorted_samples)
        std_dev = statistics.stdev(sorted_samples)

        def percentile(data: list[float], p: float) -> float:
            """Calculate percentile using linear interpolation."""
            k = (len(data) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            return data[int(f)] * (c - k) + data[int(c)] * (k - f)

        return cls(
            count=count,
            mean=mean,
            median=median,
            std_dev=std_dev,
            min_val=sorted_samples[0],
            max_val=sorted_samples[-1],
            p50=percentile(sorted_samples, 50),
            p75=percentile(sorted_samples, 75),
            p90=percentile(sorted_samples, 90),
            p95=percentile(sorted_samples, 95),
            p99=percentile(sorted_samples, 99),
            coefficient_of_variation=std_dev / mean if mean > 0 else 0.0,
        )

    def format_report(self, name: str = "Performance") -> str:
        """Format a human-readable performance report.

        Args:
            name: Name to include in report header.

        Returns:
            Formatted multi-line string with statistics.
        """
        return (
            f"[{name}] {self.count} samples:\n"
            f"  Mean: {self.mean:.4f}s | Median: {self.median:.4f}s\n"
            f"  StdDev: {self.std_dev:.4f}s | CV: {self.coefficient_of_variation:.2%}\n"
            f"  Min: {self.min_val:.4f}s | Max: {self.max_val:.4f}s\n"
            f"  P50: {self.p50:.4f}s | P90: {self.p90:.4f}s | "
            f"P95: {self.p95:.4f}s | P99: {self.p99:.4f}s"
        )


# =============================================================================
# Memory Tracking
# =============================================================================


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage at a point in time.

    Attributes:
        name: Identifier for this snapshot.
        current_bytes: Current memory usage in bytes.
        peak_bytes: Peak memory usage since last reset.
        timestamp: When snapshot was taken.
    """

    name: str
    current_bytes: int
    peak_bytes: int
    timestamp: float


@dataclass
class MemoryTracker:
    """Track memory usage during test execution.

    Uses tracemalloc for accurate memory measurement rather than
    sys.getsizeof() which only measures shallow object size.

    Thread Safety:
        This tracker is NOT thread-safe. Use separate trackers per thread
        or external synchronization for multi-threaded usage.

    Usage:
        >>> tracker = MemoryTracker()
        >>> tracker.start()
        >>> # ... allocate memory ...
        >>> tracker.snapshot("after_allocation")
        >>> growth_mb = tracker.get_growth_mb("after_allocation")
        >>> tracker.stop()
    """

    baseline_bytes: int = 0
    peak_bytes: int = 0
    snapshots: dict[str, MemorySnapshot] = field(default_factory=dict)
    _started: bool = False

    def start(self) -> None:
        """Start memory tracking.

        Forces garbage collection and records baseline memory.
        """
        if self._started:
            return

        # Force GC before baseline
        gc.collect()
        gc.collect()  # Second pass for cyclic references

        tracemalloc.start()
        self._started = True

        # Record baseline
        current, peak = tracemalloc.get_traced_memory()
        self.baseline_bytes = current
        self.peak_bytes = peak

    def stop(self) -> None:
        """Stop memory tracking and clean up."""
        if self._started:
            tracemalloc.stop()
            self._started = False

    def snapshot(self, name: str) -> MemorySnapshot:
        """Take a memory snapshot.

        Args:
            name: Identifier for this snapshot.

        Returns:
            MemorySnapshot with current memory state.

        Raises:
            RuntimeError: If tracking not started.
        """
        if not self._started:
            raise RuntimeError("Memory tracking not started. Call start() first.")

        # Force GC before snapshot for accuracy
        gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        snapshot = MemorySnapshot(
            name=name,
            current_bytes=current,
            peak_bytes=peak,
            timestamp=time.perf_counter(),
        )
        self.snapshots[name] = snapshot
        self.peak_bytes = max(self.peak_bytes, peak)
        return snapshot

    def get_growth_mb(self, snapshot_name: str | None = None) -> float:
        """Get memory growth in megabytes from baseline.

        Args:
            snapshot_name: Snapshot to compare against baseline.
                          If None, uses current memory.

        Returns:
            Memory growth in MB (can be negative if memory was freed).

        Raises:
            KeyError: If snapshot_name not found.
            RuntimeError: If tracking not started.
        """
        if snapshot_name:
            if snapshot_name not in self.snapshots:
                raise KeyError(f"Snapshot '{snapshot_name}' not found")
            current_bytes = self.snapshots[snapshot_name].current_bytes
        else:
            if not self._started:
                raise RuntimeError("Memory tracking not started. Call start() first.")
            gc.collect()
            current_bytes, _ = tracemalloc.get_traced_memory()

        growth_bytes = current_bytes - self.baseline_bytes
        return growth_bytes / (1024 * 1024)

    def get_peak_mb(self) -> float:
        """Get peak memory usage in megabytes.

        Returns:
            Peak memory in MB since tracking started.
        """
        if self._started:
            _, peak = tracemalloc.get_traced_memory()
            self.peak_bytes = max(self.peak_bytes, peak)
        return self.peak_bytes / (1024 * 1024)

    def format_report(self) -> str:
        """Format a memory usage report.

        Returns:
            Formatted string with memory statistics.
        """
        baseline_mb = self.baseline_bytes / (1024 * 1024)
        peak_mb = self.get_peak_mb()
        current_growth = self.get_growth_mb() if self._started else 0.0

        lines = [
            "[Memory Report]",
            f"  Baseline: {baseline_mb:.3f} MB",
            f"  Peak: {peak_mb:.3f} MB",
            f"  Current Growth: {current_growth:.3f} MB",
        ]

        if self.snapshots:
            lines.append("  Snapshots:")
            for name, snapshot in self.snapshots.items():
                snap_mb = snapshot.current_bytes / (1024 * 1024)
                growth = (snapshot.current_bytes - self.baseline_bytes) / (1024 * 1024)
                lines.append(f"    {name}: {snap_mb:.3f} MB (growth: {growth:+.3f} MB)")

        return "\n".join(lines)


# =============================================================================
# Statistical Tolerance Calculation
# =============================================================================


@dataclass(frozen=True)
class BinomialConfidenceInterval:
    """Confidence interval for binomial proportion.

    Attributes:
        lower: Lower bound of confidence interval.
        upper: Upper bound of confidence interval.
        confidence_level: Confidence level (e.g., 0.95 for 95%).
        sample_size: Number of trials.
        observed_rate: Observed success/failure rate.
    """

    lower: float
    upper: float
    confidence_level: float
    sample_size: int
    observed_rate: float


def calculate_binomial_confidence_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> BinomialConfidenceInterval:
    """Calculate confidence interval for binomial proportion.

    Uses the Wilson score interval which is more accurate than the normal
    approximation, especially for extreme proportions or small samples.

    Args:
        successes: Number of successes (or failures, depending on
            what you're measuring).
        trials: Total number of trials.
        confidence_level: Confidence level (0.0 to 1.0, default 0.95).

    Returns:
        BinomialConfidenceInterval with bounds.

    Raises:
        ValueError: If trials <= 0 or successes < 0 or successes > trials.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if successes < 0:
        raise ValueError(f"successes must be non-negative, got {successes}")
    if successes > trials:
        raise ValueError(f"successes ({successes}) cannot exceed trials ({trials})")

    # Wilson score interval
    p_hat = successes / trials
    z = _normal_quantile((1 + confidence_level) / 2)
    z2 = z * z
    n = trials

    denominator = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denominator
    spread = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return BinomialConfidenceInterval(
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        sample_size=trials,
        observed_rate=p_hat,
    )


def _normal_quantile(p: float) -> float:
    """Approximate quantile function for standard normal distribution.

    Uses Abramowitz and Stegun approximation (26.2.23).

    Args:
        p: Probability (0 < p < 1).

    Returns:
        Approximate z-score for probability p.
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Coefficients for rational approximation
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        # Lower region
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    elif p <= p_high:
        # Central region
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    else:
        # Upper region
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )


def minimum_sample_size_for_tolerance(
    expected_rate: float,
    tolerance: float,
    confidence_level: float = 0.95,
) -> int:
    """Calculate minimum sample size needed for given tolerance.

    Uses the formula derived from binomial confidence interval width.

    Args:
        expected_rate: Expected proportion (0.0 to 1.0).
        tolerance: Acceptable deviation from expected rate (e.g., 0.1 for 10%).
        confidence_level: Desired confidence level.

    Returns:
        Minimum number of samples needed.

    Raises:
        ValueError: If parameters are out of valid ranges.
    """
    if not 0.0 <= expected_rate <= 1.0:
        raise ValueError(f"expected_rate must be in [0, 1], got {expected_rate}")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    z = _normal_quantile((1 + confidence_level) / 2)

    # For binomial: margin of error = z * sqrt(p(1-p)/n)
    # Solving for n: n = z^2 * p(1-p) / margin^2
    # margin = tolerance * expected_rate
    margin = tolerance * max(expected_rate, 0.01)  # Avoid division by zero
    variance = expected_rate * (1 - expected_rate)

    if variance == 0:
        # Edge case: p=0 or p=1, variance is 0
        return 10  # Minimum practical sample size

    n = (z * z * variance) / (margin * margin)
    return max(10, math.ceil(n))  # At least 10 samples


# =============================================================================
# Execution Utilities
# =============================================================================


async def run_with_warmup(
    operation: Callable[[], Awaitable[T] | T],
    iterations: int = 30,
    warmup_iterations: int = 3,
) -> list[float]:
    """Run an operation multiple times with warmup, collecting timing data.

    Discards warmup iterations to eliminate JIT compilation and cache effects.
    Uses time.perf_counter() for high-resolution timing.

    Args:
        operation: Callable to time (can be sync or async).
        iterations: Number of timed iterations (after warmup).
        warmup_iterations: Number of warmup iterations (discarded).

    Returns:
        List of timing values in seconds for non-warmup iterations.

    Raises:
        ValueError: If iterations < 1.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")

    timings: list[float] = []

    for i in range(warmup_iterations + iterations):
        start = time.perf_counter()

        result = operation()
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]

        elapsed = time.perf_counter() - start

        # Only record after warmup
        if i >= warmup_iterations:
            timings.append(elapsed)

    return timings


def run_with_warmup_sync(
    operation: Callable[[], object],
    iterations: int = 30,
    warmup_iterations: int = 3,
) -> list[float]:
    """Synchronous version of run_with_warmup.

    Args:
        operation: Synchronous callable to time.
        iterations: Number of timed iterations (after warmup).
        warmup_iterations: Number of warmup iterations (discarded).

    Returns:
        List of timing values in seconds for non-warmup iterations.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")

    timings: list[float] = []

    for i in range(warmup_iterations + iterations):
        start = time.perf_counter()
        operation()
        elapsed = time.perf_counter() - start

        if i >= warmup_iterations:
            timings.append(elapsed)

    return timings


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Statistics
    "PerformanceStats",
    # Memory tracking
    "MemorySnapshot",
    "MemoryTracker",
    # Binomial confidence intervals
    "BinomialConfidenceInterval",
    "calculate_binomial_confidence_interval",
    "minimum_sample_size_for_tolerance",
    # Execution utilities
    "run_with_warmup",
    "run_with_warmup_sync",
]
