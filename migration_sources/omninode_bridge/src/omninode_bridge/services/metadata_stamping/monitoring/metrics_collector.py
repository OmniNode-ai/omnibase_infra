"""Comprehensive metrics collection for performance monitoring.

Implements omnibase_3 performance requirements:
- Database operations: <100ms
- Batch throughput: >50 items/second
- Event publishing: <50ms
- Hash generation: <2ms
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Optional

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class PerformanceGrade(Enum):
    """Performance grade system."""

    A = "A"  # Excellent performance (above threshold)
    B = "B"  # Good performance (within threshold)
    C = "C"  # Poor performance (below threshold)
    F = "F"  # Failed operation


class OperationType(Enum):
    """Types of operations to monitor."""

    DATABASE_QUERY = "database_query"
    DATABASE_INSERT = "database_insert"
    DATABASE_UPDATE = "database_update"
    DATABASE_BATCH = "database_batch"
    HASH_GENERATION = "hash_generation"
    BATCH_PROCESSING = "batch_processing"
    EVENT_PUBLISHING = "event_publishing"
    API_REQUEST = "api_request"
    HEALTH_CHECK = "health_check"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    operation_type: OperationType
    execution_time_ms: float
    timestamp: datetime
    success: bool
    error_type: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    grade: PerformanceGrade = PerformanceGrade.B

    def __post_init__(self):
        """Calculate performance grade based on thresholds."""
        self.grade = self._calculate_grade()

    def _calculate_grade(self) -> PerformanceGrade:
        """Calculate performance grade based on operation type and timing."""
        if not self.success:
            return PerformanceGrade.F

        # Performance thresholds based on omnibase_3 requirements
        thresholds = {
            OperationType.DATABASE_QUERY: {"A": 25, "B": 100},
            OperationType.DATABASE_INSERT: {"A": 50, "B": 100},
            OperationType.DATABASE_UPDATE: {"A": 50, "B": 100},
            OperationType.DATABASE_BATCH: {"A": 75, "B": 100},
            OperationType.HASH_GENERATION: {"A": 1, "B": 2},
            OperationType.BATCH_PROCESSING: {"A": 15, "B": 20},  # For 50+ items/sec
            OperationType.EVENT_PUBLISHING: {"A": 25, "B": 50},
            OperationType.API_REQUEST: {"A": 50, "B": 100},
            OperationType.HEALTH_CHECK: {"A": 10, "B": 25},
        }

        threshold = thresholds.get(self.operation_type, {"A": 50, "B": 100})

        if self.execution_time_ms <= threshold["A"]:
            return PerformanceGrade.A
        elif self.execution_time_ms <= threshold["B"]:
            return PerformanceGrade.B
        else:
            return PerformanceGrade.C


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""

    operation_type: OperationType
    count: int
    avg_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    success_rate: float
    throughput_per_second: float
    grade_distribution: dict[PerformanceGrade, int]
    last_updated: datetime


class MetricsCollector:
    """Comprehensive metrics collector with Prometheus integration."""

    def __init__(self, max_samples: int = 10000, sample_window_minutes: int = 60):
        """Initialize metrics collector.

        Args:
            max_samples: Maximum samples to keep in memory
            sample_window_minutes: Time window for calculating stats
        """
        self.max_samples = max_samples
        self.sample_window = timedelta(minutes=sample_window_minutes)

        # In-memory storage for recent metrics
        self.metrics: deque[PerformanceMetric] = deque(maxlen=max_samples)
        self.metrics_lock = Lock()

        # Prometheus metrics
        self._setup_prometheus_metrics()

        # Performance statistics cache
        self._stats_cache: dict[OperationType, PerformanceStats] = {}
        self._cache_lock = Lock()
        self._last_cache_update = datetime.now()

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        # Operation timing histograms
        self.operation_duration = Histogram(
            "metadata_stamping_operation_duration_seconds",
            "Time spent on operations",
            ["operation_type", "success"],
            buckets=(
                0.001,
                0.002,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
            ),
        )

        # Operation counters
        self.operation_total = Counter(
            "metadata_stamping_operations_total",
            "Total number of operations",
            ["operation_type", "grade", "success"],
        )

        # Current performance gauges
        self.current_throughput = Gauge(
            "metadata_stamping_throughput_per_second",
            "Current operations per second",
            ["operation_type"],
        )

        # Performance grade distribution
        self.grade_distribution = Gauge(
            "metadata_stamping_grade_distribution",
            "Distribution of performance grades",
            ["operation_type", "grade"],
        )

        # Resource utilization
        self.cpu_usage = Gauge(
            "metadata_stamping_cpu_usage_percent", "CPU usage percentage"
        )

        self.memory_usage = Gauge(
            "metadata_stamping_memory_usage_mb", "Memory usage in MB"
        )

        self.database_connections = Gauge(
            "metadata_stamping_database_connections",
            "Number of database connections",
            ["state"],  # active, idle, total
        )

    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric.

        Args:
            metric: Performance metric to record
        """
        with self.metrics_lock:
            self.metrics.append(metric)

        # Update Prometheus metrics
        self.operation_duration.labels(
            operation_type=metric.operation_type.value, success=str(metric.success)
        ).observe(metric.execution_time_ms / 1000)

        self.operation_total.labels(
            operation_type=metric.operation_type.value,
            grade=metric.grade.value,
            success=str(metric.success),
        ).inc()

    async def record_operation(
        self,
        operation_type: OperationType,
        operation_func,
        *args,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Record an operation with automatic timing.

        Args:
            operation_type: Type of operation
            operation_func: Function to execute and time
            *args: Function arguments
            metadata: Additional metadata
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        start_time = time.perf_counter()
        success = True
        error_type = None
        result = None

        try:
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
        except Exception as e:
            success = False
            error_type = type(e).__name__
            logger.error(f"Operation {operation_type.value} failed: {e}")
            raise
        finally:
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            metric = PerformanceMetric(
                operation_type=operation_type,
                execution_time_ms=execution_time_ms,
                timestamp=datetime.now(),
                success=success,
                error_type=error_type,
                metadata=metadata or {},
            )

            self.record_metric(metric)

        return result

    def get_recent_metrics(
        self,
        operation_type: Optional[OperationType] = None,
        time_window: Optional[timedelta] = None,
    ) -> list[PerformanceMetric]:
        """Get recent metrics within time window.

        Args:
            operation_type: Filter by operation type
            time_window: Time window (default: sample_window)

        Returns:
            List of recent metrics
        """
        time_window = time_window or self.sample_window
        cutoff_time = datetime.now() - time_window

        with self.metrics_lock:
            filtered_metrics = [
                metric
                for metric in self.metrics
                if metric.timestamp >= cutoff_time
                and (operation_type is None or metric.operation_type == operation_type)
            ]

        return filtered_metrics

    def calculate_performance_stats(
        self, operation_type: OperationType, force_refresh: bool = False
    ) -> Optional[PerformanceStats]:
        """Calculate performance statistics for operation type.

        Args:
            operation_type: Operation type to analyze
            force_refresh: Force cache refresh

        Returns:
            Performance statistics or None if no data
        """
        with self._cache_lock:
            # Check cache freshness
            cache_age = datetime.now() - self._last_cache_update
            if not force_refresh and cache_age < timedelta(minutes=1):
                return self._stats_cache.get(operation_type)

        # Get recent metrics
        recent_metrics = self.get_recent_metrics(operation_type)

        if not recent_metrics:
            return None

        # Calculate statistics
        execution_times = [m.execution_time_ms for m in recent_metrics]
        success_count = sum(1 for m in recent_metrics if m.success)

        # Calculate throughput (operations per second)
        time_span = (
            recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        ).total_seconds()
        throughput = len(recent_metrics) / max(time_span, 1)

        # Grade distribution
        grade_distribution = defaultdict(int)
        for metric in recent_metrics:
            grade_distribution[metric.grade] += 1

        # Create statistics
        stats = PerformanceStats(
            operation_type=operation_type,
            count=len(recent_metrics),
            avg_time_ms=statistics.mean(execution_times),
            p50_time_ms=statistics.median(execution_times),
            p95_time_ms=(
                statistics.quantiles(execution_times, n=20)[18]
                if len(execution_times) > 1
                else execution_times[0]
            ),
            p99_time_ms=(
                statistics.quantiles(execution_times, n=100)[98]
                if len(execution_times) > 1
                else execution_times[0]
            ),
            success_rate=success_count / len(recent_metrics),
            throughput_per_second=throughput,
            grade_distribution=dict(grade_distribution),
            last_updated=datetime.now(),
        )

        # Update cache
        with self._cache_lock:
            self._stats_cache[operation_type] = stats
            self._last_cache_update = datetime.now()

        # Update Prometheus gauges
        self.current_throughput.labels(operation_type=operation_type.value).set(
            throughput
        )

        for grade, count in grade_distribution.items():
            self.grade_distribution.labels(
                operation_type=operation_type.value, grade=grade.value
            ).set(count)

        return stats

    def get_all_performance_stats(
        self, force_refresh: bool = False
    ) -> dict[OperationType, PerformanceStats]:
        """Get performance statistics for all operation types.

        Args:
            force_refresh: Force cache refresh

        Returns:
            Dictionary of performance statistics
        """
        stats = {}

        for operation_type in OperationType:
            stat = self.calculate_performance_stats(operation_type, force_refresh)
            if stat:
                stats[operation_type] = stat

        return stats

    def get_performance_summary(self) -> dict[str, Any]:
        """Get overall performance summary.

        Returns:
            Performance summary with grades and alerts
        """
        all_stats = self.get_all_performance_stats()

        summary = {
            "overall_grade": PerformanceGrade.A.value,
            "operations": {},
            "alerts": [],
            "last_updated": datetime.now().isoformat(),
        }

        overall_grade = PerformanceGrade.A

        for operation_type, stats in all_stats.items():
            # Determine operation grade based on requirements
            operation_grade = self._determine_operation_grade(operation_type, stats)

            summary["operations"][operation_type.value] = {
                "grade": operation_grade.value,
                "avg_time_ms": round(stats.avg_time_ms, 2),
                "p95_time_ms": round(stats.p95_time_ms, 2),
                "p99_time_ms": round(stats.p99_time_ms, 2),
                "success_rate": round(stats.success_rate, 3),
                "throughput_per_second": round(stats.throughput_per_second, 2),
                "count": stats.count,
            }

            # Check for performance violations
            alerts = self._check_performance_alerts(operation_type, stats)
            summary["alerts"].extend(alerts)

            # Update overall grade
            if operation_grade.value < overall_grade.value:
                overall_grade = operation_grade

        summary["overall_grade"] = overall_grade.value

        return summary

    def _determine_operation_grade(
        self, operation_type: OperationType, stats: PerformanceStats
    ) -> PerformanceGrade:
        """Determine operation grade based on performance statistics.

        Args:
            operation_type: Operation type
            stats: Performance statistics

        Returns:
            Overall grade for the operation
        """
        # Grade distribution analysis
        total_operations = sum(stats.grade_distribution.values())
        if total_operations == 0:
            return PerformanceGrade.C

        a_grade_ratio = (
            stats.grade_distribution.get(PerformanceGrade.A, 0) / total_operations
        )
        b_grade_ratio = (
            stats.grade_distribution.get(PerformanceGrade.B, 0) / total_operations
        )
        c_grade_ratio = (
            stats.grade_distribution.get(PerformanceGrade.C, 0) / total_operations
        )
        f_grade_ratio = (
            stats.grade_distribution.get(PerformanceGrade.F, 0) / total_operations
        )

        # Grading criteria
        if f_grade_ratio > 0.05:  # More than 5% failures
            return PerformanceGrade.F
        elif c_grade_ratio > 0.2:  # More than 20% poor performance
            return PerformanceGrade.C
        elif a_grade_ratio >= 0.8:  # 80% or more excellent performance
            return PerformanceGrade.A
        else:
            return PerformanceGrade.B

    def _check_performance_alerts(
        self, operation_type: OperationType, stats: PerformanceStats
    ) -> list[dict[str, Any]]:
        """Check for performance alerts based on omnibase_3 requirements.

        Args:
            operation_type: Operation type
            stats: Performance statistics

        Returns:
            List of performance alerts
        """
        alerts = []

        # Define alert thresholds based on omnibase_3 requirements
        alert_thresholds = {
            OperationType.DATABASE_QUERY: {"p95": 100, "p99": 150},
            OperationType.DATABASE_INSERT: {"p95": 100, "p99": 150},
            OperationType.DATABASE_UPDATE: {"p95": 100, "p99": 150},
            OperationType.DATABASE_BATCH: {"p95": 100, "p99": 150},
            OperationType.HASH_GENERATION: {"p95": 2, "p99": 5},
            OperationType.BATCH_PROCESSING: {"throughput_min": 50},
            OperationType.EVENT_PUBLISHING: {"p95": 50, "p99": 100},
        }

        thresholds = alert_thresholds.get(operation_type, {})

        # Check latency alerts
        if "p95" in thresholds and stats.p95_time_ms > thresholds["p95"]:
            alerts.append(
                {
                    "type": "latency_violation",
                    "severity": "warning",
                    "operation": operation_type.value,
                    "message": f"P95 latency ({stats.p95_time_ms:.1f}ms) exceeds threshold ({thresholds['p95']}ms)",
                    "current_value": stats.p95_time_ms,
                    "threshold": thresholds["p95"],
                }
            )

        if "p99" in thresholds and stats.p99_time_ms > thresholds["p99"]:
            alerts.append(
                {
                    "type": "latency_violation",
                    "severity": "critical",
                    "operation": operation_type.value,
                    "message": f"P99 latency ({stats.p99_time_ms:.1f}ms) exceeds threshold ({thresholds['p99']}ms)",
                    "current_value": stats.p99_time_ms,
                    "threshold": thresholds["p99"],
                }
            )

        # Check throughput alerts
        if (
            "throughput_min" in thresholds
            and stats.throughput_per_second < thresholds["throughput_min"]
        ):
            alerts.append(
                {
                    "type": "throughput_violation",
                    "severity": "warning",
                    "operation": operation_type.value,
                    "message": f"Throughput ({stats.throughput_per_second:.1f}/s) below minimum ({thresholds['throughput_min']}/s)",
                    "current_value": stats.throughput_per_second,
                    "threshold": thresholds["throughput_min"],
                }
            )

        # Check success rate alerts
        if stats.success_rate < 0.95:  # Less than 95% success rate
            severity = "critical" if stats.success_rate < 0.9 else "warning"
            alerts.append(
                {
                    "type": "success_rate_violation",
                    "severity": severity,
                    "operation": operation_type.value,
                    "message": f"Success rate ({stats.success_rate:.1%}) below threshold (95%)",
                    "current_value": stats.success_rate,
                    "threshold": 0.95,
                }
            )

        return alerts

    def update_resource_metrics(
        self, cpu_percent: float, memory_mb: float, db_connections: dict[str, int]
    ):
        """Update resource utilization metrics.

        Args:
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            db_connections: Database connection counts by state
        """
        self.cpu_usage.set(cpu_percent)
        self.memory_usage.set(memory_mb)

        for state, count in db_connections.items():
            self.database_connections.labels(state=state).set(count)
