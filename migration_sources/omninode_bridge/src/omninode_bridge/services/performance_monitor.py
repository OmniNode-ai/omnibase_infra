"""Comprehensive performance monitoring and resilience dashboard for OmniNode Bridge."""

import asyncio
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram, generate_latest

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    timestamp: datetime
    component: str
    operation: str
    duration_ms: float
    success: bool
    error_type: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ResilienceMetrics:
    """Resilience pattern metrics."""

    circuit_breaker_trips: int = 0
    timeout_failures: int = 0
    retry_attempts: int = 0
    dead_letter_queue_messages: int = 0
    grace_period_recoveries: int = 0


@dataclass
class SystemHealth:
    """Overall system health indicators."""

    status: str  # healthy, degraded, critical
    uptime_seconds: float
    error_rate_percent: float
    average_response_time_ms: float
    active_connections: int
    last_updated: datetime


class PerformanceMonitor:
    """Comprehensive performance monitoring and resilience dashboard."""

    def __init__(
        self,
        metrics_retention_hours: int = 24,
        sampling_interval_seconds: int = 60,
        alert_threshold_error_rate: float = 5.0,
        alert_threshold_response_time_ms: float = 5000.0,
    ):
        """Initialize performance monitor.

        Args:
            metrics_retention_hours: How long to retain metrics data
            sampling_interval_seconds: Interval for periodic sampling
            alert_threshold_error_rate: Error rate threshold for alerts (%)
            alert_threshold_response_time_ms: Response time threshold for alerts (ms)
        """
        self.metrics_retention = timedelta(hours=metrics_retention_hours)
        self.sampling_interval = sampling_interval_seconds
        self.alert_threshold_error_rate = alert_threshold_error_rate
        self.alert_threshold_response_time_ms = alert_threshold_response_time_ms

        # Environment-based memory limits for high-throughput scenarios
        environment = os.getenv("ENVIRONMENT", "development").lower()

        if environment == "production":
            # Production: Conservative limits for high-throughput
            main_metrics_limit = int(os.getenv("PERF_MONITOR_MAIN_LIMIT", "2000"))
            component_metrics_limit = int(
                os.getenv("PERF_MONITOR_COMPONENT_LIMIT", "500")
            )
            alerts_limit = int(os.getenv("PERF_MONITOR_ALERTS_LIMIT", "500"))
        elif environment == "staging":
            # Staging: Moderate limits
            main_metrics_limit = int(os.getenv("PERF_MONITOR_MAIN_LIMIT", "5000"))
            component_metrics_limit = int(
                os.getenv("PERF_MONITOR_COMPONENT_LIMIT", "750")
            )
            alerts_limit = int(os.getenv("PERF_MONITOR_ALERTS_LIMIT", "750"))
        else:  # development
            # Development: Higher limits for debugging
            main_metrics_limit = int(os.getenv("PERF_MONITOR_MAIN_LIMIT", "10000"))
            component_metrics_limit = int(
                os.getenv("PERF_MONITOR_COMPONENT_LIMIT", "1000")
            )
            alerts_limit = int(os.getenv("PERF_MONITOR_ALERTS_LIMIT", "1000"))

        # Metrics storage with intelligent memory management
        self.performance_metrics: deque = deque(maxlen=main_metrics_limit)
        self.resilience_metrics = ResilienceMetrics()
        self.component_metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=component_metrics_limit),
        )

        # System state
        self.start_time = datetime.now(UTC)
        self.last_health_check = datetime.now(UTC)
        self.alerts: deque = deque(maxlen=alerts_limit)

        # Memory management configuration
        self.max_memory_mb = int(os.getenv("PERFORMANCE_MONITOR_MAX_MEMORY_MB", "50"))
        self.cleanup_threshold = float(
            os.getenv("PERFORMANCE_MONITOR_CLEANUP_THRESHOLD", "0.8")
        )
        self.last_cleanup = time.time()
        self.main_metrics_limit = main_metrics_limit
        self.component_metrics_limit = component_metrics_limit
        self.alerts_limit = alerts_limit

        # Prometheus metrics
        self.setup_prometheus_metrics()

        # Background monitoring task
        self.monitoring_task: asyncio.Task | None = None
        self.is_running = False

    def setup_prometheus_metrics(self) -> None:
        """Set up Prometheus metrics collectors."""
        # Performance metrics
        self.request_duration = Histogram(
            "omninode_request_duration_seconds",
            "Request duration in seconds",
            ["component", "operation", "status"],
        )

        self.request_count = Counter(
            "omninode_requests_total",
            "Total number of requests",
            ["component", "operation", "status"],
        )

        self.error_count = Counter(
            "omninode_errors_total",
            "Total number of errors",
            ["component", "error_type"],
        )

        # Resilience metrics
        self.circuit_breaker_trips = Counter(
            "omninode_circuit_breaker_trips_total",
            "Total circuit breaker trips",
            ["component"],
        )

        self.timeout_failures = Counter(
            "omninode_timeout_failures_total",
            "Total timeout failures",
            ["component"],
        )

        self.retry_attempts = Counter(
            "omninode_retry_attempts_total",
            "Total retry attempts",
            ["component"],
        )

        self.dead_letter_messages = Counter(
            "omninode_dead_letter_messages_total",
            "Total dead letter queue messages",
            ["topic"],
        )

        # System health metrics
        self.system_health_status = Gauge(
            "omninode_system_health_status",
            "System health status (0=critical, 1=degraded, 2=healthy)",
        )

        self.active_connections = Gauge(
            "omninode_active_connections",
            "Number of active connections",
            ["service"],
        )

        self.memory_usage = Gauge(
            "omninode_memory_usage_bytes",
            "Memory usage in bytes",
        )

    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self.is_running:
            return

        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        self.is_running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def record_performance_metric(
        self,
        component: str,
        operation: str,
        duration_ms: float,
        success: bool,
        error_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a performance metric.

        Args:
            component: Component name (e.g., 'kafka_client', 'workflow_coordinator')
            operation: Operation name (e.g., 'publish_event', 'execute_task')
            duration_ms: Operation duration in milliseconds
            success: Whether operation was successful
            error_type: Type of error if not successful
            metadata: Additional metadata
        """
        metric = PerformanceMetrics(
            timestamp=datetime.now(UTC),
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
            metadata=metadata or {},
        )

        # Store in local collections
        self.performance_metrics.append(metric)
        self.component_metrics[component].append(metric)

        # Check if memory cleanup is needed
        await self._cleanup_old_metrics_if_needed()

        # Update Prometheus metrics
        status = "success" if success else "error"
        self.request_duration.labels(
            component=component,
            operation=operation,
            status=status,
        ).observe(duration_ms / 1000.0)

        self.request_count.labels(
            component=component,
            operation=operation,
            status=status,
        ).inc()

        if not success and error_type:
            self.error_count.labels(component=component, error_type=error_type).inc()

        # Log significant events
        if not success:
            logger.warning(
                "Performance metric recorded",
                component=component,
                operation=operation,
                duration_ms=duration_ms,
                success=success,
                error_type=error_type,
            )

    async def record_resilience_event(
        self,
        event_type: str,
        component: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a resilience pattern event.

        Args:
            event_type: Type of resilience event
            component: Component where event occurred
            details: Additional event details
        """
        details = details or {}

        # Update resilience metrics
        if event_type == "circuit_breaker_trip":
            self.resilience_metrics.circuit_breaker_trips += 1
            self.circuit_breaker_trips.labels(component=component).inc()

        elif event_type == "timeout_failure":
            self.resilience_metrics.timeout_failures += 1
            self.timeout_failures.labels(component=component).inc()

        elif event_type == "retry_attempt":
            self.resilience_metrics.retry_attempts += 1
            self.retry_attempts.labels(component=component).inc()

        elif event_type == "dead_letter_message":
            self.resilience_metrics.dead_letter_queue_messages += 1
            topic = details.get("topic", "unknown")
            self.dead_letter_messages.labels(topic=topic).inc()

        elif event_type == "grace_period_recovery":
            self.resilience_metrics.grace_period_recoveries += 1

        logger.info(
            "Resilience event recorded",
            event_type=event_type,
            component=component,
            details=details,
        )

    async def _cleanup_old_metrics_if_needed(self) -> None:
        """Clean up old metrics if memory usage is high."""
        current_time = time.time()

        # Check if cleanup is needed (every 5 minutes minimum)
        if current_time - self.last_cleanup < 300:
            return

        # Estimate current memory usage
        estimated_memory_mb = self._estimate_memory_usage()

        if estimated_memory_mb > self.max_memory_mb * self.cleanup_threshold:
            logger.info(
                f"Cleaning up old metrics - estimated memory: {estimated_memory_mb:.1f}MB"
            )

            # Reduce deque sizes by 25% for emergency cleanup
            reduction_factor = 0.75

            # Resize main metrics deque
            new_main_limit = max(int(self.main_metrics_limit * reduction_factor), 500)
            if len(self.performance_metrics) > new_main_limit:
                # Convert to list, keep most recent items, recreate deque
                recent_items = list(self.performance_metrics)[-new_main_limit:]
                self.performance_metrics = deque(recent_items, maxlen=new_main_limit)

            # Resize component metrics deques
            new_component_limit = max(
                int(self.component_metrics_limit * reduction_factor), 100
            )
            for component in list(self.component_metrics.keys()):
                component_deque = self.component_metrics[component]
                if len(component_deque) > new_component_limit:
                    recent_items = list(component_deque)[-new_component_limit:]
                    self.component_metrics[component] = deque(
                        recent_items, maxlen=new_component_limit
                    )

            # Resize alerts deque
            new_alerts_limit = max(int(self.alerts_limit * reduction_factor), 100)
            if len(self.alerts) > new_alerts_limit:
                recent_alerts = list(self.alerts)[-new_alerts_limit:]
                self.alerts = deque(recent_alerts, maxlen=new_alerts_limit)

            self.last_cleanup = current_time
            logger.info(
                f"Metrics cleanup completed - new limits: main={new_main_limit}, component={new_component_limit}, alerts={new_alerts_limit}"
            )

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        try:
            # Rough estimate based on deque sizes and average metric size
            avg_metric_size = 1024  # ~1KB per metric (conservative estimate)

            main_metrics_size = len(self.performance_metrics) * avg_metric_size
            component_metrics_size = sum(
                len(deque_obj) * avg_metric_size
                for deque_obj in self.component_metrics.values()
            )
            alerts_size = len(self.alerts) * avg_metric_size

            total_bytes = main_metrics_size + component_metrics_size + alerts_size
            return total_bytes / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to estimate memory usage: {e}")
            return 0.0

    async def get_system_health(self) -> SystemHealth:
        """Get current system health status.

        Returns:
            SystemHealth object with current status
        """
        now = datetime.now(UTC)
        uptime = (now - self.start_time).total_seconds()

        # Calculate error rate from recent metrics (last 5 minutes)
        recent_cutoff = now - timedelta(minutes=5)
        recent_metrics = [
            m for m in self.performance_metrics if m.timestamp >= recent_cutoff
        ]

        if recent_metrics:
            error_rate = (
                len([m for m in recent_metrics if not m.success])
                / len(recent_metrics)
                * 100
            )
            avg_response_time = sum(m.duration_ms for m in recent_metrics) / len(
                recent_metrics,
            )
        else:
            error_rate = 0.0
            avg_response_time = 0.0

        # Determine health status
        if (
            error_rate > self.alert_threshold_error_rate
            or avg_response_time > self.alert_threshold_response_time_ms
        ):
            status = "critical"
        elif (
            error_rate > self.alert_threshold_error_rate / 2
            or avg_response_time > self.alert_threshold_response_time_ms / 2
        ):
            status = "degraded"
        else:
            status = "healthy"

        # Update Prometheus metric
        status_value = {"healthy": 2, "degraded": 1, "critical": 0}[status]
        self.system_health_status.set(status_value)

        health = SystemHealth(
            status=status,
            uptime_seconds=uptime,
            error_rate_percent=error_rate,
            average_response_time_ms=avg_response_time,
            active_connections=0,  # Would be populated by specific services
            last_updated=now,
        )

        self.last_health_check = now
        return health

    async def get_performance_dashboard(self) -> dict[str, Any]:
        """Get comprehensive performance dashboard data.

        Returns:
            Dictionary containing all dashboard data
        """
        system_health = await self.get_system_health()

        # Component performance summary
        component_summary = {}
        for component, metrics in self.component_metrics.items():
            if not metrics:
                continue

            recent_metrics = [
                m
                for m in metrics
                if m.timestamp >= datetime.now(UTC) - timedelta(hours=1)
            ]

            if recent_metrics:
                success_rate = (
                    len([m for m in recent_metrics if m.success])
                    / len(recent_metrics)
                    * 100
                )
                avg_duration = sum(m.duration_ms for m in recent_metrics) / len(
                    recent_metrics,
                )
                p95_duration = sorted([m.duration_ms for m in recent_metrics])[
                    int(len(recent_metrics) * 0.95)
                ]

                component_summary[component] = {
                    "total_requests": len(recent_metrics),
                    "success_rate_percent": success_rate,
                    "average_duration_ms": avg_duration,
                    "p95_duration_ms": p95_duration,
                    "error_count": len([m for m in recent_metrics if not m.success]),
                }

        # Recent alerts
        recent_alerts = [
            alert
            for alert in self.alerts
            if alert["timestamp"] >= datetime.now(UTC) - timedelta(hours=1)
        ]

        return {
            "system_health": asdict(system_health),
            "resilience_metrics": asdict(self.resilience_metrics),
            "component_performance": component_summary,
            "recent_alerts": recent_alerts,
            "monitoring_config": {
                "metrics_retention_hours": self.metrics_retention.total_seconds()
                / 3600,
                "sampling_interval_seconds": self.sampling_interval,
                "alert_threshold_error_rate": self.alert_threshold_error_rate,
                "alert_threshold_response_time_ms": self.alert_threshold_response_time_ms,
            },
            "statistics": {
                "total_metrics_collected": len(self.performance_metrics),
                "monitored_components": len(self.component_metrics),
                "uptime_hours": (datetime.now(UTC) - self.start_time).total_seconds()
                / 3600,
                "last_health_check": self.last_health_check.isoformat(),
            },
        }

    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics.

        Returns:
            Prometheus metrics in text format
        """
        return generate_latest().decode("utf-8")

    async def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        component: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a performance or health alert.

        Args:
            alert_type: Type of alert (e.g., 'high_error_rate', 'slow_response')
            severity: Alert severity ('info', 'warning', 'critical')
            message: Alert message
            component: Component that triggered the alert
            metadata: Additional alert metadata
        """
        alert = {
            "id": f"alert_{int(time.time())}_{len(self.alerts)}",
            "timestamp": datetime.now(UTC),
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "component": component,
            "metadata": metadata or {},
            "resolved": False,
        }

        self.alerts.append(alert)

        # Log alert
        logger.warning(
            "Performance alert created",
            alert_type=alert_type,
            severity=severity,
            message=message,
            component=component,
        )

        # Bounded deque automatically handles size limits

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for periodic health checks and cleanup."""
        try:
            while self.is_running:
                # Perform health check
                await self.get_system_health()

                # PERFORMANCE OPTIMIZATION: Remove redundant manual filtering
                # The bounded deques (maxlen=10000, maxlen=1000) automatically handle size limits
                # Manual filtering is redundant and wastes CPU/memory on O(n) operations

                # Optional: Clean up stale component metrics (components not seen recently)
                await self._cleanup_stale_components()

                # Check for alert conditions
                await self._check_alert_conditions()

                await asyncio.sleep(self.sampling_interval)

        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error("Error in monitoring loop", error=str(e))

    async def _cleanup_stale_components(self) -> None:
        """Remove component metrics for components not seen in retention period.

        This is more efficient than filtering all metrics as it only removes
        entire component buckets that haven't been used recently.
        """
        cutoff_time = datetime.now(UTC) - self.metrics_retention
        stale_components = []

        for component, metrics in self.component_metrics.items():
            if not metrics or (metrics[-1].timestamp < cutoff_time):
                stale_components.append(component)

        for component in stale_components:
            del self.component_metrics[component]
            logger.debug(f"Cleaned up stale component metrics: {component}")

    async def _check_alert_conditions(self) -> None:
        """Check for conditions that should trigger alerts."""
        system_health = await self.get_system_health()

        # High error rate alert
        if system_health.error_rate_percent > self.alert_threshold_error_rate:
            await self.create_alert(
                alert_type="high_error_rate",
                severity=(
                    "critical"
                    if system_health.error_rate_percent
                    > self.alert_threshold_error_rate * 2
                    else "warning"
                ),
                message=f"Error rate is {system_health.error_rate_percent:.2f}% (threshold: {self.alert_threshold_error_rate}%)",
                metadata={"error_rate": system_health.error_rate_percent},
            )

        # Slow response time alert
        if (
            system_health.average_response_time_ms
            > self.alert_threshold_response_time_ms
        ):
            await self.create_alert(
                alert_type="slow_response_time",
                severity=(
                    "critical"
                    if system_health.average_response_time_ms
                    > self.alert_threshold_response_time_ms * 2
                    else "warning"
                ),
                message=f"Average response time is {system_health.average_response_time_ms:.2f}ms (threshold: {self.alert_threshold_response_time_ms}ms)",
                metadata={"response_time_ms": system_health.average_response_time_ms},
            )

        # Circuit breaker trip alert
        if self.resilience_metrics.circuit_breaker_trips > 0:
            await self.create_alert(
                alert_type="circuit_breaker_trips",
                severity="warning",
                message=f"Circuit breakers have tripped {self.resilience_metrics.circuit_breaker_trips} times",
                metadata={"trip_count": self.resilience_metrics.circuit_breaker_trips},
            )


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


async def init_performance_monitoring() -> None:
    """Initialize and start performance monitoring."""
    await performance_monitor.start_monitoring()
    logger.info("Global performance monitoring initialized")


async def shutdown_performance_monitoring() -> None:
    """Shutdown performance monitoring."""
    await performance_monitor.stop_monitoring()
    logger.info("Global performance monitoring shut down")
