"""
Advanced metrics collection and monitoring for MetadataStampingService Phase 2.

Provides comprehensive performance monitoring, alerting, and observability
with Prometheus metrics, custom collectors, and real-time alerting.
"""

import asyncio
import logging
import statistics
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to collect."""

    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    INFO = "info"


@dataclass
class AlertRule:
    """Configuration for performance alerts."""

    name: str
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    level: AlertLevel
    message: str
    cooldown_seconds: int = 300  # 5 minutes
    enabled: bool = True
    last_triggered: float = 0.0


class Alert(BaseModel):
    """Alert notification model."""

    rule_name: str
    level: AlertLevel
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: float = Field(default_factory=time.time)
    instance: str = "metadata-stamping-service"


class MetricsCollector:
    """
    Advanced metrics collector for MetadataStampingService.

    Features:
    - Prometheus metrics integration
    - Custom performance metrics
    - Real-time alerting
    - Performance trend analysis
    - Resource usage monitoring
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.alert_rules: list[AlertRule] = []
        self.alert_callbacks: list[Callable[[Alert], None]] = []

        # Initialize Prometheus metrics
        self._initialize_metrics()

        # Performance tracking
        self._performance_history: dict[str, list[float]] = {}
        self._alert_history: list[Alert] = []

        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""
        # Request metrics
        self.request_counter = Counter(
            "metadata_stamping_requests_total",
            "Total number of metadata stamping requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "metadata_stamping_request_duration_seconds",
            "Time spent processing requests",
            ["method", "endpoint"],
            buckets=[
                0.001,
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
                10.0,
            ],
            registry=self.registry,
        )

        # BLAKE3 hash metrics
        self.hash_counter = Counter(
            "blake3_hash_operations_total",
            "Total number of BLAKE3 hash operations",
            registry=self.registry,
        )

        self.hash_duration = Histogram(
            "blake3_hash_duration_seconds",
            "Time spent generating BLAKE3 hashes",
            buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1],
            registry=self.registry,
        )

        self.hash_violations = Counter(
            "blake3_hash_violations_total",
            "Number of BLAKE3 hash operations exceeding 2ms",
            registry=self.registry,
        )

        # Cache metrics
        self.cache_hits = Counter(
            "redis_cache_hits_total",
            "Total number of cache hits",
            ["cache_type"],
            registry=self.registry,
        )

        self.cache_misses = Counter(
            "redis_cache_misses_total",
            "Total number of cache misses",
            ["cache_type"],
            registry=self.registry,
        )

        self.cache_operations = Counter(
            "redis_cache_operations_total",
            "Total number of cache operations",
            ["operation"],
            registry=self.registry,
        )

        self.cache_size = Gauge(
            "redis_cache_size_bytes",
            "Current cache size in bytes",
            registry=self.registry,
        )

        # Kafka metrics
        self.kafka_messages_produced = Counter(
            "kafka_messages_produced_total",
            "Total number of Kafka messages produced",
            ["topic"],
            registry=self.registry,
        )

        self.kafka_messages_consumed = Counter(
            "kafka_messages_consumed_total",
            "Total number of Kafka messages consumed",
            ["topic"],
            registry=self.registry,
        )

        self.kafka_production_errors = Counter(
            "kafka_production_errors_total",
            "Total number of Kafka production errors",
            registry=self.registry,
        )

        self.kafka_consumer_lag = Gauge(
            "kafka_consumer_lag_sum",
            "Current Kafka consumer lag",
            ["topic", "partition"],
            registry=self.registry,
        )

        # Database metrics
        self.db_connections_active = Gauge(
            "postgres_connections_active",
            "Number of active PostgreSQL connections",
            registry=self.registry,
        )

        self.db_query_duration = Histogram(
            "postgres_query_duration_seconds",
            "Time spent executing PostgreSQL queries",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry,
        )

        # Metadata extraction metrics
        self.extraction_counter = Counter(
            "metadata_extraction_total",
            "Total number of metadata extractions",
            ["file_type", "extraction_level"],
            registry=self.registry,
        )

        self.extraction_duration = Histogram(
            "metadata_extraction_duration_seconds",
            "Time spent extracting metadata",
            ["file_type"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

        self.extraction_errors = Counter(
            "metadata_extraction_errors_total",
            "Total number of metadata extraction errors",
            ["file_type", "error_type"],
            registry=self.registry,
        )

        # System metrics
        self.memory_usage = Gauge(
            "metadata_stamping_memory_usage_bytes",
            "Current memory usage in bytes",
            registry=self.registry,
        )

        self.cpu_usage = Gauge(
            "metadata_stamping_cpu_usage_percent",
            "Current CPU usage percentage",
            registry=self.registry,
        )

        self.file_descriptors = Gauge(
            "metadata_stamping_file_descriptors",
            "Number of open file descriptors",
            registry=self.registry,
        )

        # Service info
        self.service_info = Info(
            "metadata_stamping_service_info",
            "Information about the metadata stamping service",
            registry=self.registry,
        )

        # Set service info
        self.service_info.info(
            {
                "version": "0.2.0",
                "phase": "Phase 2 - Advanced Features",
                "python_version": "3.12",
                "build_date": "2024-01-15",
            }
        )

    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_worker())
        logger.info("Metrics monitoring started")

    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Metrics monitoring stopped")

    async def _monitoring_worker(self):
        """Background worker for monitoring and alerting."""
        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Check alert rules
                await self._check_alert_rules()

                # Update system metrics
                await self._update_system_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring worker error: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    async def _check_alert_rules(self):
        """Check all alert rules and trigger alerts if needed."""
        current_time = time.time()

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            # Check cooldown
            if current_time - rule.last_triggered < rule.cooldown_seconds:
                continue

            try:
                # Get current metric value
                current_value = await self._get_metric_value(rule.metric_name)
                if current_value is None:
                    continue

                # Check threshold
                should_alert = False
                if (
                    (rule.comparison == "gt" and current_value > rule.threshold)
                    or (rule.comparison == "lt" and current_value < rule.threshold)
                    or (rule.comparison == "gte" and current_value >= rule.threshold)
                    or (rule.comparison == "lte" and current_value <= rule.threshold)
                    or (rule.comparison == "eq" and current_value == rule.threshold)
                ):
                    should_alert = True

                if should_alert:
                    alert = Alert(
                        rule_name=rule.name,
                        level=rule.level,
                        metric_name=rule.metric_name,
                        current_value=current_value,
                        threshold_value=rule.threshold,
                        message=rule.message.format(
                            current_value=current_value, threshold=rule.threshold
                        ),
                    )

                    await self._trigger_alert(alert)
                    rule.last_triggered = current_time

            except Exception as e:
                logger.error(f"Error checking alert rule {rule.name}: {e}")

    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric."""
        try:
            # This would integrate with actual metric collection
            # For now, return simulated values
            if "hash_duration" in metric_name:
                return statistics.mean(
                    self._performance_history.get("hash_duration", [0.001])
                )
            elif "error_rate" in metric_name:
                return 0.5  # 0.5% error rate
            elif "cpu_usage" in metric_name:
                return 65.0  # 65% CPU usage
            elif "memory_usage" in metric_name:
                return 450.0  # 450MB memory usage

            return None

        except Exception as e:
            logger.error(f"Error getting metric value for {metric_name}: {e}")
            return None

    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert notification."""
        self._alert_history.append(alert)

        # Keep only last 100 alerts
        if len(self._alert_history) > 100:
            self._alert_history = self._alert_history[-100:]

        logger.warning(
            f"ALERT [{alert.level.value.upper()}] {alert.rule_name}: {alert.message}"
        )

        # Notify registered callbacks
        for callback in self.alert_callbacks:
            try:
                (
                    await callback(alert)
                    if asyncio.iscoroutinefunction(callback)
                    else callback(alert)
                )
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def _update_system_metrics(self):
        """Update system resource metrics."""
        try:
            import psutil

            process = psutil.Process()

            # Memory usage
            memory_info = process.memory_info()
            self.memory_usage.set(memory_info.rss)

            # CPU usage
            cpu_percent = process.cpu_percent()
            self.cpu_usage.set(cpu_percent)

            # File descriptors
            try:
                num_fds = process.num_fds()
                self.file_descriptors.set(num_fds)
            except AttributeError:
                # Windows doesn't have num_fds
                pass

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    # Context managers for timing operations

    @asynccontextmanager
    async def time_request(self, method: str, endpoint: str):
        """Time an HTTP request."""
        start_time = time.perf_counter()
        status = "success"

        try:
            yield
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - start_time
            self.request_duration.labels(method=method, endpoint=endpoint).observe(
                duration
            )
            self.request_counter.labels(
                method=method, endpoint=endpoint, status=status
            ).inc()

    @asynccontextmanager
    async def time_hash_operation(self):
        """Time a BLAKE3 hash operation."""
        start_time = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.hash_duration.observe(duration)
            self.hash_counter.inc()

            # Track performance history
            if "hash_duration" not in self._performance_history:
                self._performance_history["hash_duration"] = []

            self._performance_history["hash_duration"].append(duration)

            # Keep only last 1000 measurements
            if len(self._performance_history["hash_duration"]) > 1000:
                self._performance_history["hash_duration"] = self._performance_history[
                    "hash_duration"
                ][-1000:]

            # Check for performance violations (>2ms)
            if duration > 0.002:
                self.hash_violations.inc()

    @asynccontextmanager
    async def time_db_query(self, operation: str):
        """Time a database query."""
        start_time = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.db_query_duration.labels(operation=operation).observe(duration)

    @asynccontextmanager
    async def time_metadata_extraction(self, file_type: str):
        """Time a metadata extraction operation."""
        start_time = time.perf_counter()
        extraction_level = "standard"  # Default

        try:
            yield
            self.extraction_counter.labels(
                file_type=file_type, extraction_level=extraction_level
            ).inc()
        except Exception as e:
            error_type = type(e).__name__
            self.extraction_errors.labels(
                file_type=file_type, error_type=error_type
            ).inc()
            raise
        finally:
            duration = time.perf_counter() - start_time
            self.extraction_duration.labels(file_type=file_type).observe(duration)

    # Metric recording methods

    def record_cache_hit(self, cache_type: str = "redis"):
        """Record a cache hit."""
        self.cache_hits.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str = "redis"):
        """Record a cache miss."""
        self.cache_misses.labels(cache_type=cache_type).inc()

    def record_cache_operation(self, operation: str):
        """Record a cache operation."""
        self.cache_operations.labels(operation=operation).inc()

    def set_cache_size(self, size_bytes: int):
        """Set current cache size."""
        self.cache_size.set(size_bytes)

    def record_kafka_message_produced(self, topic: str):
        """Record a Kafka message production."""
        self.kafka_messages_produced.labels(topic=topic).inc()

    def record_kafka_message_consumed(self, topic: str):
        """Record a Kafka message consumption."""
        self.kafka_messages_consumed.labels(topic=topic).inc()

    def record_kafka_production_error(self):
        """Record a Kafka production error."""
        self.kafka_production_errors.inc()

    def set_kafka_consumer_lag(self, topic: str, partition: int, lag: int):
        """Set Kafka consumer lag."""
        self.kafka_consumer_lag.labels(topic=topic, partition=str(partition)).set(lag)

    def set_db_connections_active(self, count: int):
        """Set number of active database connections."""
        self.db_connections_active.set(count)

    # Alert management

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule by name."""
        self.alert_rules = [rule for rule in self.alert_rules if rule.name != rule_name]
        logger.info(f"Removed alert rule: {rule_name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)

    def get_alert_history(self, limit: int = 50) -> list[Alert]:
        """Get recent alerts."""
        return self._alert_history[-limit:]

    # Metrics export

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary statistics."""
        # Collect metrics using proper Prometheus API
        metric_values = self._collect_metric_values()

        summary = {
            "requests": {
                "total": metric_values.get("metadata_stamping_requests_total", 0),
                "errors": metric_values.get(
                    "metadata_stamping_requests_total_errors", 0
                ),
            },
            "hash_operations": {
                "total": metric_values.get("blake3_hash_operations_total", 0),
                "violations": metric_values.get("blake3_hash_violations_total", 0),
                "average_duration_ms": 0.0,
                "p95_duration_ms": 0.0,
            },
            "cache": {
                "hits": metric_values.get("redis_cache_hits_total", 0),
                "misses": metric_values.get("redis_cache_misses_total", 0),
                "hit_ratio": 0.0,
            },
            "kafka": {
                "messages_produced": metric_values.get(
                    "kafka_messages_produced_total", 0
                ),
                "messages_consumed": metric_values.get(
                    "kafka_messages_consumed_total", 0
                ),
                "production_errors": metric_values.get(
                    "kafka_production_errors_total", 0
                ),
            },
            "metadata_extraction": {
                "total": metric_values.get("metadata_extraction_total", 0),
                "errors": metric_values.get("metadata_extraction_errors_total", 0),
            },
            "alerts": {
                "total_rules": len(self.alert_rules),
                "active_rules": len([r for r in self.alert_rules if r.enabled]),
                "recent_alerts": len(self._alert_history),
            },
        }

        # Calculate derived metrics
        if summary["hash_operations"]["total"] > 0:
            hash_durations = self._performance_history.get("hash_duration", [])
            if hash_durations:
                summary["hash_operations"]["average_duration_ms"] = (
                    statistics.mean(hash_durations) * 1000
                )
                # Calculate P95 with proper sample count validation
                if len(hash_durations) >= 2:
                    summary["hash_operations"]["p95_duration_ms"] = (
                        statistics.quantiles(hash_durations, n=20)[18] * 1000
                    )
                elif len(hash_durations) == 1:
                    summary["hash_operations"]["p95_duration_ms"] = (
                        hash_durations[0] * 1000
                    )
                else:
                    summary["hash_operations"]["p95_duration_ms"] = 0.0

        cache_total = summary["cache"]["hits"] + summary["cache"]["misses"]
        if cache_total > 0:
            summary["cache"]["hit_ratio"] = summary["cache"]["hits"] / cache_total

        return summary

    def _collect_metric_values(self) -> dict[str, float]:
        """Collect current metric values using proper Prometheus API."""
        metric_values = {}

        # Collect all metrics from registry
        for metric_family in self.registry.collect():
            metric_name = metric_family.name

            # Sum all samples for each metric
            total_value = 0.0
            error_value = 0.0

            for sample in metric_family.samples:
                # Get sample value
                value = sample.value

                # For counters and gauges, sum all labeled values
                total_value += value

                # Track errors separately for request metrics
                if metric_name == "metadata_stamping_requests_total":
                    labels = sample.labels
                    if labels.get("status") == "error":
                        error_value += value

            # Store aggregated values
            metric_values[metric_name] = total_value
            if error_value > 0:
                metric_values[f"{metric_name}_errors"] = error_value

        return metric_values


# Default alert rules for MetadataStampingService
DEFAULT_ALERT_RULES = [
    AlertRule(
        name="hash_performance_degradation",
        metric_name="hash_duration_p95",
        threshold=0.002,  # 2ms
        comparison="gt",
        level=AlertLevel.WARNING,
        message="BLAKE3 hash P95 latency ({current_value:.3f}s) exceeds threshold ({threshold:.3f}s)",
        cooldown_seconds=300,
    ),
    AlertRule(
        name="high_error_rate",
        metric_name="error_rate_percent",
        threshold=5.0,  # 5%
        comparison="gt",
        level=AlertLevel.ERROR,
        message="Error rate ({current_value:.1f}%) exceeds threshold ({threshold:.1f}%)",
        cooldown_seconds=600,
    ),
    AlertRule(
        name="high_cpu_usage",
        metric_name="cpu_usage_percent",
        threshold=80.0,  # 80%
        comparison="gt",
        level=AlertLevel.WARNING,
        message="CPU usage ({current_value:.1f}%) exceeds threshold ({threshold:.1f}%)",
        cooldown_seconds=300,
    ),
    AlertRule(
        name="high_memory_usage",
        metric_name="memory_usage_mb",
        threshold=512.0,  # 512MB
        comparison="gt",
        level=AlertLevel.WARNING,
        message="Memory usage ({current_value:.1f}MB) exceeds threshold ({threshold:.1f}MB)",
        cooldown_seconds=600,
    ),
    AlertRule(
        name="cache_hit_rate_low",
        metric_name="cache_hit_rate_percent",
        threshold=70.0,  # 70%
        comparison="lt",
        level=AlertLevel.WARNING,
        message="Cache hit rate ({current_value:.1f}%) below threshold ({threshold:.1f}%)",
        cooldown_seconds=900,
    ),
]


# Factory function
def create_metrics_collector(
    registry: Optional[CollectorRegistry] = None, include_default_alerts: bool = True
) -> MetricsCollector:
    """Factory function to create metrics collector with default configuration."""
    collector = MetricsCollector(registry)

    if include_default_alerts:
        for rule in DEFAULT_ALERT_RULES:
            collector.add_alert_rule(rule)

    return collector
