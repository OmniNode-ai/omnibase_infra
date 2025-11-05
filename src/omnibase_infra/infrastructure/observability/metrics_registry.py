"""
Metrics Registry for ONEX Infrastructure.

Provides Prometheus metrics registry setup with pre-configured metrics for
infrastructure components. Includes factory methods for common metrics patterns
and integration with health checks.

Features:
- Prometheus metrics registry with standard metrics
- Pre-configured counters, gauges, histograms for infrastructure
- Factory methods for creating component-specific metrics
- Integration with health checks and monitoring
- Automatic metric labeling for services and environments
"""

import os
from typing import Any

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError


class MetricsConfig:
    """Configuration for metrics collection."""

    def __init__(
        self,
        service_name: str = "omnibase_infrastructure",
        environment: str | None = None,
        include_default_metrics: bool = True,
        enable_process_metrics: bool = True,
    ):
        """
        Initialize metrics configuration.

        Args:
            service_name: Name of the service for metric labeling
            environment: Deployment environment
            include_default_metrics: Include default Prometheus metrics
            enable_process_metrics: Enable process-level metrics
        """
        self.service_name = service_name
        self.environment = environment or self._detect_environment()
        self.include_default_metrics = include_default_metrics
        self.enable_process_metrics = (
            os.getenv("METRICS_ENABLE_PROCESS", str(enable_process_metrics)).lower()
            == "true"
        )

    def _detect_environment(self) -> str:
        """Detect current deployment environment."""
        env_vars = ["ENVIRONMENT", "ENV", "DEPLOYMENT_ENV", "NODE_ENV"]
        for var in env_vars:
            value = os.getenv(var)
            if value:
                return value.lower()
        return "development"


class MetricsRegistry:
    """
    Prometheus metrics registry for ONEX infrastructure.

    Provides centralized metrics management with pre-configured metrics
    for common infrastructure patterns.
    """

    _instance: "MetricsRegistry | None" = None
    _is_initialized: bool = False

    def __init__(
        self,
        config: MetricsConfig | None = None,
        registry: CollectorRegistry | None = None,
    ):
        """
        Initialize metrics registry.

        Args:
            config: Metrics configuration (auto-detected if None)
            registry: Custom Prometheus registry (uses default if None)
        """
        self.config = config or MetricsConfig()
        self.registry = registry or REGISTRY

        # Standard infrastructure metrics
        self._request_counter: Counter | None = None
        self._error_counter: Counter | None = None
        self._message_counter: Counter | None = None
        self._connection_gauge: Gauge | None = None
        self._queue_size_gauge: Gauge | None = None
        self._latency_histogram: Histogram | None = None
        self._duration_histogram: Histogram | None = None

    @classmethod
    def get_instance(cls, config: MetricsConfig | None = None) -> "MetricsRegistry":
        """
        Get singleton metrics registry instance.

        Args:
            config: Configuration for first initialization

        Returns:
            Singleton metrics registry instance
        """
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def initialize(self) -> None:
        """
        Initialize standard infrastructure metrics.

        Creates pre-configured metrics for common infrastructure patterns.

        Raises:
            OnexError: If initialization fails
        """
        if self._is_initialized:
            return

        try:
            # Request metrics
            self._request_counter = Counter(
                "infrastructure_requests_total",
                "Total number of infrastructure requests",
                ["service", "operation", "status", "environment"],
                registry=self.registry,
            )

            # Error metrics
            self._error_counter = Counter(
                "infrastructure_errors_total",
                "Total number of infrastructure errors",
                ["service", "error_type", "error_code", "environment"],
                registry=self.registry,
            )

            # Message/event metrics
            self._message_counter = Counter(
                "infrastructure_messages_total",
                "Total number of messages processed",
                ["service", "topic", "operation", "status", "environment"],
                registry=self.registry,
            )

            # Connection pool metrics
            self._connection_gauge = Gauge(
                "infrastructure_connections_active",
                "Number of active connections",
                ["service", "pool_name", "environment"],
                registry=self.registry,
            )

            # Queue metrics
            self._queue_size_gauge = Gauge(
                "infrastructure_queue_size",
                "Current queue size",
                ["service", "queue_name", "environment"],
                registry=self.registry,
            )

            # Latency metrics (for operations like network calls, queries)
            self._latency_histogram = Histogram(
                "infrastructure_operation_latency_seconds",
                "Operation latency in seconds",
                ["service", "operation", "environment"],
                buckets=(
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
                ),
                registry=self.registry,
            )

            # Duration metrics (for longer-running operations)
            self._duration_histogram = Histogram(
                "infrastructure_operation_duration_seconds",
                "Operation duration in seconds",
                ["service", "operation", "environment"],
                buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
                registry=self.registry,
            )

            self._is_initialized = True

        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.CONFIGURATION_ERROR,
                message=f"Failed to initialize metrics registry: {e!s}",
            ) from e

    # Request metrics
    def record_request(
        self, service: str, operation: str, status: str = "success"
    ) -> None:
        """
        Record infrastructure request.

        Args:
            service: Service name (e.g., "postgres_adapter", "kafka_adapter")
            operation: Operation name (e.g., "query", "publish")
            status: Request status ("success", "failure", "timeout")
        """
        if self._request_counter:
            self._request_counter.labels(
                service=service,
                operation=operation,
                status=status,
                environment=self.config.environment,
            ).inc()

    # Error metrics
    def record_error(
        self, service: str, error_type: str, error_code: str | None = None
    ) -> None:
        """
        Record infrastructure error.

        Args:
            service: Service name
            error_type: Type of error (e.g., "connection_error", "timeout")
            error_code: Optional error code (e.g., CoreErrorCode name)
        """
        if self._error_counter:
            self._error_counter.labels(
                service=service,
                error_type=error_type,
                error_code=error_code or "unknown",
                environment=self.config.environment,
            ).inc()

    # Message metrics
    def record_message(
        self,
        service: str,
        topic: str,
        operation: str = "process",
        status: str = "success",
    ) -> None:
        """
        Record message/event processing.

        Args:
            service: Service name (e.g., "kafka_adapter")
            topic: Topic or queue name
            operation: Operation type ("produce", "consume", "process")
            status: Processing status ("success", "failure", "retry")
        """
        if self._message_counter:
            self._message_counter.labels(
                service=service,
                topic=topic,
                operation=operation,
                status=status,
                environment=self.config.environment,
            ).inc()

    # Connection metrics
    def set_active_connections(
        self, service: str, pool_name: str, count: int
    ) -> None:
        """
        Set number of active connections in a pool.

        Args:
            service: Service name (e.g., "postgres_adapter")
            pool_name: Connection pool name
            count: Current number of active connections
        """
        if self._connection_gauge:
            self._connection_gauge.labels(
                service=service,
                pool_name=pool_name,
                environment=self.config.environment,
            ).set(count)

    def increment_connections(self, service: str, pool_name: str) -> None:
        """Increment active connection count."""
        if self._connection_gauge:
            self._connection_gauge.labels(
                service=service,
                pool_name=pool_name,
                environment=self.config.environment,
            ).inc()

    def decrement_connections(self, service: str, pool_name: str) -> None:
        """Decrement active connection count."""
        if self._connection_gauge:
            self._connection_gauge.labels(
                service=service,
                pool_name=pool_name,
                environment=self.config.environment,
            ).dec()

    # Queue metrics
    def set_queue_size(self, service: str, queue_name: str, size: int) -> None:
        """
        Set current queue size.

        Args:
            service: Service name
            queue_name: Queue or buffer name
            size: Current queue size
        """
        if self._queue_size_gauge:
            self._queue_size_gauge.labels(
                service=service,
                queue_name=queue_name,
                environment=self.config.environment,
            ).set(size)

    # Latency and duration metrics
    def record_latency(self, service: str, operation: str, latency_seconds: float) -> None:
        """
        Record operation latency.

        Use for fast operations (network calls, queries, etc.)

        Args:
            service: Service name
            operation: Operation name
            latency_seconds: Latency in seconds
        """
        if self._latency_histogram:
            self._latency_histogram.labels(
                service=service,
                operation=operation,
                environment=self.config.environment,
            ).observe(latency_seconds)

    def record_duration(
        self, service: str, operation: str, duration_seconds: float
    ) -> None:
        """
        Record operation duration.

        Use for longer-running operations (batch processing, migrations, etc.)

        Args:
            service: Service name
            operation: Operation name
            duration_seconds: Duration in seconds
        """
        if self._duration_histogram:
            self._duration_histogram.labels(
                service=service,
                operation=operation,
                environment=self.config.environment,
            ).observe(duration_seconds)

    # Custom metrics factory
    def create_counter(
        self, name: str, documentation: str, labels: list[str] | None = None
    ) -> Counter:
        """
        Create a custom counter metric.

        Args:
            name: Metric name (will be prefixed with "infrastructure_")
            documentation: Metric documentation
            labels: Label names (environment is added automatically)

        Returns:
            Prometheus Counter instance
        """
        metric_labels = labels or []
        if "environment" not in metric_labels:
            metric_labels.append("environment")

        return Counter(
            f"infrastructure_{name}",
            documentation,
            metric_labels,
            registry=self.registry,
        )

    def create_gauge(
        self, name: str, documentation: str, labels: list[str] | None = None
    ) -> Gauge:
        """
        Create a custom gauge metric.

        Args:
            name: Metric name (will be prefixed with "infrastructure_")
            documentation: Metric documentation
            labels: Label names (environment is added automatically)

        Returns:
            Prometheus Gauge instance
        """
        metric_labels = labels or []
        if "environment" not in metric_labels:
            metric_labels.append("environment")

        return Gauge(
            f"infrastructure_{name}",
            documentation,
            metric_labels,
            registry=self.registry,
        )

    def create_histogram(
        self,
        name: str,
        documentation: str,
        labels: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        """
        Create a custom histogram metric.

        Args:
            name: Metric name (will be prefixed with "infrastructure_")
            documentation: Metric documentation
            labels: Label names (environment is added automatically)
            buckets: Histogram buckets (uses Prometheus defaults if None)

        Returns:
            Prometheus Histogram instance
        """
        metric_labels = labels or []
        if "environment" not in metric_labels:
            metric_labels.append("environment")

        kwargs: dict[str, Any] = {
            "registry": self.registry,
        }
        if buckets:
            kwargs["buckets"] = buckets

        return Histogram(
            f"infrastructure_{name}",
            documentation,
            metric_labels,
            **kwargs,
        )

    # Export metrics
    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)

    def get_health_metrics(self) -> dict[str, Any]:
        """
        Get health metrics for monitoring endpoints.

        Returns:
            Dictionary of health-related metrics
        """
        # This would typically query the registry for current metric values
        # For now, return basic health status
        return {
            "metrics_initialized": self._is_initialized,
            "service_name": self.config.service_name,
            "environment": self.config.environment,
        }


# Global registry accessor functions
def get_metrics_registry(config: MetricsConfig | None = None) -> MetricsRegistry:
    """
    Get the global metrics registry instance.

    Args:
        config: Configuration for first initialization

    Returns:
        Global metrics registry instance
    """
    return MetricsRegistry.get_instance(config)


def initialize_metrics(config: MetricsConfig | None = None) -> None:
    """
    Initialize global metrics registry.

    Args:
        config: Metrics configuration (auto-detected if None)
    """
    registry = get_metrics_registry(config)
    registry.initialize()
