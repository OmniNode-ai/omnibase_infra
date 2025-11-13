"""
Shared Prometheus metrics for OmniNode Bridge nodes.

Provides comprehensive metrics collection for orchestrator, reducer, and registry nodes
with industry-standard Prometheus instrumentation.

Key Features:
- Counter, Gauge, Histogram, Summary metrics
- Node-specific and shared metrics
- Context managers for automatic timing
- Feature flag support (enable_prometheus)
- Thread-safe collectors with separate registries

Metrics Categories:
1. Workflow metrics (orchestrator)
2. Aggregation metrics (reducer)
3. Event publishing metrics (all nodes)
4. Node registration metrics (registry)
5. Error and latency metrics (all nodes)
"""

import logging
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Node types for metric labeling."""

    ORCHESTRATOR = "orchestrator"
    REDUCER = "reducer"
    REGISTRY = "registry"


class BridgeMetricsCollector:
    """
    Prometheus metrics collector for bridge nodes.

    Provides standardized metrics collection across orchestrator, reducer,
    and registry nodes with consistent naming and labeling conventions.

    Thread-safe with separate registry per instance.
    """

    def __init__(
        self,
        node_type: NodeType,
        registry: Optional[CollectorRegistry] = None,
        enable_prometheus: bool = True,
    ):
        """
        Initialize metrics collector.

        Args:
            node_type: Type of node (orchestrator, reducer, registry)
            registry: Optional custom registry (creates new if not provided)
            enable_prometheus: Feature flag to enable/disable metrics collection
        """
        self.node_type = node_type
        self.enable_prometheus = enable_prometheus
        self.registry = registry or CollectorRegistry()

        # Only initialize if enabled
        if self.enable_prometheus:
            self._initialize_metrics()
            logger.info(
                f"Prometheus metrics initialized for {node_type.value} node "
                f"(enable_prometheus={enable_prometheus})"
            )
        else:
            logger.info(
                f"Prometheus metrics DISABLED for {node_type.value} node "
                f"(enable_prometheus={enable_prometheus})"
            )

    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        # === Workflow Metrics (Orchestrator) ===
        self.workflow_counter = Counter(
            "bridge_workflow_total",
            "Total number of workflows processed",
            ["node_type", "status"],
            registry=self.registry,
        )

        self.workflow_duration = Histogram(
            "bridge_workflow_duration_seconds",
            "Workflow execution duration in seconds",
            ["node_type", "status"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        self.workflow_in_progress = Gauge(
            "bridge_workflow_in_progress",
            "Number of workflows currently in progress",
            ["node_type"],
            registry=self.registry,
        )

        self.workflow_step_duration = Histogram(
            "bridge_workflow_step_duration_seconds",
            "Workflow step execution duration",
            ["node_type", "step_name"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )

        # === Aggregation Metrics (Reducer) ===
        self.aggregation_counter = Counter(
            "bridge_aggregation_total",
            "Total number of aggregation operations",
            ["node_type", "aggregation_type", "status"],
            registry=self.registry,
        )

        self.aggregation_duration = Histogram(
            "bridge_aggregation_duration_seconds",
            "Aggregation operation duration",
            ["node_type", "aggregation_type"],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )

        self.aggregation_items_processed = Counter(
            "bridge_aggregation_items_processed_total",
            "Total items processed by aggregation",
            ["node_type", "aggregation_type"],
            registry=self.registry,
        )

        self.aggregation_buffer_size = Gauge(
            "bridge_aggregation_buffer_size",
            "Current size of aggregation buffer",
            ["node_type", "namespace"],
            registry=self.registry,
        )

        # === Event Publishing Metrics (All Nodes) ===
        self.event_published_counter = Counter(
            "bridge_event_published_total",
            "Total events published to Kafka",
            ["node_type", "event_type", "status"],
            registry=self.registry,
        )

        self.event_publish_duration = Histogram(
            "bridge_event_publish_duration_seconds",
            "Event publishing duration",
            ["node_type", "event_type"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry,
        )

        self.event_publish_errors = Counter(
            "bridge_event_publish_errors_total",
            "Total event publishing errors",
            ["node_type", "event_type", "error_type"],
            registry=self.registry,
        )

        # === Node Registration Metrics (Registry & All) ===
        self.node_registration_counter = Counter(
            "bridge_node_registration_total",
            "Total node registrations",
            ["node_type", "status"],
            registry=self.registry,
        )

        self.registered_nodes_gauge = Gauge(
            "bridge_registered_nodes",
            "Number of currently registered nodes",
            ["node_type"],
            registry=self.registry,
        )

        # === Error Tracking Metrics (All Nodes) ===
        self.error_counter = Counter(
            "bridge_errors_total",
            "Total errors by type",
            ["node_type", "error_type", "operation"],
            registry=self.registry,
        )

        # === Latency Summary Metrics (All Nodes) ===
        self.operation_latency = Summary(
            "bridge_operation_latency_seconds",
            "Operation latency summary",
            ["node_type", "operation"],
            registry=self.registry,
        )

        # === Database Metrics (All Nodes) ===
        self.db_query_duration = Histogram(
            "bridge_db_query_duration_seconds",
            "Database query duration",
            ["node_type", "operation"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )

        self.db_query_counter = Counter(
            "bridge_db_query_total",
            "Total database queries",
            ["node_type", "operation", "status"],
            registry=self.registry,
        )

        # === HTTP Request Metrics (All Nodes) ===
        self.http_requests_counter = Counter(
            "bridge_http_requests_total",
            "Total HTTP requests",
            ["node_type", "method", "endpoint", "status_code"],
            registry=self.registry,
        )

        self.http_request_duration = Histogram(
            "bridge_http_request_duration_seconds",
            "HTTP request duration",
            ["node_type", "method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        logger.debug(f"All Prometheus metrics initialized for {self.node_type.value}")

    # === Context Managers for Automatic Timing ===

    @contextmanager
    def time_workflow(self, status: str = "success"):
        """
        Time a workflow execution.

        Usage:
            with metrics.time_workflow(status="success"):
                # Execute workflow
                pass
        """
        if not self.enable_prometheus:
            yield
            return

        start_time = time.perf_counter()
        self.workflow_in_progress.labels(node_type=self.node_type.value).inc()

        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.workflow_duration.labels(
                node_type=self.node_type.value, status=status
            ).observe(duration)
            self.workflow_counter.labels(
                node_type=self.node_type.value, status=status
            ).inc()
            self.workflow_in_progress.labels(node_type=self.node_type.value).dec()

    @contextmanager
    def time_workflow_step(self, step_name: str):
        """Time an individual workflow step."""
        if not self.enable_prometheus:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.workflow_step_duration.labels(
                node_type=self.node_type.value, step_name=step_name
            ).observe(duration)

    @contextmanager
    def time_aggregation(self, aggregation_type: str, status: str = "success"):
        """Time an aggregation operation."""
        if not self.enable_prometheus:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.aggregation_duration.labels(
                node_type=self.node_type.value, aggregation_type=aggregation_type
            ).observe(duration)
            self.aggregation_counter.labels(
                node_type=self.node_type.value,
                aggregation_type=aggregation_type,
                status=status,
            ).inc()

    @contextmanager
    def time_event_publish(self, event_type: str, status: str = "success"):
        """Time event publishing operation."""
        if not self.enable_prometheus:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.event_publish_duration.labels(
                node_type=self.node_type.value, event_type=event_type
            ).observe(duration)
            self.event_published_counter.labels(
                node_type=self.node_type.value, event_type=event_type, status=status
            ).inc()

    @contextmanager
    def time_db_query(self, operation: str, status: str = "success"):
        """Time database query operation."""
        if not self.enable_prometheus:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.db_query_duration.labels(
                node_type=self.node_type.value, operation=operation
            ).observe(duration)
            self.db_query_counter.labels(
                node_type=self.node_type.value, operation=operation, status=status
            ).inc()

    @contextmanager
    def time_http_request(self, method: str, endpoint: str, status_code: int = 200):
        """Time HTTP request."""
        if not self.enable_prometheus:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.http_request_duration.labels(
                node_type=self.node_type.value, method=method, endpoint=endpoint
            ).observe(duration)
            self.http_requests_counter.labels(
                node_type=self.node_type.value,
                method=method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).inc()

    # === Direct Recording Methods ===

    def record_aggregation_items(self, aggregation_type: str, count: int):
        """Record number of items processed in aggregation."""
        if not self.enable_prometheus:
            return

        self.aggregation_items_processed.labels(
            node_type=self.node_type.value, aggregation_type=aggregation_type
        ).inc(count)

    def set_aggregation_buffer_size(self, namespace: str, size: int):
        """Set current aggregation buffer size."""
        if not self.enable_prometheus:
            return

        self.aggregation_buffer_size.labels(
            node_type=self.node_type.value, namespace=namespace
        ).set(size)

    def record_event_publish_error(self, event_type: str, error_type: str):
        """Record event publishing error."""
        if not self.enable_prometheus:
            return

        self.event_publish_errors.labels(
            node_type=self.node_type.value, event_type=event_type, error_type=error_type
        ).inc()

    def record_node_registration(self, status: str):
        """Record node registration event."""
        if not self.enable_prometheus:
            return

        self.node_registration_counter.labels(
            node_type=self.node_type.value, status=status
        ).inc()

    def set_registered_nodes_count(self, count: int):
        """Set current count of registered nodes."""
        if not self.enable_prometheus:
            return

        self.registered_nodes_gauge.labels(node_type=self.node_type.value).set(count)

    def record_error(self, error_type: str, operation: str):
        """Record error occurrence."""
        if not self.enable_prometheus:
            return

        self.error_counter.labels(
            node_type=self.node_type.value, error_type=error_type, operation=operation
        ).inc()

    def record_operation_latency(self, operation: str, duration_seconds: float):
        """Record operation latency."""
        if not self.enable_prometheus:
            return

        self.operation_latency.labels(
            node_type=self.node_type.value, operation=operation
        ).observe(duration_seconds)

    # === Metrics Export ===

    def get_metrics(self) -> bytes:
        """
        Get Prometheus metrics in text format for scraping.

        Returns:
            Metrics in Prometheus text format
        """
        if not self.enable_prometheus:
            return b"# Prometheus metrics disabled\n"

        return generate_latest(self.registry)

    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get human-readable metrics summary.

        Returns:
            Dictionary with key metrics
        """
        if not self.enable_prometheus:
            return {
                "enabled": False,
                "message": "Prometheus metrics are disabled",
            }

        # Collect metric samples
        metric_values: dict[str, Any] = {
            "enabled": True,
            "node_type": self.node_type.value,
            "metrics": {},
        }

        # Parse registry metrics
        for family in self.registry.collect():
            metric_name = family.name
            samples = list(family.samples)

            if samples:
                # Store first sample value as example
                metric_values["metrics"][metric_name] = {
                    "type": family.type,
                    "sample_count": len(samples),
                    "example_value": samples[0].value if samples else 0,
                }

        return metric_values


# === Factory Functions ===


def create_orchestrator_metrics(
    registry: Optional[CollectorRegistry] = None, enable_prometheus: bool = True
) -> BridgeMetricsCollector:
    """
    Create metrics collector for orchestrator node.

    Args:
        registry: Optional custom registry
        enable_prometheus: Feature flag to enable/disable metrics

    Returns:
        Configured metrics collector
    """
    return BridgeMetricsCollector(
        node_type=NodeType.ORCHESTRATOR,
        registry=registry,
        enable_prometheus=enable_prometheus,
    )


def create_reducer_metrics(
    registry: Optional[CollectorRegistry] = None, enable_prometheus: bool = True
) -> BridgeMetricsCollector:
    """
    Create metrics collector for reducer node.

    Args:
        registry: Optional custom registry
        enable_prometheus: Feature flag to enable/disable metrics

    Returns:
        Configured metrics collector
    """
    return BridgeMetricsCollector(
        node_type=NodeType.REDUCER,
        registry=registry,
        enable_prometheus=enable_prometheus,
    )


def create_registry_metrics(
    registry: Optional[CollectorRegistry] = None, enable_prometheus: bool = True
) -> BridgeMetricsCollector:
    """
    Create metrics collector for registry node.

    Args:
        registry: Optional custom registry
        enable_prometheus: Feature flag to enable/disable metrics

    Returns:
        Configured metrics collector
    """
    return BridgeMetricsCollector(
        node_type=NodeType.REGISTRY,
        registry=registry,
        enable_prometheus=enable_prometheus,
    )
