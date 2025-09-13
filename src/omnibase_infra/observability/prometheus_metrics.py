"""
Prometheus Metrics Integration for ONEX Infrastructure

Provides comprehensive Prometheus metrics collection for infrastructure
components including Kafka adapters, database connections, and outbox processing.

Per ONEX observability requirements:
- Infrastructure-wide metrics collection
- Performance and health monitoring
- Error rate and latency tracking
- Business metrics for event processing
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, 
        CollectorRegistry, generate_latest,
        CONTENT_TYPE_LATEST, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Graceful degradation if prometheus_client is not installed
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Info = CollectorRegistry = None

from omnibase_core.core.errors.onex_error import OnexError, CoreErrorCode


class MetricType(Enum):
    """Types of Prometheus metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition for a Prometheus metric."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str]
    buckets: Optional[List[float]] = None  # For histograms


class ONEXPrometheusMetrics:
    """
    ONEX Prometheus metrics collector for infrastructure monitoring.
    
    Provides centralized metrics collection with automatic instrumentation
    for infrastructure components and business processes.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus metrics collector.
        
        Args:
            registry: Custom Prometheus registry (uses default if None)
        """
        self._logger = logging.getLogger(__name__)
        
        if not PROMETHEUS_AVAILABLE:
            self._logger.warning("Prometheus client not available, metrics collection disabled")
            self._enabled = False
            return
        
        self._enabled = True
        self._registry = registry or CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        
        # Initialize standard infrastructure metrics
        self._initialize_infrastructure_metrics()
        
        self._logger.info("Prometheus metrics collector initialized")
    
    def _initialize_infrastructure_metrics(self):
        """Initialize standard infrastructure metrics."""
        if not self._enabled:
            return
        
        # Kafka adapter metrics
        kafka_metrics = [
            MetricDefinition(
                "kafka_messages_published_total",
                "Total number of Kafka messages published",
                MetricType.COUNTER,
                ["topic", "status", "client_id"]
            ),
            MetricDefinition(
                "kafka_publish_duration_seconds",
                "Time spent publishing Kafka messages",
                MetricType.HISTOGRAM,
                ["topic", "client_id"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            ),
            MetricDefinition(
                "kafka_circuit_breaker_state",
                "Circuit breaker state (0=closed, 1=half_open, 2=open)",
                MetricType.GAUGE,
                ["client_id"]
            ),
            MetricDefinition(
                "kafka_rate_limit_violations_total",
                "Total number of rate limit violations",
                MetricType.COUNTER,
                ["client_id"]
            ),
            MetricDefinition(
                "kafka_payload_encrypted_total",
                "Total number of encrypted payloads",
                MetricType.COUNTER,
                ["topic", "client_id"]
            )
        ]
        
        # Database metrics
        database_metrics = [
            MetricDefinition(
                "database_connections_active",
                "Number of active database connections",
                MetricType.GAUGE,
                ["database", "schema"]
            ),
            MetricDefinition(
                "database_query_duration_seconds",
                "Time spent executing database queries",
                MetricType.HISTOGRAM,
                ["operation", "table", "status"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
            ),
            MetricDefinition(
                "database_queries_total",
                "Total number of database queries",
                MetricType.COUNTER,
                ["operation", "table", "status"]
            )
        ]
        
        # Outbox pattern metrics
        outbox_metrics = [
            MetricDefinition(
                "outbox_events_pending",
                "Number of pending outbox events",
                MetricType.GAUGE,
                ["schema"]
            ),
            MetricDefinition(
                "outbox_events_processed_total",
                "Total number of processed outbox events",
                MetricType.COUNTER,
                ["schema", "status"]
            ),
            MetricDefinition(
                "outbox_processing_duration_seconds",
                "Time spent processing outbox events",
                MetricType.HISTOGRAM,
                ["schema"],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
            ),
            MetricDefinition(
                "outbox_batch_size",
                "Size of outbox event batches processed",
                MetricType.HISTOGRAM,
                ["schema"],
                buckets=[1, 5, 10, 25, 50, 100, 250, 500]
            )
        ]
        
        # Security metrics
        security_metrics = [
            MetricDefinition(
                "security_violations_total",
                "Total number of security violations",
                MetricType.COUNTER,
                ["violation_type", "client_id"]
            ),
            MetricDefinition(
                "audit_events_total",
                "Total number of audit events logged",
                MetricType.COUNTER,
                ["event_type", "severity"]
            )
        ]
        
        # Register all metrics
        all_metrics = kafka_metrics + database_metrics + outbox_metrics + security_metrics
        
        for metric_def in all_metrics:
            self._register_metric(metric_def)
    
    def _register_metric(self, metric_def: MetricDefinition):
        """Register a single metric with Prometheus."""
        if not self._enabled:
            return
        
        try:
            if metric_def.metric_type == MetricType.COUNTER:
                metric = Counter(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    registry=self._registry
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    registry=self._registry
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                metric = Histogram(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    buckets=metric_def.buckets,
                    registry=self._registry
                )
            elif metric_def.metric_type == MetricType.INFO:
                metric = Info(
                    metric_def.name,
                    metric_def.description,
                    registry=self._registry
                )
            else:
                self._logger.warning(f"Unknown metric type: {metric_def.metric_type}")
                return
            
            self._metrics[metric_def.name] = metric
            self._logger.debug(f"Registered metric: {metric_def.name}")
            
        except Exception as e:
            self._logger.error(f"Failed to register metric {metric_def.name}: {str(e)}")
    
    def record_kafka_message_published(self, topic: str, client_id: str, status: str, duration_seconds: float):
        """Record Kafka message publishing metrics."""
        if not self._enabled:
            return
        
        try:
            # Increment message counter
            counter = self._metrics.get("kafka_messages_published_total")
            if counter:
                counter.labels(topic=topic, status=status, client_id=client_id).inc()
            
            # Record duration
            histogram = self._metrics.get("kafka_publish_duration_seconds")
            if histogram:
                histogram.labels(topic=topic, client_id=client_id).observe(duration_seconds)
                
        except Exception as e:
            self._logger.error(f"Failed to record Kafka metrics: {str(e)}")
    
    def record_payload_encrypted(self, topic: str, client_id: str):
        """Record payload encryption metrics."""
        if not self._enabled:
            return
        
        try:
            counter = self._metrics.get("kafka_payload_encrypted_total")
            if counter:
                counter.labels(topic=topic, client_id=client_id).inc()
        except Exception as e:
            self._logger.error(f"Failed to record encryption metrics: {str(e)}")
    
    def set_circuit_breaker_state(self, client_id: str, state: str):
        """Set circuit breaker state metric."""
        if not self._enabled:
            return
        
        try:
            gauge = self._metrics.get("kafka_circuit_breaker_state")
            if gauge:
                # Map state to numeric value
                state_values = {"closed": 0, "half_open": 1, "open": 2}
                gauge.labels(client_id=client_id).set(state_values.get(state, -1))
        except Exception as e:
            self._logger.error(f"Failed to set circuit breaker metrics: {str(e)}")
    
    def record_rate_limit_violation(self, client_id: str):
        """Record rate limit violation."""
        if not self._enabled:
            return
        
        try:
            counter = self._metrics.get("kafka_rate_limit_violations_total")
            if counter:
                counter.labels(client_id=client_id).inc()
        except Exception as e:
            self._logger.error(f"Failed to record rate limit metrics: {str(e)}")
    
    def set_database_connections(self, database: str, schema: str, count: int):
        """Set database connection count."""
        if not self._enabled:
            return
        
        try:
            gauge = self._metrics.get("database_connections_active")
            if gauge:
                gauge.labels(database=database, schema=schema).set(count)
        except Exception as e:
            self._logger.error(f"Failed to set database connection metrics: {str(e)}")
    
    def record_database_query(self, operation: str, table: str, status: str, duration_seconds: float):
        """Record database query metrics."""
        if not self._enabled:
            return
        
        try:
            # Increment query counter
            counter = self._metrics.get("database_queries_total")
            if counter:
                counter.labels(operation=operation, table=table, status=status).inc()
            
            # Record duration
            histogram = self._metrics.get("database_query_duration_seconds")
            if histogram:
                histogram.labels(operation=operation, table=table, status=status).observe(duration_seconds)
                
        except Exception as e:
            self._logger.error(f"Failed to record database metrics: {str(e)}")
    
    def set_outbox_pending_events(self, schema: str, count: int):
        """Set number of pending outbox events."""
        if not self._enabled:
            return
        
        try:
            gauge = self._metrics.get("outbox_events_pending")
            if gauge:
                gauge.labels(schema=schema).set(count)
        except Exception as e:
            self._logger.error(f"Failed to set outbox pending metrics: {str(e)}")
    
    def record_outbox_event_processed(self, schema: str, status: str):
        """Record processed outbox event."""
        if not self._enabled:
            return
        
        try:
            counter = self._metrics.get("outbox_events_processed_total")
            if counter:
                counter.labels(schema=schema, status=status).inc()
        except Exception as e:
            self._logger.error(f"Failed to record outbox processed metrics: {str(e)}")
    
    def record_outbox_processing(self, schema: str, duration_seconds: float, batch_size: int):
        """Record outbox processing metrics."""
        if not self._enabled:
            return
        
        try:
            # Record processing duration
            duration_histogram = self._metrics.get("outbox_processing_duration_seconds")
            if duration_histogram:
                duration_histogram.labels(schema=schema).observe(duration_seconds)
            
            # Record batch size
            batch_histogram = self._metrics.get("outbox_batch_size")
            if batch_histogram:
                batch_histogram.labels(schema=schema).observe(batch_size)
                
        except Exception as e:
            self._logger.error(f"Failed to record outbox processing metrics: {str(e)}")
    
    def record_security_violation(self, violation_type: str, client_id: str):
        """Record security violation."""
        if not self._enabled:
            return
        
        try:
            counter = self._metrics.get("security_violations_total")
            if counter:
                counter.labels(violation_type=violation_type, client_id=client_id).inc()
        except Exception as e:
            self._logger.error(f"Failed to record security violation metrics: {str(e)}")
    
    def record_audit_event(self, event_type: str, severity: str):
        """Record audit event."""
        if not self._enabled:
            return
        
        try:
            counter = self._metrics.get("audit_events_total")
            if counter:
                counter.labels(event_type=event_type, severity=severity).inc()
        except Exception as e:
            self._logger.error(f"Failed to record audit event metrics: {str(e)}")
    
    def get_metrics_text(self) -> str:
        """
        Get metrics in Prometheus text format.
        
        Returns:
            Metrics in Prometheus exposition format
        """
        if not self._enabled:
            return "# Prometheus metrics disabled\n"
        
        try:
            return generate_latest(self._registry).decode('utf-8')
        except Exception as e:
            self._logger.error(f"Failed to generate metrics: {str(e)}")
            return f"# Error generating metrics: {str(e)}\n"
    
    def start_metrics_server(self, port: int = 8000) -> bool:
        """
        Start HTTP metrics server.
        
        Args:
            port: Port to serve metrics on
            
        Returns:
            True if server started successfully
        """
        if not self._enabled:
            self._logger.warning("Cannot start metrics server - Prometheus not available")
            return False
        
        try:
            start_http_server(port, registry=self._registry)
            self._logger.info(f"Prometheus metrics server started on port {port}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to start metrics server: {str(e)}")
            return False
    
    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._enabled
    
    def get_registry(self) -> Optional[CollectorRegistry]:
        """Get the Prometheus registry."""
        return self._registry if self._enabled else None


# Global metrics instance
_metrics_collector: Optional[ONEXPrometheusMetrics] = None


def get_metrics_collector() -> ONEXPrometheusMetrics:
    """
    Get global metrics collector instance.
    
    Returns:
        ONEXPrometheusMetrics singleton instance
    """
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = ONEXPrometheusMetrics()
    
    return _metrics_collector


def initialize_metrics_server(port: int = 8000) -> bool:
    """
    Initialize and start the metrics server.
    
    Args:
        port: Port to serve metrics on
        
    Returns:
        True if server started successfully
    """
    metrics = get_metrics_collector()
    return metrics.start_metrics_server(port)