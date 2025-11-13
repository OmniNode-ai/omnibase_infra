"""
OpenTelemetry integration for MetadataStampingService.

Provides comprehensive distributed tracing, metrics collection, and observability
for microservice deployments with automatic instrumentation and custom metrics.
"""

import asyncio
import logging
import os
import platform
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import psutil
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from prometheus_client import start_http_server

# Optional OTLP exporters (may not be available in all environments)
try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "OTLP exporters not available - install opentelemetry-exporter-otlp for OTLP support"
    )

# Optional instrumentation libraries
try:
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    INSTRUMENTATION_AVAILABLE = True
except ImportError:
    INSTRUMENTATION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenTelemetry instrumentation libraries not available")

# Optional propagators
try:
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.propagators.composite import CompositePropagator

    PROPAGATORS_AVAILABLE = True
except ImportError:
    PROPAGATORS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenTelemetry propagators not available")

logger = logging.getLogger(__name__)


@dataclass
class TelemetryConfig:
    """Configuration for OpenTelemetry setup."""

    service_name: str = "metadata-stamping"
    service_version: str = "3.0.0"
    environment: str = "production"

    # OTLP configuration (replaces deprecated Jaeger exporter)
    otlp_endpoint: Optional[str] = None
    otlp_insecure: bool = False
    otlp_headers: dict[str, str] = field(default_factory=dict)

    # Prometheus configuration
    prometheus_port: int = 9090
    prometheus_endpoint: str = "/metrics"

    # Sampling configuration
    sampling_ratio: float = 1.0

    # Resource attributes
    cluster_name: str = "metadata-stamping-cluster"
    namespace: str = "metadata-stamping"
    region: str = "us-west-2"
    availability_zone: str = "us-west-2a"

    # Instrumentation settings
    instrument_asyncpg: bool = True
    instrument_redis: bool = True
    instrument_http_client: bool = True
    instrument_fastapi: bool = True
    instrument_logging: bool = True

    # Custom settings
    enable_custom_metrics: bool = True
    enable_runtime_metrics: bool = True
    metrics_export_interval: int = 30


class MetricsCollector:
    """Collects custom metrics for the metadata stamping service."""

    def __init__(self, meter_provider: MeterProvider):
        self.meter = meter_provider.get_meter("metadata_stamping_metrics")

        # Track current values for observable gauges
        self._current_pool_sizes: dict[str, int] = {}
        self._current_memory: int = 0
        self._current_cpu: float = 0.0
        self._current_shard_health: dict[str, int] = {}
        self._current_circuit_breaker_states: dict[str, int] = {}

        # Performance metrics
        self.hash_duration_histogram = self.meter.create_histogram(
            name="blake3_hash_duration_seconds",
            description="Time taken to generate BLAKE3 hashes",
            unit="s",
        )

        self.hash_operations_counter = self.meter.create_counter(
            name="blake3_hash_operations_total",
            description="Total number of hash operations",
        )

        # API metrics
        self.stamp_requests_counter = self.meter.create_counter(
            name="stamp_requests_total", description="Total number of stamp requests"
        )

        self.stamp_duration_histogram = self.meter.create_histogram(
            name="stamp_duration_seconds",
            description="Time taken to process stamp requests",
            unit="s",
        )

        # Database metrics
        self.db_operation_duration_histogram = self.meter.create_histogram(
            name="database_operation_duration_seconds",
            description="Database operation duration",
            unit="s",
        )

        # Observable gauge for connection pool size (absolute value)
        self.db_connection_pool_gauge = self.meter.create_observable_gauge(
            name="database_connection_pool_size",
            description="Current database connection pool size",
            callbacks=[self._observe_connection_pool_sizes],
        )

        # Observable gauge for shard health (absolute value)
        self.db_shard_health_gauge = self.meter.create_observable_gauge(
            name="database_shard_health",
            description="Database shard health status (1=healthy, 0=unhealthy)",
            callbacks=[self._observe_shard_health],
        )

        # Observable gauge for circuit breaker state (absolute value)
        self.circuit_breaker_state_gauge = self.meter.create_observable_gauge(
            name="circuit_breaker_state",
            description="Circuit breaker state (0=closed, 1=open, 2=half_open)",
            callbacks=[self._observe_circuit_breaker_states],
        )

        self.circuit_breaker_requests_counter = self.meter.create_counter(
            name="circuit_breaker_requests_total",
            description="Total circuit breaker requests",
        )

        # Observable gauges for system metrics (absolute values)
        self.memory_usage_gauge = self.meter.create_observable_gauge(
            name="process_memory_usage_bytes",
            description="Process memory usage in bytes",
            callbacks=[self._observe_memory_usage],
        )

        self.cpu_usage_gauge = self.meter.create_observable_gauge(
            name="process_cpu_usage_percent",
            description="Process CPU usage percentage",
            callbacks=[self._observe_cpu_usage],
        )

        # Redis metrics
        self.redis_operation_duration_histogram = self.meter.create_histogram(
            name="redis_operation_duration_seconds",
            description="Redis operation duration",
            unit="s",
        )

        self.redis_connection_errors_counter = self.meter.create_counter(
            name="redis_connection_errors_total",
            description="Total Redis connection errors",
        )

    def _observe_connection_pool_sizes(self, options):
        """Callback for observing connection pool sizes."""
        from opentelemetry.metrics import Observation

        observations = []
        for shard_id, size in self._current_pool_sizes.items():
            observations.append(Observation(size, {"shard_id": shard_id}))
        return observations

    def _observe_shard_health(self, options):
        """Callback for observing shard health status."""
        from opentelemetry.metrics import Observation

        observations = []
        for shard_id, health in self._current_shard_health.items():
            observations.append(Observation(health, {"shard_id": shard_id}))
        return observations

    def _observe_circuit_breaker_states(self, options):
        """Callback for observing circuit breaker states."""
        from opentelemetry.metrics import Observation

        observations = []
        for name, state in self._current_circuit_breaker_states.items():
            observations.append(Observation(state, {"circuit_breaker": name}))
        return observations

    def _observe_memory_usage(self, options):
        """Callback for observing memory usage."""
        from opentelemetry.metrics import Observation

        return [Observation(self._current_memory)]

    def _observe_cpu_usage(self, options):
        """Callback for observing CPU usage."""
        from opentelemetry.metrics import Observation

        return [Observation(self._current_cpu)]

    def record_hash_operation(self, duration: float, file_size: int, grade: str):
        """Record hash operation metrics."""
        self.hash_duration_histogram.record(
            duration, {"file_size_category": self._get_file_size_category(file_size)}
        )
        self.hash_operations_counter.add(
            1, {"file_size_category": self._get_file_size_category(file_size)}
        )

    def record_stamp_request(self, duration: float, status_code: int, operation: str):
        """Record stamp request metrics."""
        self.stamp_requests_counter.add(
            1, {"status_code": str(status_code), "operation": operation}
        )
        self.stamp_duration_histogram.record(
            duration, {"status_code": str(status_code), "operation": operation}
        )

    def record_database_operation(self, duration: float, shard_id: str, operation: str):
        """Record database operation metrics."""
        self.db_operation_duration_histogram.record(
            duration, {"shard_id": shard_id, "operation": operation}
        )

    def update_connection_pool_size(self, shard_id: str, size: int):
        """Update database connection pool size with absolute value."""
        self._current_pool_sizes[shard_id] = size

    def update_shard_health(self, shard_id: str, healthy: bool):
        """Update shard health status with absolute value."""
        self._current_shard_health[shard_id] = 1 if healthy else 0

    def update_circuit_breaker_state(self, name: str, state: int):
        """Update circuit breaker state with absolute value."""
        self._current_circuit_breaker_states[name] = state

    def record_circuit_breaker_request(self, name: str, allowed: bool):
        """Record circuit breaker request."""
        self.circuit_breaker_requests_counter.add(
            1, {"circuit_breaker": name, "allowed": str(allowed)}
        )

    def _get_file_size_category(self, size: int) -> str:
        """Categorize file size for metrics."""
        if size <= 1024:  # 1KB
            return "small"
        elif size <= 1024 * 1024:  # 1MB
            return "medium"
        elif size <= 10 * 1024 * 1024:  # 10MB
            return "large"
        else:
            return "xlarge"


class RuntimeMetricsCollector:
    """Collects runtime metrics about the service."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.process = psutil.Process()
        self.collection_task: Optional[asyncio.Task] = None
        self.running = False

    async def start_collection(self, interval: int = 30):
        """Start collecting runtime metrics."""
        if self.running:
            return

        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop(interval))
        logger.info("Started runtime metrics collection")

    async def stop_collection(self):
        """Stop collecting runtime metrics."""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped runtime metrics collection")

    async def _collection_loop(self, interval: int):
        """Main collection loop."""
        while self.running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting runtime metrics: {e}")
                await asyncio.sleep(5)

    async def _collect_metrics(self):
        """Collect current runtime metrics."""
        if self.metrics_collector is None:
            return

        # Memory metrics - update absolute value
        memory_info = self.process.memory_info()
        self.metrics_collector._current_memory = memory_info.rss

        # CPU metrics - update absolute value
        cpu_percent = self.process.cpu_percent()
        self.metrics_collector._current_cpu = cpu_percent


class TraceEnricher:
    """Enriches traces with additional context and metadata."""

    @staticmethod
    def enrich_span_with_request_info(span: trace.Span, request_data: dict[str, Any]):
        """Enrich span with HTTP request information."""
        span.set_attribute("http.method", request_data.get("method", ""))
        span.set_attribute("http.url", request_data.get("url", ""))
        span.set_attribute("http.user_agent", request_data.get("user_agent", ""))
        span.set_attribute("http.request_size", request_data.get("content_length", 0))

        # Add custom metadata stamping attributes
        if "file_size" in request_data:
            span.set_attribute("metadata_stamping.file_size", request_data["file_size"])
        if "file_type" in request_data:
            span.set_attribute("metadata_stamping.file_type", request_data["file_type"])
        if "hash_algorithm" in request_data:
            span.set_attribute(
                "metadata_stamping.hash_algorithm", request_data["hash_algorithm"]
            )

    @staticmethod
    def enrich_span_with_database_info(span: trace.Span, shard_id: str, operation: str):
        """Enrich span with database operation information."""
        span.set_attribute("db.shard_id", shard_id)
        span.set_attribute("db.operation", operation)
        span.set_attribute("db.system", "postgresql")

    @staticmethod
    def enrich_span_with_performance_info(
        span: trace.Span, performance_data: dict[str, Any]
    ):
        """Enrich span with performance metrics."""
        if "execution_time_ms" in performance_data:
            span.set_attribute(
                "metadata_stamping.execution_time_ms",
                performance_data["execution_time_ms"],
            )
        if "performance_grade" in performance_data:
            span.set_attribute(
                "metadata_stamping.performance_grade",
                performance_data["performance_grade"],
            )
        if "cpu_usage_percent" in performance_data:
            span.set_attribute(
                "metadata_stamping.cpu_usage_percent",
                performance_data["cpu_usage_percent"],
            )


class TelemetryManager:
    """Manages OpenTelemetry setup and configuration."""

    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.runtime_collector: Optional[RuntimeMetricsCollector] = None
        self._initialized = False

    def initialize(self):
        """Initialize OpenTelemetry with configuration."""
        if self._initialized:
            return

        logger.info("Initializing OpenTelemetry telemetry")

        # Create resource with service information
        resource = self._create_resource()

        # Setup tracing
        self._setup_tracing(resource)

        # Setup metrics
        self._setup_metrics(resource)

        # Setup propagators
        self._setup_propagators()

        # Setup instrumentation
        self._setup_instrumentation()

        # Setup custom metrics collection
        if self.config.enable_custom_metrics and self.meter_provider is not None:
            self.metrics_collector = MetricsCollector(self.meter_provider)

        # Setup runtime metrics collection (requires metrics_collector)
        if self.config.enable_runtime_metrics and self.metrics_collector is not None:
            self.runtime_collector = RuntimeMetricsCollector(self.metrics_collector)

        self._initialized = True
        logger.info("OpenTelemetry telemetry initialized successfully")

    def _create_resource(self) -> Resource:
        """Create resource with service metadata."""
        # Get environment information
        hostname = socket.gethostname()
        pod_name = os.getenv("POD_NAME", hostname)
        pod_namespace = os.getenv("POD_NAMESPACE", self.config.namespace)
        node_name = os.getenv("NODE_NAME", hostname)
        pod_ip = os.getenv("POD_IP", "")

        resource_attributes = {
            ResourceAttributes.SERVICE_NAME: self.config.service_name,
            ResourceAttributes.SERVICE_VERSION: self.config.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
            ResourceAttributes.HOST_NAME: hostname,
            ResourceAttributes.OS_TYPE: platform.system(),
            ResourceAttributes.OS_VERSION: platform.release(),
            ResourceAttributes.PROCESS_PID: os.getpid(),
            "k8s.cluster.name": self.config.cluster_name,
            "k8s.namespace.name": pod_namespace,
            "k8s.pod.name": pod_name,
            "k8s.node.name": node_name,
            "k8s.pod.ip": pod_ip,
            "cloud.region": self.config.region,
            "cloud.availability_zone": self.config.availability_zone,
        }

        return Resource.create(resource_attributes)

    def _setup_tracing(self, resource: Resource):
        """Setup distributed tracing."""
        self.tracer_provider = TracerProvider(resource=resource)

        # Setup OTLP exporter (replaces deprecated Jaeger exporter)
        if self.config.otlp_endpoint and OTLP_AVAILABLE:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=self.config.otlp_insecure,
                headers=self.config.otlp_headers,
            )
            self.tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        elif self.config.otlp_endpoint and not OTLP_AVAILABLE:
            logger.warning("OTLP endpoint configured but OTLP exporters not available")

        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)

    def _setup_metrics(self, resource: Resource):
        """Setup metrics collection."""
        readers = []

        # OTLP metrics (Prometheus metrics are handled by prometheus_client directly)
        if self.config.otlp_endpoint and OTLP_AVAILABLE:
            otlp_metric_exporter = OTLPMetricExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=self.config.otlp_insecure,
                headers=self.config.otlp_headers,
            )
            periodic_reader = PeriodicExportingMetricReader(
                exporter=otlp_metric_exporter,
                export_interval_millis=self.config.metrics_export_interval * 1000,
            )
            readers.append(periodic_reader)

            # Setup MeterProvider only when we have OTLP endpoint configured
            self.meter_provider = MeterProvider(
                resource=resource, metric_readers=readers
            )
            # Set global meter provider
            metrics.set_meter_provider(self.meter_provider)
            logger.info("OpenTelemetry MeterProvider configured with OTLP exporter")
        elif self.config.otlp_endpoint and not OTLP_AVAILABLE:
            logger.warning("OTLP endpoint configured but OTLP exporters not available")
        else:
            logger.info(
                "No OTLP endpoint configured, MeterProvider not created. Using prometheus_client only."
            )

        # Start Prometheus HTTP server (independent of OTLP metrics)
        try:
            start_http_server(self.config.prometheus_port)
            logger.info(
                f"Prometheus metrics server started on port {self.config.prometheus_port}"
            )
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")

    def _setup_propagators(self):
        """Setup trace context propagation."""
        if not PROPAGATORS_AVAILABLE:
            logger.info("Propagators not available, skipping propagator setup")
            return

        # Use B3 propagation (W3C TraceContext is the default)
        propagators = [
            B3MultiFormat(),
        ]

        set_global_textmap(CompositePropagator(propagators))

    def _setup_instrumentation(self):
        """Setup automatic instrumentation."""
        if not INSTRUMENTATION_AVAILABLE:
            logger.info(
                "Instrumentation libraries not available, skipping instrumentation setup"
            )
            return

        try:
            if self.config.instrument_fastapi:
                FastAPIInstrumentor.instrument()

            if self.config.instrument_asyncpg:
                AsyncPGInstrumentor.instrument()

            if self.config.instrument_redis:
                RedisInstrumentor.instrument()

            # Use AioHttpClientInstrumentor for async HTTP client instrumentation
            if self.config.instrument_http_client:
                AioHttpClientInstrumentor().instrument()

            logger.info("Automatic instrumentation setup completed")

        except Exception as e:
            logger.error(f"Error setting up instrumentation: {e}")

    async def start_runtime_collection(self):
        """Start runtime metrics collection."""
        if self.runtime_collector:
            await self.runtime_collector.start_collection(
                interval=self.config.metrics_export_interval
            )

    async def stop_runtime_collection(self):
        """Stop runtime metrics collection."""
        if self.runtime_collector:
            await self.runtime_collector.stop_collection()

    def get_tracer(self, name: str = None) -> trace.Tracer:
        """Get a tracer instance."""
        if not self._initialized:
            raise RuntimeError("Telemetry not initialized")

        return trace.get_tracer(name or self.config.service_name)

    def get_meter(self, name: str = None) -> metrics.Meter:
        """Get a meter instance."""
        if not self._initialized:
            raise RuntimeError("Telemetry not initialized")

        return metrics.get_meter(name or self.config.service_name)

    def shutdown(self):
        """Shutdown telemetry gracefully."""
        if self.tracer_provider:
            self.tracer_provider.shutdown()

        if self.meter_provider:
            self.meter_provider.shutdown()

        logger.info("Telemetry shutdown completed")


# Global telemetry manager instance
_telemetry_manager: Optional[TelemetryManager] = None


def initialize_telemetry(config: TelemetryConfig = None) -> TelemetryManager:
    """Initialize OpenTelemetry telemetry."""
    global _telemetry_manager

    if _telemetry_manager is not None:
        return _telemetry_manager

    if config is None:
        config = _load_config_from_env()

    _telemetry_manager = TelemetryManager(config)
    _telemetry_manager.initialize()

    return _telemetry_manager


def _load_config_from_env() -> TelemetryConfig:
    """Load telemetry configuration from environment variables."""
    return TelemetryConfig(
        service_name=os.getenv("OTEL_SERVICE_NAME", "metadata-stamping"),
        service_version=os.getenv("OTEL_SERVICE_VERSION", "3.0.0"),
        environment=os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
        cluster_name=os.getenv("CLUSTER_ID", "metadata-stamping-cluster"),
        namespace=os.getenv("POD_NAMESPACE", "metadata-stamping"),
        region=os.getenv("REGION", "us-west-2"),
        availability_zone=os.getenv("AVAILABILITY_ZONE", "us-west-2a"),
    )


def get_tracer(name: str = None) -> trace.Tracer:
    """Get a tracer instance."""
    if _telemetry_manager is None:
        initialize_telemetry()

    return _telemetry_manager.get_tracer(name)


def get_meter(name: str = None) -> metrics.Meter:
    """Get a meter instance."""
    if _telemetry_manager is None:
        initialize_telemetry()

    return _telemetry_manager.get_meter(name)


@contextmanager
def create_span(name: str, **attributes):
    """Create a new span with optional attributes."""
    tracer = get_tracer()

    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span


async def start_telemetry():
    """Start telemetry services."""
    if _telemetry_manager:
        await _telemetry_manager.start_runtime_collection()


async def stop_telemetry():
    """Stop telemetry services."""
    if _telemetry_manager:
        await _telemetry_manager.stop_runtime_collection()
        _telemetry_manager.shutdown()
