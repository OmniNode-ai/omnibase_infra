"""Distributed Tracing Integration for PostgreSQL-RedPanda Event Bus.

OpenTelemetry-based distributed tracing integration that provides end-to-end
trace correlation through the entire RedPanda → PostgreSQL → Event Bus flow.
Integrates with existing audit logging and ONEX protocol-based interfaces.

Following ONEX infrastructure observability patterns with strongly typed context.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, Union, AsyncIterator
from uuid import UUID, uuid4
from datetime import datetime

# OpenTelemetry imports
try:
    from opentelemetry import trace, context, baggage, propagate
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.kafka import KafkaInstrumentor
    from opentelemetry.trace.status import Status, StatusCode
    from opentelemetry.trace import Span, SpanKind
    from opentelemetry.context import Context
    
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Provide stub implementations for when OpenTelemetry is not available
    trace = None
    context = None

from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.errors.onex_error import CoreErrorCode
from omnibase_core.model.core.model_onex_event import ModelOnexEvent

from ..security.audit_logger import AuditLogger, AuditEvent, AuditEventType, AuditSeverity


class TracingConfiguration:
    """Configuration for distributed tracing integration."""
    
    def __init__(self, environment: Optional[str] = None):
        """Initialize tracing configuration.
        
        Args:
            environment: Target environment for configuration
        """
        self.environment = environment or self._detect_environment()
        self.service_name = "omnibase_infrastructure"
        self.service_version = "1.0.0"
        
        # OpenTelemetry configuration
        self.otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        self.otlp_headers = self._parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""))
        
        # Sampling configuration (environment-specific)
        self.trace_sample_rate = self._get_sample_rate()
        
        # Feature flags
        self.enable_db_instrumentation = os.getenv("OTEL_ENABLE_DB_INSTRUMENTATION", "true").lower() == "true"
        self.enable_kafka_instrumentation = os.getenv("OTEL_ENABLE_KAFKA_INSTRUMENTATION", "true").lower() == "true"
        self.enable_audit_integration = os.getenv("OTEL_ENABLE_AUDIT_INTEGRATION", "true").lower() == "true"
        
    def _detect_environment(self) -> str:
        """Detect current deployment environment."""
        env_vars = ["ENVIRONMENT", "ENV", "DEPLOYMENT_ENV", "NODE_ENV"]
        for var in env_vars:
            value = os.getenv(var)
            if value:
                return value.lower()
        return "development"
    
    def _get_sample_rate(self) -> float:
        """Get environment-specific trace sampling rate."""
        sample_rates = {
            "production": 0.1,    # 10% sampling in production
            "staging": 0.5,       # 50% sampling in staging
            "development": 1.0    # 100% sampling in development
        }
        
        rate = float(os.getenv("OTEL_TRACE_SAMPLE_RATE", sample_rates.get(self.environment, 1.0)))
        return max(0.0, min(1.0, rate))  # Clamp between 0 and 1
    
    def _parse_headers(self, headers_str: str) -> Dict[str, str]:
        """Parse OTLP headers from environment variable."""
        headers = {}
        if headers_str:
            for header in headers_str.split(","):
                if "=" in header:
                    key, value = header.split("=", 1)
                    headers[key.strip()] = value.strip()
        return headers


class DistributedTracingManager:
    """
    Distributed tracing manager for ONEX infrastructure.
    
    Provides:
    - OpenTelemetry integration with automatic instrumentation
    - Trace context propagation through event envelopes
    - Integration with existing audit logging system
    - ONEX protocol-based tracing interfaces
    - Environment-specific configuration management
    """
    
    def __init__(self, config: Optional[TracingConfiguration] = None):
        """Initialize distributed tracing manager.
        
        Args:
            config: Tracing configuration (optional, auto-detected if None)
        """
        self.config = config or TracingConfiguration()
        self.logger = logging.getLogger(f"{__name__}.DistributedTracingManager")
        
        # Tracing components
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[Any] = None  # OpenTelemetry tracer
        self.is_initialized = False
        
        # Integration with audit logging
        self.audit_logger: Optional[AuditLogger] = None
        
        # Check OpenTelemetry availability
        if not OPENTELEMETRY_AVAILABLE:
            self.logger.warning("OpenTelemetry not available - tracing will be disabled")
    
    async def initialize(self) -> None:
        """Initialize distributed tracing infrastructure."""
        if self.is_initialized or not OPENTELEMETRY_AVAILABLE:
            return
        
        try:
            # Create resource with service information
            resource = Resource.create({
                SERVICE_NAME: self.config.service_name,
                SERVICE_VERSION: self.config.service_version,
                "deployment.environment": self.config.environment,
                "service.namespace": "omnibase_infrastructure"
            })
            
            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            
            # Configure OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                headers=self.config.otlp_headers
            )
            
            # Add batch span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            
            # Set global tracer provider
            trace.set_tracer_provider(self.tracer_provider)
            
            # Get tracer
            self.tracer = trace.get_tracer(
                instrumenting_module_name=__name__,
                instrumenting_library_version=self.config.service_version
            )
            
            # Initialize automatic instrumentation
            await self._initialize_instrumentation()
            
            # Initialize audit integration
            if self.config.enable_audit_integration:
                self.audit_logger = AuditLogger()
            
            self.is_initialized = True
            self.logger.info(f"Distributed tracing initialized for environment: {self.config.environment}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed tracing: {e}")
            raise OnexError(
                code=CoreErrorCode.CONFIGURATION_ERROR,
                message=f"Distributed tracing initialization failed: {str(e)}"
            ) from e
    
    async def _initialize_instrumentation(self) -> None:
        """Initialize automatic instrumentation for databases and messaging."""
        try:
            # PostgreSQL instrumentation
            if self.config.enable_db_instrumentation:
                AsyncPGInstrumentor().instrument()
                self.logger.info("PostgreSQL instrumentation enabled")
            
            # Kafka instrumentation  
            if self.config.enable_kafka_instrumentation:
                KafkaInstrumentor().instrument()
                self.logger.info("Kafka instrumentation enabled")
                
        except Exception as e:
            self.logger.warning(f"Some automatic instrumentations failed: {e}")
    
    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        correlation_id: Optional[Union[str, UUID]] = None,
        parent_context: Optional[Context] = None,
        span_kind: Optional[SpanKind] = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Span]:
        """
        Create a trace span for an operation with automatic error handling.
        
        Args:
            operation_name: Name of the operation being traced
            correlation_id: Correlation ID for the operation
            parent_context: Parent trace context (optional)
            span_kind: OpenTelemetry span kind
            attributes: Additional span attributes
            
        Yields:
            Active span for the operation
        """
        if not self.is_initialized or not OPENTELEMETRY_AVAILABLE:
            # Return a no-op span when tracing is disabled
            yield self._create_noop_span()
            return
        
        # Ensure correlation ID is string
        correlation_str = str(correlation_id) if correlation_id else str(uuid4())
        
        # Set parent context if provided
        if parent_context:
            token = context.attach(parent_context)
        else:
            token = None
        
        try:
            # Create span
            span = self.tracer.start_span(
                name=operation_name,
                kind=span_kind,
                attributes={
                    "correlation_id": correlation_str,
                    "environment": self.config.environment,
                    "service.name": self.config.service_name,
                    **(attributes or {})
                }
            )
            
            # Set correlation ID in baggage for propagation
            baggage.set_baggage("correlation_id", correlation_str)
            
            # Activate span context
            with trace.use_span(span, end_on_exit=False):
                try:
                    yield span
                    
                    # Mark span as successful
                    span.set_status(Status(StatusCode.OK))
                    
                    # Log audit event for successful operation
                    if self.audit_logger and self.config.enable_audit_integration:
                        await self._log_trace_audit_event(
                            operation_name, correlation_str, "success", span
                        )
                        
                except Exception as e:
                    # Mark span as failed and record exception
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    
                    # Log audit event for failed operation
                    if self.audit_logger and self.config.enable_audit_integration:
                        await self._log_trace_audit_event(
                            operation_name, correlation_str, "failure", span, error=str(e)
                        )
                    
                    raise
                finally:
                    span.end()
        finally:
            if token:
                context.detach(token)
    
    def _create_noop_span(self) -> object:
        """Create a no-op span when tracing is disabled."""
        class NoOpSpan:
            def set_attribute(self, key: str, value: Union[str, int, float, bool]) -> None:
                pass
            def set_status(self, status: object) -> None:
                pass
            def record_exception(self, exception: Exception) -> None:
                pass
            def end(self) -> None:
                pass
        
        return NoOpSpan()
    
    def inject_trace_context(self, event: ModelOnexEvent) -> ModelOnexEvent:
        """Inject current trace context into an event envelope.
        
        Args:
            event: Event envelope to inject trace context into
            
        Returns:
            Event with trace context injected into metadata
        """
        if not self.is_initialized or not OPENTELEMETRY_AVAILABLE:
            return event
        
        try:
            # Create a carrier for trace context propagation
            carrier = {}
            propagate.inject(carrier)
            
            # Add trace context to event metadata
            if not hasattr(event, 'metadata') or event.metadata is None:
                event.metadata = {}
            
            event.metadata.update({
                "trace_context": carrier,
                "trace_timestamp": datetime.now().isoformat(),
                "trace_service": self.config.service_name,
                "trace_environment": self.config.environment
            })
            
            return event
            
        except Exception as e:
            self.logger.warning(f"Failed to inject trace context: {e}")
            return event
    
    def extract_trace_context(self, event: ModelOnexEvent) -> Optional[Context]:
        """Extract trace context from an event envelope.
        
        Args:
            event: Event envelope to extract trace context from
            
        Returns:
            Extracted trace context or None if not available
        """
        if not self.is_initialized or not OPENTELEMETRY_AVAILABLE:
            return None
        
        try:
            if not hasattr(event, 'metadata') or not event.metadata:
                return None
            
            trace_context_data = event.metadata.get("trace_context")
            if not trace_context_data:
                return None
            
            # Extract context from carrier
            return propagate.extract(trace_context_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract trace context: {e}")
            return None
    
    async def trace_database_operation(
        self,
        operation_type: str,
        query: Optional[str] = None,
        correlation_id: Optional[Union[str, UUID]] = None,
        parent_context: Optional[Context] = None
    ) -> AsyncIterator[Span]:
        """Trace a database operation with database-specific attributes.
        
        Args:
            operation_type: Type of database operation (query, transaction, etc.)
            query: SQL query (optional, will be sanitized)
            correlation_id: Correlation ID for the operation
            parent_context: Parent trace context
        """
        attributes = {
            "db.system": "postgresql",
            "db.operation": operation_type,
            "component": "postgres_adapter"
        }
        
        # Add sanitized query if provided
        if query:
            # Sanitize query to remove sensitive data
            sanitized_query = self._sanitize_query(query)
            attributes["db.statement"] = sanitized_query
        
        async with self.trace_operation(
            operation_name=f"postgres.{operation_type}",
            correlation_id=correlation_id,
            parent_context=parent_context,
            span_kind=SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            yield span
    
    async def trace_kafka_operation(
        self,
        operation_type: str,
        topic: Optional[str] = None,
        correlation_id: Optional[Union[str, UUID]] = None,
        parent_context: Optional[Context] = None
    ) -> AsyncIterator[Span]:
        """Trace a Kafka operation with messaging-specific attributes.
        
        Args:
            operation_type: Type of Kafka operation (produce, consume, etc.)
            topic: Kafka topic name
            correlation_id: Correlation ID for the operation
            parent_context: Parent trace context
        """
        attributes = {
            "messaging.system": "kafka",
            "messaging.operation": operation_type,
            "component": "kafka_adapter"
        }
        
        if topic:
            attributes["messaging.destination"] = topic
            attributes["messaging.destination_kind"] = "topic"
        
        async with self.trace_operation(
            operation_name=f"kafka.{operation_type}",
            correlation_id=correlation_id,
            parent_context=parent_context,
            span_kind=SpanKind.PRODUCER if operation_type == "produce" else SpanKind.CONSUMER,
            attributes=attributes
        ) as span:
            yield span
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize SQL query to remove sensitive data."""
        # Basic sanitization - in production, use more sophisticated approach
        import re
        
        # Remove potential passwords, keys, etc.
        sanitized = re.sub(r"'[^']*'", "'***'", query)
        sanitized = re.sub(r'"[^"]*"', '"***"', sanitized)
        
        # Truncate very long queries
        if len(sanitized) > 200:
            sanitized = sanitized[:197] + "..."
        
        return sanitized
    
    async def _log_trace_audit_event(
        self,
        operation_name: str,
        correlation_id: str,
        outcome: str,
        span: Span,
        error: Optional[str] = None
    ) -> None:
        """Log audit event for traced operation."""
        if not self.audit_logger:
            return
        
        try:
            # Get trace and span IDs
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x') if span_context else None
            span_id = format(span_context.span_id, '016x') if span_context else None
            
            audit_event = AuditEvent(
                event_id="",  # Will be auto-generated
                timestamp="",  # Will be auto-generated
                event_type=AuditEventType.SYSTEM_ERROR if outcome == "failure" else AuditEventType.DATA_ACCESS,
                severity=AuditSeverity.HIGH if outcome == "failure" else AuditSeverity.LOW,
                user_id=None,  # System operation
                client_id="infrastructure_tracing",
                session_id=None,
                correlation_id=correlation_id,
                resource=operation_name,
                action="trace_operation",
                outcome=outcome,
                details={
                    "operation_name": operation_name,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "environment": self.config.environment,
                    "error_message": error if error else None
                }
            )
            
            await self.audit_logger.log_event(audit_event)
            
        except Exception as e:
            self.logger.warning(f"Failed to log trace audit event: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown distributed tracing and flush pending spans."""
        if not self.is_initialized or not OPENTELEMETRY_AVAILABLE:
            return
        
        try:
            if self.tracer_provider:
                # Force flush pending spans
                await asyncio.to_thread(self.tracer_provider.force_flush, timeout_millis=5000)
            
            self.is_initialized = False
            self.logger.info("Distributed tracing shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during tracing shutdown: {e}")


# Global tracing manager instance
_tracing_manager: Optional[DistributedTracingManager] = None


def get_tracing_manager(config: Optional[TracingConfiguration] = None) -> DistributedTracingManager:
    """Get the global distributed tracing manager instance."""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = DistributedTracingManager(config)
    return _tracing_manager


async def initialize_distributed_tracing(config: Optional[TracingConfiguration] = None) -> None:
    """Initialize global distributed tracing."""
    manager = get_tracing_manager(config)
    await manager.initialize()


async def shutdown_distributed_tracing() -> None:
    """Shutdown global distributed tracing."""
    global _tracing_manager
    if _tracing_manager:
        await _tracing_manager.shutdown()
        _tracing_manager = None