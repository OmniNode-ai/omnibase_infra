"""Distributed Tracing Compute Node.

ONEX compute node for OpenTelemetry integration with trace context processing and enrichment.
Provides end-to-end trace correlation through the entire infrastructure flow.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Optional, AsyncIterator, Union
from uuid import UUID

from omnibase_core.base.node_compute_service import NodeComputeService
from omnibase_core.core.errors.onex_error import OnexError, CoreErrorCode
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.models.core.model_onex_event import ModelOnexEvent

# OpenTelemetry imports with availability check
try:
    from opentelemetry import trace, context, baggage, propagate
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.trace.status import Status, StatusCode
    from opentelemetry.trace import Span, SpanKind
    from opentelemetry.context import Context
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Stub implementations when OpenTelemetry not available
    trace = None
    context = None

from .models.model_distributed_tracing_input import (
    ModelDistributedTracingInput,
    TracingOperation,
    SpanKind as InputSpanKind
)
from .models.model_distributed_tracing_output import ModelDistributedTracingOutput
from .config import TracingConfig
from .utils.sql_sanitizer import SqlSanitizer


class NodeDistributedTracingCompute(NodeComputeService[ModelDistributedTracingInput, ModelDistributedTracingOutput]):
    """
    Distributed Tracing Compute Node.
    
    Provides:
    - OpenTelemetry integration with automatic instrumentation
    - Trace context propagation through event envelopes
    - Environment-specific tracing configuration
    - Database and Kafka operation tracing
    - Graceful degradation when OpenTelemetry unavailable
    """
    
    def __init__(self, container: ModelONEXContainer, tracing_config: TracingConfig):
        """Initialize the distributed tracing compute node.

        Args:
            container: ONEX container for dependency injection
            tracing_config: Validated tracing configuration with endpoint validation
        """
        super().__init__(container)
        self.logger = logging.getLogger(f"{__name__}.NodeDistributedTracingCompute")

        # Inject validated configuration following ONEX patterns
        self.tracing_config = tracing_config

        # Tracing components - using Union for proper typing with graceful degradation
        self.tracer_provider: Optional[Union["TracerProvider", object]] = None  # TracerProvider when available
        self.tracer: Optional[Union["trace.Tracer", object]] = None  # OpenTelemetry tracer
        self.is_initialized = False

        # Check OpenTelemetry availability
        if not OPENTELEMETRY_AVAILABLE:
            self.logger.warning("OpenTelemetry not available - tracing will be disabled")
    
    async def initialize(self) -> None:
        """Initialize the distributed tracing node."""
        try:
            # Initialize OpenTelemetry if available
            if OPENTELEMETRY_AVAILABLE:
                await self._initialize_opentelemetry()

            self.logger.info(
                f"Distributed tracing compute node initialized successfully "
                f"(environment: {self.tracing_config.environment}, "
                f"endpoint: {self.tracing_config.otel_exporter_otlp_endpoint})"
            )

        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.INITIALIZATION_ERROR,
                message=f"Failed to initialize distributed tracing compute node: {str(e)}"
            ) from e
    
    async def compute(self, input_data: ModelDistributedTracingInput) -> ModelDistributedTracingOutput:
        """Execute distributed tracing operations.
        
        Args:
            input_data: Input containing tracing operation type and parameters
            
        Returns:
            Output with operation result and tracing status
        """
        try:
            # Route to appropriate operation handler
            if input_data.operation_type == TracingOperation.INITIALIZE_TRACING:
                result = await self._handle_initialize_tracing(input_data)
            elif input_data.operation_type == TracingOperation.TRACE_OPERATION:
                result = await self._handle_trace_operation(input_data)
            elif input_data.operation_type == TracingOperation.INJECT_CONTEXT:
                result = await self._handle_inject_context(input_data)
            elif input_data.operation_type == TracingOperation.EXTRACT_CONTEXT:
                result = await self._handle_extract_context(input_data)
            elif input_data.operation_type == TracingOperation.TRACE_DATABASE:
                result = await self._handle_trace_database(input_data)
            elif input_data.operation_type == TracingOperation.TRACE_KAFKA:
                result = await self._handle_trace_kafka(input_data)
            elif input_data.operation_type == TracingOperation.SHUTDOWN_TRACING:
                result = await self._handle_shutdown_tracing(input_data)
            else:
                raise OnexError(
                    code=CoreErrorCode.INVALID_INPUT,
                    message=f"Unsupported tracing operation type: {input_data.operation_type}"
                )
            
            return ModelDistributedTracingOutput(
                success=True,
                operation_type=input_data.operation_type.value,
                correlation_id=input_data.correlation_id,
                result=result,
                trace_id=result.get("trace_id"),
                span_id=result.get("span_id"),
                tracing_enabled=OPENTELEMETRY_AVAILABLE and self.is_initialized,
                timestamp=datetime.now()
            )
            
        except OnexError:
            # Re-raise ONEX errors as-is
            raise
        except Exception as e:
            # Wrap other exceptions in OnexError
            raise OnexError(
                code=CoreErrorCode.PROCESSING_ERROR,
                message=f"Distributed tracing operation failed: {str(e)}"
            ) from e
    
    async def _handle_initialize_tracing(self, input_data: ModelDistributedTracingInput) -> Dict[str, Union[str, bool, float]]:
        """Handle tracing initialization."""
        if not OPENTELEMETRY_AVAILABLE:
            return {
                "initialized": False,
                "reason": "OpenTelemetry not available",
                "fallback_mode": True
            }
        
        if not self.is_initialized:
            await self._initialize_opentelemetry()
        
        return {
            "initialized": self.is_initialized,
            "service_name": self.tracing_config.service_name,
            "environment": self.tracing_config.environment,
            "otlp_endpoint": str(self.tracing_config.otel_exporter_otlp_endpoint),
            "sample_rate": self.tracing_config.trace_sample_rate
        }
    
    async def _handle_trace_operation(self, input_data: ModelDistributedTracingInput) -> Dict[str, Union[str, bool, Optional[str], Dict[str, str]]]:
        """Handle generic operation tracing."""
        if not self.is_initialized or not input_data.operation_name:
            return {
                "traced": False,
                "reason": "Tracing not initialized or operation name missing"
            }
        
        # Convert span kind
        span_kind = self._convert_span_kind(input_data.span_kind)
        
        # Create span attributes
        attributes = {
            "correlation_id": str(input_data.correlation_id),
            "environment": self.tracing_config.environment,
            "service.name": self.tracing_config.service_name,
            **(input_data.attributes or {})
        }
        
        # Create and manage span
        span = self.tracer.start_span(
            name=input_data.operation_name,
            kind=span_kind,
            attributes=attributes
        )
        
        try:
            # Get trace and span IDs
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x') if span_context else None
            span_id = format(span_context.span_id, '016x') if span_context else None
            
            # Mark span as successful
            span.set_status(Status(StatusCode.OK))
            
            return {
                "traced": True,
                "operation_name": input_data.operation_name,
                "trace_id": trace_id,
                "span_id": span_id,
                "attributes": attributes
            }
            
        finally:
            span.end()
    
    async def _handle_inject_context(self, input_data: ModelDistributedTracingInput) -> Dict[str, Union[str, bool, List[str]]]:
        """Handle trace context injection into event."""
        if not input_data.event or not self.is_initialized:
            return {
                "injected": False,
                "reason": "Event missing or tracing not initialized"
            }
        
        try:
            # Create a carrier for trace context propagation
            carrier = {}
            propagate.inject(carrier)
            
            # Add trace context to event metadata
            if not hasattr(input_data.event, 'metadata') or input_data.event.metadata is None:
                input_data.event.metadata = {}
            
            input_data.event.metadata.update({
                "trace_context": carrier,
                "trace_timestamp": datetime.now().isoformat(),
                "trace_service": self.tracing_config.service_name,
                "trace_environment": self.tracing_config.environment
            })
            
            return {
                "injected": True,
                "event_id": str(input_data.event.correlation_id),
                "context_keys": list(carrier.keys())
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to inject trace context: {e}")
            return {
                "injected": False,
                "reason": str(e)
            }
    
    async def _handle_extract_context(self, input_data: ModelDistributedTracingInput) -> Dict[str, Union[str, bool, Optional[str]]]:
        """Handle trace context extraction from event."""
        if not input_data.event or not self.is_initialized:
            return {
                "extracted": False,
                "reason": "Event missing or tracing not initialized"
            }
        
        try:
            if not hasattr(input_data.event, 'metadata') or not input_data.event.metadata:
                return {
                    "extracted": False,
                    "reason": "No metadata in event"
                }
            
            trace_context_data = input_data.event.metadata.get("trace_context")
            if not trace_context_data:
                return {
                    "extracted": False,
                    "reason": "No trace context in event metadata"
                }
            
            # Extract context from carrier
            extracted_context = propagate.extract(trace_context_data)
            
            return {
                "extracted": True,
                "event_id": str(input_data.event.correlation_id),
                "context_available": extracted_context is not None,
                "trace_service": input_data.event.metadata.get("trace_service"),
                "trace_environment": input_data.event.metadata.get("trace_environment")
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to extract trace context: {e}")
            return {
                "extracted": False,
                "reason": str(e)
            }
    
    async def _handle_trace_database(self, input_data: ModelDistributedTracingInput) -> Dict[str, Union[str, bool, Optional[str]]]:
        """Handle database operation tracing."""
        if not self.is_initialized or not input_data.operation_name:
            return {
                "traced": False,
                "reason": "Tracing not initialized or operation name missing"
            }
        
        attributes = {
            "db.system": "postgresql",
            "db.operation": input_data.operation_name,
            "component": "postgres_adapter",
            "correlation_id": str(input_data.correlation_id)
        }
        
        # Add sanitized query if provided (ONEX-compliant sanitization)
        if input_data.database_query:
            sanitized_query = SqlSanitizer.sanitize_for_observability(input_data.database_query)
            attributes["db.statement"] = sanitized_query
        
        # Create database span
        span = self.tracer.start_span(
            name=f"postgres.{input_data.operation_name}",
            kind=SpanKind.CLIENT,
            attributes=attributes
        )
        
        try:
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x') if span_context else None
            span_id = format(span_context.span_id, '016x') if span_context else None
            
            span.set_status(Status(StatusCode.OK))
            
            return {
                "traced": True,
                "operation_type": "database",
                "operation_name": input_data.operation_name,
                "trace_id": trace_id,
                "span_id": span_id,
                "database_system": "postgresql"
            }
            
        finally:
            span.end()
    
    async def _handle_trace_kafka(self, input_data: ModelDistributedTracingInput) -> Dict[str, Union[str, bool, Optional[str]]]:
        """Handle Kafka operation tracing."""
        if not self.is_initialized or not input_data.operation_name:
            return {
                "traced": False,
                "reason": "Tracing not initialized or operation name missing"
            }
        
        attributes = {
            "messaging.system": "kafka",
            "messaging.operation": input_data.operation_name,
            "component": "kafka_adapter",
            "correlation_id": str(input_data.correlation_id)
        }
        
        if input_data.kafka_topic:
            attributes["messaging.destination"] = input_data.kafka_topic
            attributes["messaging.destination_kind"] = "topic"
        
        # Determine span kind based on operation
        span_kind = SpanKind.PRODUCER if input_data.operation_name == "produce" else SpanKind.CONSUMER
        
        # Create Kafka span
        span = self.tracer.start_span(
            name=f"kafka.{input_data.operation_name}",
            kind=span_kind,
            attributes=attributes
        )
        
        try:
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x') if span_context else None
            span_id = format(span_context.span_id, '016x') if span_context else None
            
            span.set_status(Status(StatusCode.OK))
            
            return {
                "traced": True,
                "operation_type": "kafka",
                "operation_name": input_data.operation_name,
                "trace_id": trace_id,
                "span_id": span_id,
                "messaging_system": "kafka",
                "topic": input_data.kafka_topic
            }
            
        finally:
            span.end()
    
    async def _handle_shutdown_tracing(self, input_data: ModelDistributedTracingInput) -> Dict[str, Union[str, bool]]:
        """Handle tracing shutdown."""
        if not self.is_initialized:
            return {
                "shutdown": False,
                "reason": "Tracing not initialized"
            }
        
        try:
            if self.tracer_provider:
                # Force flush pending spans
                await asyncio.to_thread(self.tracer_provider.force_flush, timeout_millis=5000)
            
            self.is_initialized = False
            self.logger.info("Distributed tracing shutdown complete")
            
            return {
                "shutdown": True,
                "flushed_spans": True
            }
            
        except Exception as e:
            self.logger.error(f"Error during tracing shutdown: {e}")
            return {
                "shutdown": False,
                "reason": str(e)
            }
    
    async def _initialize_opentelemetry(self) -> None:
        """Initialize OpenTelemetry tracing infrastructure."""
        if self.is_initialized or not OPENTELEMETRY_AVAILABLE:
            return
        
        try:
            # Create resource with service information from validated config
            resource = Resource.create({
                SERVICE_NAME: self.tracing_config.service_name,
                SERVICE_VERSION: self.tracing_config.service_version,
                "deployment.environment": self.tracing_config.environment,
                "service.namespace": "omnibase_infrastructure"
            })

            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)

            # Configure OTLP exporter with validated endpoint
            otlp_exporter = OTLPSpanExporter(endpoint=str(self.tracing_config.otel_exporter_otlp_endpoint))

            # Add batch span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(span_processor)

            # Set global tracer provider
            trace.set_tracer_provider(self.tracer_provider)

            # Get tracer
            self.tracer = trace.get_tracer(
                instrumenting_module_name=__name__,
                instrumenting_library_version=self.tracing_config.service_version
            )

            self.is_initialized = True
            self.logger.info(
                f"OpenTelemetry tracing initialized for environment: {self.tracing_config.environment} "
                f"with endpoint: {self.tracing_config.otel_exporter_otlp_endpoint}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenTelemetry: {e}")
            raise OnexError(
                code=CoreErrorCode.CONFIGURATION_ERROR,
                message=f"OpenTelemetry initialization failed: {str(e)}"
            ) from e
    
    
    def _convert_span_kind(self, input_span_kind: InputSpanKind) -> Optional[Union["SpanKind", object]]:
        """Convert input span kind to OpenTelemetry span kind."""
        if not OPENTELEMETRY_AVAILABLE:
            return None
        
        span_kind_mapping = {
            InputSpanKind.INTERNAL: SpanKind.INTERNAL,
            InputSpanKind.SERVER: SpanKind.SERVER,
            InputSpanKind.CLIENT: SpanKind.CLIENT,
            InputSpanKind.PRODUCER: SpanKind.PRODUCER,
            InputSpanKind.CONSUMER: SpanKind.CONSUMER
        }
        
        return span_kind_mapping.get(input_span_kind, SpanKind.INTERNAL)
    
