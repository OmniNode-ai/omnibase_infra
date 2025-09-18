"""Distributed Tracing Input Model.

Node-specific input model for the distributed tracing compute node.
"""

from enum import Enum
from uuid import UUID

from omnibase_core.models.core.model_onex_event import ModelOnexEvent
from pydantic import BaseModel, Field

from omnibase_infra.models.tracing.model_span_attributes import ModelSpanAttributes


class TracingOperation(str, Enum):
    """Distributed tracing operations."""
    INITIALIZE_TRACING = "initialize_tracing"
    TRACE_OPERATION = "trace_operation"
    INJECT_CONTEXT = "inject_context"
    EXTRACT_CONTEXT = "extract_context"
    TRACE_DATABASE = "trace_database"
    TRACE_KAFKA = "trace_kafka"
    SHUTDOWN_TRACING = "shutdown_tracing"


class SpanKind(str, Enum):
    """OpenTelemetry span kinds."""
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


class ModelDistributedTracingInput(BaseModel):
    """Input model for distributed tracing operations."""

    operation_type: TracingOperation = Field(
        description="Type of tracing operation to perform",
    )

    operation_name: str | None = Field(
        default=None,
        description="Name of the operation to trace (required for trace operations)",
    )

    correlation_id: UUID = Field(
        description="Correlation ID for the operation",
    )

    event: ModelOnexEvent | None = Field(
        default=None,
        description="Event for context injection/extraction operations",
    )

    span_kind: SpanKind = Field(
        default=SpanKind.INTERNAL,
        description="OpenTelemetry span kind",
    )

    attributes: ModelSpanAttributes | None = Field(
        default=None,
        description="Additional span attributes",
    )

    environment: str = Field(
        default="development",
        description="Environment configuration (development, staging, production)",
    )

    database_query: str | None = Field(
        default=None,
        description="Database query for database tracing operations",
    )

    kafka_topic: str | None = Field(
        default=None,
        description="Kafka topic for Kafka tracing operations",
    )
