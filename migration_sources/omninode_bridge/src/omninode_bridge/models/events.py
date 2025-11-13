"""Event models for OmniNode Bridge service lifecycle management."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class EventType(str, Enum):
    """Top-level event categories."""

    SERVICE_LIFECYCLE = "service_lifecycle"
    TOOL_EXECUTION = "tool_execution"
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"


class ServiceEventType(str, Enum):
    """Service lifecycle event subtypes."""

    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    HEALTH_CHECK = "health_check"
    REGISTRATION = "registration"
    DEREGISTRATION = "deregistration"
    READY = "ready"
    ERROR = "error"


class ToolEventType(str, Enum):
    """Tool execution event subtypes."""

    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    TOOL_ERROR = "tool_error"
    TOOL_TIMEOUT = "tool_timeout"


class BaseEvent(BaseModel):
    """Base event model for all OmniNode Bridge events."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    type: EventType = Field(..., description="Event category")
    event: str = Field(..., description="Specific event name")
    service: str = Field(..., description="Source service name")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp",
    )
    correlation_id: UUID | None = Field(None, description="Request correlation ID")

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Event metadata (version, environment, etc.)",
    )

    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data",
    )

    @field_serializer("id", "correlation_id")
    def serialize_uuid(self, value: UUID | None) -> str | None:
        return str(value) if value is not None else None

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()


class ServiceLifecycleEvent(BaseEvent):
    """Service lifecycle events (startup, shutdown, health checks, etc.)."""

    type: Literal[EventType.SERVICE_LIFECYCLE] = EventType.SERVICE_LIFECYCLE
    event: ServiceEventType = Field(..., description="Service lifecycle event type")

    # Common service lifecycle payload fields
    service_version: str | None = Field(None, description="Service version")
    environment: str | None = Field(
        None,
        description="Environment (dev/staging/prod)",
    )
    instance_id: str | None = Field(None, description="Service instance identifier")
    health_status: str | None = Field(None, description="Health check status")
    dependencies: dict[str, str] | None = Field(
        None,
        description="Service dependencies status",
    )

    def to_kafka_topic(self) -> str:
        """Generate Kafka topic name for this event."""
        return f"dev.omninode_bridge.onex.evt.{self.event.value}.v1"


class ToolExecutionEvent(BaseEvent):
    """Tool execution events (calls, responses, errors)."""

    type: Literal[EventType.TOOL_EXECUTION] = EventType.TOOL_EXECUTION
    event: ToolEventType = Field(..., description="Tool execution event type")

    # Tool execution specific fields
    tool_name: str | None = Field(None, description="Name of the executed tool")
    execution_time_ms: int | None = Field(
        None,
        description="Tool execution time in milliseconds",
    )
    input_size: int | None = Field(None, description="Input data size in bytes")
    output_size: int | None = Field(None, description="Output data size in bytes")
    success: bool | None = Field(None, description="Tool execution success status")
    error_message: str | None = Field(None, description="Error message if failed")

    def to_kafka_topic(self) -> str:
        """Generate Kafka topic name for this event."""
        return f"dev.omninode_bridge.onex.evt.{self.event.value.replace('_', '-')}.v1"


class PerformanceEvent(BaseEvent):
    """Performance monitoring events (metrics, resource usage)."""

    type: Literal[EventType.PERFORMANCE] = EventType.PERFORMANCE

    # Performance specific fields
    response_time_ms: float | None = Field(
        None,
        description="Response time in milliseconds",
    )
    throughput: float | None = Field(
        None,
        description="Throughput (requests/second)",
    )
    cpu_usage: float | None = Field(None, description="CPU usage percentage")
    memory_usage: float | None = Field(None, description="Memory usage in MB")
    disk_usage: float | None = Field(None, description="Disk usage percentage")
    network_io: dict[str, float] | None = Field(
        None,
        description="Network I/O metrics",
    )

    def to_kafka_topic(self) -> str:
        """Generate Kafka topic name for this event."""
        return "dev.omninode_bridge.onex.met.performance.v1"


class ConfigurationEvent(BaseEvent):
    """Configuration change events."""

    type: Literal[EventType.CONFIGURATION] = EventType.CONFIGURATION

    # Configuration specific fields
    config_key: str | None = Field(
        None,
        description="Configuration key that changed",
    )
    old_value: str | None = Field(None, description="Previous configuration value")
    new_value: str | None = Field(None, description="New configuration value")
    change_reason: str | None = Field(
        None,
        description="Reason for configuration change",
    )
    applied_by: str | None = Field(None, description="Who/what applied the change")

    def to_kafka_topic(self) -> str:
        """Generate Kafka topic name for this event."""
        return "dev.omninode_bridge.onex.evt.configuration-changed.v1"


# Union type for all event types
AnyEvent = Union[
    ServiceLifecycleEvent,
    ToolExecutionEvent,
    PerformanceEvent,
    ConfigurationEvent,
]
