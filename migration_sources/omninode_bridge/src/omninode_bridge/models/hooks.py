"""Hook event models for incoming webhook processing."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_serializer


class HookMetadata(BaseModel):
    """Metadata for incoming hook events."""

    source: str = Field(..., description="Source service or system")
    version: str = Field(default="1.0.0", description="Hook schema version")
    environment: str = Field(default="development", description="Environment")
    correlation_id: UUID | None = Field(None, description="Request correlation ID")
    trace_id: str | None = Field(None, description="Distributed tracing ID")
    user_agent: str | None = Field(None, description="User agent from HTTP request")
    source_ip: str | None = Field(None, description="Source IP address")


class HookPayload(BaseModel):
    """Generic payload structure for hook events."""

    action: str = Field(..., description="Action that triggered the hook")
    resource: str = Field(..., description="Resource type (service, tool, etc.)")
    resource_id: str = Field(..., description="Unique identifier for the resource")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data",
    )
    previous_state: dict[str, Any] | None = Field(
        None,
        description="Previous resource state",
    )
    current_state: dict[str, Any] | None = Field(
        None,
        description="Current resource state",
    )


class HookEvent(BaseModel):
    """Complete hook event received by the HookReceiver service."""

    id: UUID = Field(default_factory=uuid4, description="Unique hook event ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Hook received timestamp",
    )
    metadata: HookMetadata = Field(..., description="Hook metadata")
    payload: HookPayload = Field(..., description="Hook payload")

    # Processing status
    processed: bool = Field(
        default=False,
        description="Whether the hook has been processed",
    )
    processing_errors: list[str] = Field(
        default_factory=list,
        description="Processing errors",
    )
    retry_count: int = Field(default=0, description="Number of processing retries")

    @field_serializer("id")
    def serialize_id(self, value: UUID) -> str:
        return str(value)

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()

    def to_service_event(self) -> dict[str, Any]:
        """Convert hook event to internal service event format."""
        return {
            "id": str(self.id),
            "type": self._determine_event_type(),
            "event": self.payload.action,
            "service": self.metadata.source,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": (
                str(self.metadata.correlation_id)
                if self.metadata.correlation_id
                else None
            ),
            "metadata": {
                "version": self.metadata.version,
                "environment": self.metadata.environment,
                "trace_id": self.metadata.trace_id,
                "user_agent": self.metadata.user_agent,
                "source_ip": self.metadata.source_ip,
            },
            "payload": self.payload.data,
        }

    def _determine_event_type(self) -> str:
        """Determine internal event type based on hook payload."""
        action = self.payload.action.lower()
        resource = self.payload.resource.lower()

        # Service lifecycle events
        if action in ["startup", "shutdown", "ready", "health_check"]:
            return "service_lifecycle"

        # Tool execution events
        if "tool" in resource or action in ["execute", "call", "invoke"]:
            return "tool_execution"

        # Performance events
        if "performance" in resource or "metric" in resource:
            return "performance"

        # Configuration events
        if "config" in resource or action in ["configure", "update_config"]:
            return "configuration"

        # Default to service lifecycle
        return "service_lifecycle"


class HookResponse(BaseModel):
    """Response sent back to hook sender."""

    success: bool = Field(
        ...,
        description="Whether the hook was processed successfully",
    )
    message: str = Field(..., description="Human-readable response message")
    event_id: UUID = Field(..., description="Generated event ID for tracking")
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds",
    )
    errors: list[str] = Field(default_factory=list, description="Any processing errors")

    @field_serializer("event_id")
    def serialize_event_id(self, value: UUID) -> str:
        return str(value)
