"""Security event details model with strong typing."""

from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, Field


class ModelSecurityEventDetails(BaseModel):
    """Security event details with strong typing."""

    # Event identification
    event_id: UUID = Field(description="Unique event identifier")
    event_type: str = Field(description="Type of security event")
    severity: str = Field(description="Event severity level")

    # Source information
    source_ip: str | None = Field(default=None, description="Source IP address")
    user_agent: str | None = Field(default=None, description="User agent string")
    user_id: UUID | None = Field(default=None, description="User identifier")
    session_id: UUID | None = Field(default=None, description="Session identifier")

    # Event context
    resource_accessed: str | None = Field(
        default=None, description="Resource that was accessed",
    )
    action_attempted: str | None = Field(
        default=None, description="Action that was attempted",
    )
    result: str | None = Field(default=None, description="Result of the action")

    # Timing
    timestamp: datetime = Field(description="Event timestamp")
    duration_ms: float | None = Field(
        default=None, description="Duration in milliseconds",
    )

    # Additional context
    tags: list[str] = Field(default_factory=list, description="Event tags")
    custom_fields: list[str] = Field(
        default_factory=list, description="Custom field values",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }