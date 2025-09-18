"""Strongly typed models for security event data."""

from pydantic import BaseModel, Field


class ModelSecurityEventDetails(BaseModel):
    """Security event details with strong typing."""

    # Event identification
    event_id: str = Field(description="Unique event identifier")
    event_type: str = Field(description="Type of security event")
    severity: str = Field(description="Event severity level")

    # Source information
    source_ip: str | None = Field(default=None, description="Source IP address")
    user_agent: str | None = Field(default=None, description="User agent string")
    user_id: str | None = Field(default=None, description="User identifier")
    session_id: str | None = Field(default=None, description="Session identifier")

    # Event context
    resource_accessed: str | None = Field(
        default=None, description="Resource that was accessed",
    )
    action_attempted: str | None = Field(
        default=None, description="Action that was attempted",
    )
    result: str | None = Field(default=None, description="Result of the action")

    # Timing
    timestamp: str = Field(description="ISO timestamp of the event")
    duration_ms: float | None = Field(
        default=None, description="Duration in milliseconds",
    )

    # Additional context
    tags: list[str] = Field(default_factory=list, description="Event tags")
    custom_fields: list[str] = Field(
        default_factory=list, description="Custom field values",
    )


class ModelSecurityEventMetadata(BaseModel):
    """Security event metadata."""

    correlation_id: str | None = Field(
        default=None, description="Request correlation ID",
    )
    tenant_id: str | None = Field(default=None, description="Tenant identifier")
    environment: str | None = Field(default=None, description="Environment context")
    service_name: str | None = Field(
        default=None, description="Service that generated the event",
    )
    service_version: str | None = Field(default=None, description="Service version")

    # Security context
    security_level: str | None = Field(
        default=None, description="Security level classification",
    )
    risk_score: float | None = Field(default=None, description="Risk score 0-100")
    threat_indicators: list[str] = Field(
        default_factory=list, description="Threat indicator flags",
    )


class ModelAuditLogEntry(BaseModel):
    """Complete audit log entry structure."""

    # Core event data
    event_details: ModelSecurityEventDetails = Field(description="Event details")
    metadata: ModelSecurityEventMetadata | None = Field(
        default=None, description="Event metadata",
    )

    # Audit trail
    created_at: str = Field(description="ISO timestamp when log entry was created")
    hash_chain_value: str = Field(
        description="Hash chain value for integrity verification",
    )
    previous_hash: str | None = Field(
        default=None, description="Previous entry hash for chain verification",
    )

    # Processing status
    is_processed: bool = Field(
        default=False, description="Whether the event has been processed",
    )
    processing_notes: list[str] = Field(
        default_factory=list, description="Processing notes and actions taken",
    )
