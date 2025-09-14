"""Span Attributes Model.

Strongly-typed model for OpenTelemetry span attributes.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ModelSpanAttributes(BaseModel):
    """Model for OpenTelemetry span attributes."""

    # Service identification
    service_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Name of the service creating the span"
    )

    service_version: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Version of the service creating the span"
    )

    service_instance_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Instance ID of the service"
    )

    # HTTP attributes (if applicable)
    http_method: Optional[str] = Field(
        default=None,
        pattern="^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)$",
        description="HTTP method"
    )

    http_url: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Full HTTP URL"
    )

    http_status_code: Optional[int] = Field(
        default=None,
        ge=100,
        le=599,
        description="HTTP response status code"
    )

    http_user_agent: Optional[str] = Field(
        default=None,
        max_length=500,
        description="HTTP User-Agent header value"
    )

    # Database attributes (if applicable)
    db_system: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Database management system identifier"
    )

    db_connection_string: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Database connection string (sanitized)"
    )

    db_user: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Database user name"
    )

    db_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Database name"
    )

    db_operation: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Database operation name"
    )

    # Messaging attributes (if applicable)
    messaging_system: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Messaging system identifier"
    )

    messaging_destination: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Message destination name"
    )

    messaging_destination_kind: Optional[str] = Field(
        default=None,
        pattern="^(queue|topic|exchange)$",
        description="Kind of message destination"
    )

    messaging_operation: Optional[str] = Field(
        default=None,
        pattern="^(publish|receive|process)$",
        description="Messaging operation type"
    )

    # RPC attributes (if applicable)
    rpc_system: Optional[str] = Field(
        default=None,
        max_length=50,
        description="RPC system identifier"
    )

    rpc_service: Optional[str] = Field(
        default=None,
        max_length=100,
        description="RPC service name"
    )

    rpc_method: Optional[str] = Field(
        default=None,
        max_length=100,
        description="RPC method name"
    )

    # Error attributes
    error_type: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Error type/class name"
    )

    error_message: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Error message"
    )

    # Performance attributes
    operation_duration_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Operation duration in milliseconds"
    )

    cpu_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage during operation"
    )

    memory_usage_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Memory usage in megabytes"
    )

    # Custom business attributes
    user_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="User identifier"
    )

    tenant_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Tenant identifier"
    )

    correlation_ids: Optional[List[str]] = Field(
        default=None,
        max_items=10,
        description="List of correlation identifiers"
    )

    # Environment attributes
    environment: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Deployment environment"
    )

    region: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Geographic region"
    )

    availability_zone: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Availability zone"
    )