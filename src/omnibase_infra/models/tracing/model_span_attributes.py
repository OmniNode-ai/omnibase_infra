"""Span Attributes Model.

Strongly-typed model for OpenTelemetry span attributes.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""


from pydantic import BaseModel, Field


class ModelSpanAttributes(BaseModel):
    """Model for OpenTelemetry span attributes."""

    # Service identification
    service_name: str | None = Field(
        default=None,
        max_length=100,
        description="Name of the service creating the span",
    )

    service_version: str | None = Field(
        default=None,
        max_length=50,
        description="Version of the service creating the span",
    )

    service_instance_id: str | None = Field(
        default=None,
        max_length=100,
        description="Instance ID of the service",
    )

    # HTTP attributes (if applicable)
    http_method: str | None = Field(
        default=None,
        pattern="^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)$",
        description="HTTP method",
    )

    http_url: str | None = Field(
        default=None,
        max_length=500,
        description="Full HTTP URL",
    )

    http_status_code: int | None = Field(
        default=None,
        ge=100,
        le=599,
        description="HTTP response status code",
    )

    http_user_agent: str | None = Field(
        default=None,
        max_length=500,
        description="HTTP User-Agent header value",
    )

    # Database attributes (if applicable)
    db_system: str | None = Field(
        default=None,
        max_length=50,
        description="Database management system identifier",
    )

    db_connection_string: str | None = Field(
        default=None,
        max_length=500,
        description="Database connection string (sanitized)",
    )

    db_user: str | None = Field(
        default=None,
        max_length=100,
        description="Database user name",
    )

    db_name: str | None = Field(
        default=None,
        max_length=100,
        description="Database name",
    )

    db_operation: str | None = Field(
        default=None,
        max_length=50,
        description="Database operation name",
    )

    # Messaging attributes (if applicable)
    messaging_system: str | None = Field(
        default=None,
        max_length=50,
        description="Messaging system identifier",
    )

    messaging_destination: str | None = Field(
        default=None,
        max_length=200,
        description="Message destination name",
    )

    messaging_destination_kind: str | None = Field(
        default=None,
        pattern="^(queue|topic|exchange)$",
        description="Kind of message destination",
    )

    messaging_operation: str | None = Field(
        default=None,
        pattern="^(publish|receive|process)$",
        description="Messaging operation type",
    )

    # RPC attributes (if applicable)
    rpc_system: str | None = Field(
        default=None,
        max_length=50,
        description="RPC system identifier",
    )

    rpc_service: str | None = Field(
        default=None,
        max_length=100,
        description="RPC service name",
    )

    rpc_method: str | None = Field(
        default=None,
        max_length=100,
        description="RPC method name",
    )

    # Error attributes
    error_type: str | None = Field(
        default=None,
        max_length=100,
        description="Error type/class name",
    )

    error_message: str | None = Field(
        default=None,
        max_length=1000,
        description="Error message",
    )

    # Performance attributes
    operation_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Operation duration in milliseconds",
    )

    cpu_usage_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage during operation",
    )

    memory_usage_mb: float | None = Field(
        default=None,
        ge=0.0,
        description="Memory usage in megabytes",
    )

    # Custom business attributes
    user_id: str | None = Field(
        default=None,
        max_length=100,
        description="User identifier",
    )

    tenant_id: str | None = Field(
        default=None,
        max_length=100,
        description="Tenant identifier",
    )

    correlation_ids: list[str] | None = Field(
        default=None,
        max_items=10,
        description="List of correlation identifiers",
    )

    # Environment attributes
    environment: str | None = Field(
        default=None,
        max_length=50,
        description="Deployment environment",
    )

    region: str | None = Field(
        default=None,
        max_length=50,
        description="Geographic region",
    )

    availability_zone: str | None = Field(
        default=None,
        max_length=50,
        description="Availability zone",
    )
