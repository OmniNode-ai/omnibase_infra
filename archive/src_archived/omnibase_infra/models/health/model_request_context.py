"""Health Request Context Model.

Strongly-typed model for health request context information.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field


class ModelHealthRequestContext(BaseModel):
    """Model for health monitoring request context."""

    # Request source information
    source_service: str | None = Field(
        default=None,
        max_length=100,
        description="Name of the service making the request",
    )

    source_version: str | None = Field(
        default=None,
        max_length=50,
        description="Version of the service making the request",
    )

    user_agent: str | None = Field(
        default=None,
        max_length=200,
        description="User agent string for the request",
    )

    # Request configuration
    timeout_seconds: int | None = Field(
        default=None,
        gt=0,
        le=300,
        description="Request timeout in seconds",
    )

    retry_count: int | None = Field(
        default=None,
        ge=0,
        le=5,
        description="Number of retries to attempt",
    )

    priority_level: str | None = Field(
        default=None,
        pattern="^(low|normal|high|critical)$",
        description="Request priority level",
    )

    # Monitoring context
    alert_thresholds_override: bool | None = Field(
        default=None,
        description="Whether to use custom alert thresholds",
    )

    custom_error_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Custom error rate threshold percentage",
    )

    custom_response_time_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description="Custom response time threshold in milliseconds",
    )

    # Data collection preferences
    detailed_metrics_required: bool | None = Field(
        default=None,
        description="Whether detailed metrics are required",
    )

    historical_data_required: bool | None = Field(
        default=None,
        description="Whether historical data should be included",
    )

    include_resource_metrics: bool | None = Field(
        default=None,
        description="Whether to include resource utilization metrics",
    )

    # Notification preferences
    notification_channels: list[str] | None = Field(
        default=None,
        max_items=10,
        description="Notification channels for alerts",
    )

    suppress_notifications: bool | None = Field(
        default=None,
        description="Whether to suppress notifications for this request",
    )

    # Debugging and tracing
    debug_mode: bool | None = Field(
        default=None,
        description="Whether to enable debug mode for this request",
    )

    trace_id: str | None = Field(
        default=None,
        max_length=100,
        description="Distributed tracing trace ID",
    )

    span_id: str | None = Field(
        default=None,
        max_length=50,
        description="Distributed tracing span ID",
    )

    # Performance preferences
    cache_results: bool | None = Field(
        default=None,
        description="Whether results should be cached",
    )

    cache_ttl_seconds: int | None = Field(
        default=None,
        gt=0,
        le=3600,
        description="Cache time-to-live in seconds",
    )

    # Environment-specific context
    deployment_stage: str | None = Field(
        default=None,
        pattern="^(development|staging|production|test)$",
        description="Deployment stage context",
    )

    region: str | None = Field(
        default=None,
        max_length=50,
        description="Deployment region",
    )

    availability_zone: str | None = Field(
        default=None,
        max_length=50,
        description="Availability zone",
    )
