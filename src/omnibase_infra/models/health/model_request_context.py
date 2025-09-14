"""Health Request Context Model.

Strongly-typed model for health request context information.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ModelHealthRequestContext(BaseModel):
    """Model for health monitoring request context."""

    # Request source information
    source_service: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Name of the service making the request"
    )

    source_version: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Version of the service making the request"
    )

    user_agent: Optional[str] = Field(
        default=None,
        max_length=200,
        description="User agent string for the request"
    )

    # Request configuration
    timeout_seconds: Optional[int] = Field(
        default=None,
        gt=0,
        le=300,
        description="Request timeout in seconds"
    )

    retry_count: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="Number of retries to attempt"
    )

    priority_level: Optional[str] = Field(
        default=None,
        pattern="^(low|normal|high|critical)$",
        description="Request priority level"
    )

    # Monitoring context
    alert_thresholds_override: Optional[bool] = Field(
        default=None,
        description="Whether to use custom alert thresholds"
    )

    custom_error_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Custom error rate threshold percentage"
    )

    custom_response_time_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Custom response time threshold in milliseconds"
    )

    # Data collection preferences
    detailed_metrics_required: Optional[bool] = Field(
        default=None,
        description="Whether detailed metrics are required"
    )

    historical_data_required: Optional[bool] = Field(
        default=None,
        description="Whether historical data should be included"
    )

    include_resource_metrics: Optional[bool] = Field(
        default=None,
        description="Whether to include resource utilization metrics"
    )

    # Notification preferences
    notification_channels: Optional[List[str]] = Field(
        default=None,
        max_items=10,
        description="Notification channels for alerts"
    )

    suppress_notifications: Optional[bool] = Field(
        default=None,
        description="Whether to suppress notifications for this request"
    )

    # Debugging and tracing
    debug_mode: Optional[bool] = Field(
        default=None,
        description="Whether to enable debug mode for this request"
    )

    trace_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Distributed tracing trace ID"
    )

    span_id: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Distributed tracing span ID"
    )

    # Performance preferences
    cache_results: Optional[bool] = Field(
        default=None,
        description="Whether results should be cached"
    )

    cache_ttl_seconds: Optional[int] = Field(
        default=None,
        gt=0,
        le=3600,
        description="Cache time-to-live in seconds"
    )

    # Environment-specific context
    deployment_stage: Optional[str] = Field(
        default=None,
        pattern="^(development|staging|production|test)$",
        description="Deployment stage context"
    )

    region: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Deployment region"
    )

    availability_zone: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Availability zone"
    )