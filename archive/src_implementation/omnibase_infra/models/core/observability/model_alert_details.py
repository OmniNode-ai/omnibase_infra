"""Alert Details Model.

Strongly-typed model for infrastructure alert details and context.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field


class ModelAlertDetails(BaseModel):
    """Model for infrastructure alert details and context."""

    # Metric information
    metric_name: str | None = Field(
        default=None,
        max_length=100,
        description="Name of the metric that triggered the alert",
    )

    metric_unit: str | None = Field(
        default=None,
        max_length=20,
        description="Unit of measurement for the metric",
    )

    measurement_interval: str | None = Field(
        default=None,
        max_length=50,
        description="Interval over which the metric was measured",
    )

    # Threshold information
    warning_threshold: float | None = Field(
        default=None,
        description="Warning threshold value",
    )

    critical_threshold: float | None = Field(
        default=None,
        description="Critical threshold value",
    )

    threshold_operator: str | None = Field(
        default=None,
        pattern="^(gt|gte|lt|lte|eq|neq)$",
        description="Threshold comparison operator",
    )

    # Time-based information
    duration_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Duration the condition has been active in seconds",
    )

    first_occurrence: str | None = Field(
        default=None,
        description="ISO timestamp of first occurrence",
    )

    last_occurrence: str | None = Field(
        default=None,
        description="ISO timestamp of last occurrence",
    )

    occurrence_count: int | None = Field(
        default=None,
        ge=1,
        description="Number of times the condition has occurred",
    )

    # Component information
    affected_components: list[str] | None = Field(
        default=None,
        max_items=20,
        description="List of components affected by this alert",
    )

    component_health_status: str | None = Field(
        default=None,
        pattern="^(healthy|degraded|unhealthy|unknown)$",
        description="Health status of the affected component",
    )

    # Impact assessment
    impact_level: str | None = Field(
        default=None,
        pattern="^(none|low|medium|high|severe)$",
        description="Assessed impact level",
    )

    users_affected_estimate: int | None = Field(
        default=None,
        ge=0,
        description="Estimated number of users affected",
    )

    services_affected: list[str] | None = Field(
        default=None,
        max_items=20,
        description="List of services affected by this alert",
    )

    # Resolution information
    auto_resolution_available: bool | None = Field(
        default=None,
        description="Whether automatic resolution is available",
    )

    manual_intervention_required: bool | None = Field(
        default=None,
        description="Whether manual intervention is required",
    )

    runbook_url: str | None = Field(
        default=None,
        max_length=500,
        description="URL to relevant runbook or documentation",
    )

    escalation_policy: str | None = Field(
        default=None,
        max_length=100,
        description="Escalation policy to follow",
    )

    # Notification information
    notification_channels: list[str] | None = Field(
        default=None,
        max_items=10,
        description="Notification channels used for this alert",
    )

    notification_sent: bool | None = Field(
        default=None,
        description="Whether notifications have been sent",
    )

    # Context and metadata
    deployment_version: str | None = Field(
        default=None,
        max_length=50,
        description="Deployment version when alert was triggered",
    )

    region: str | None = Field(
        default=None,
        max_length=50,
        description="Geographic region where alert occurred",
    )

    availability_zone: str | None = Field(
        default=None,
        max_length=50,
        description="Availability zone where alert occurred",
    )

    # Performance context
    baseline_value: float | None = Field(
        default=None,
        description="Baseline value for comparison",
    )

    deviation_percentage: float | None = Field(
        default=None,
        description="Percentage deviation from baseline",
    )

    trend_direction: str | None = Field(
        default=None,
        pattern="^(improving|stable|degrading|unknown)$",
        description="Trend direction of the metric",
    )
