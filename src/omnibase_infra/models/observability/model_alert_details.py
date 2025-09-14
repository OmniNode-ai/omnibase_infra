"""Alert Details Model.

Strongly-typed model for infrastructure alert details and context.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ModelAlertDetails(BaseModel):
    """Model for infrastructure alert details and context."""

    # Metric information
    metric_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Name of the metric that triggered the alert"
    )

    metric_unit: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Unit of measurement for the metric"
    )

    measurement_interval: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Interval over which the metric was measured"
    )

    # Threshold information
    warning_threshold: Optional[float] = Field(
        default=None,
        description="Warning threshold value"
    )

    critical_threshold: Optional[float] = Field(
        default=None,
        description="Critical threshold value"
    )

    threshold_operator: Optional[str] = Field(
        default=None,
        pattern="^(gt|gte|lt|lte|eq|neq)$",
        description="Threshold comparison operator"
    )

    # Time-based information
    duration_seconds: Optional[int] = Field(
        default=None,
        ge=0,
        description="Duration the condition has been active in seconds"
    )

    first_occurrence: Optional[str] = Field(
        default=None,
        description="ISO timestamp of first occurrence"
    )

    last_occurrence: Optional[str] = Field(
        default=None,
        description="ISO timestamp of last occurrence"
    )

    occurrence_count: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of times the condition has occurred"
    )

    # Component information
    affected_components: Optional[List[str]] = Field(
        default=None,
        max_items=20,
        description="List of components affected by this alert"
    )

    component_health_status: Optional[str] = Field(
        default=None,
        pattern="^(healthy|degraded|unhealthy|unknown)$",
        description="Health status of the affected component"
    )

    # Impact assessment
    impact_level: Optional[str] = Field(
        default=None,
        pattern="^(none|low|medium|high|severe)$",
        description="Assessed impact level"
    )

    users_affected_estimate: Optional[int] = Field(
        default=None,
        ge=0,
        description="Estimated number of users affected"
    )

    services_affected: Optional[List[str]] = Field(
        default=None,
        max_items=20,
        description="List of services affected by this alert"
    )

    # Resolution information
    auto_resolution_available: Optional[bool] = Field(
        default=None,
        description="Whether automatic resolution is available"
    )

    manual_intervention_required: Optional[bool] = Field(
        default=None,
        description="Whether manual intervention is required"
    )

    runbook_url: Optional[str] = Field(
        default=None,
        max_length=500,
        description="URL to relevant runbook or documentation"
    )

    escalation_policy: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Escalation policy to follow"
    )

    # Notification information
    notification_channels: Optional[List[str]] = Field(
        default=None,
        max_items=10,
        description="Notification channels used for this alert"
    )

    notification_sent: Optional[bool] = Field(
        default=None,
        description="Whether notifications have been sent"
    )

    # Context and metadata
    deployment_version: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Deployment version when alert was triggered"
    )

    region: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Geographic region where alert occurred"
    )

    availability_zone: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Availability zone where alert occurred"
    )

    # Performance context
    baseline_value: Optional[float] = Field(
        default=None,
        description="Baseline value for comparison"
    )

    deviation_percentage: Optional[float] = Field(
        default=None,
        description="Percentage deviation from baseline"
    )

    trend_direction: Optional[str] = Field(
        default=None,
        pattern="^(improving|stable|degrading|unknown)$",
        description="Trend direction of the metric"
    )