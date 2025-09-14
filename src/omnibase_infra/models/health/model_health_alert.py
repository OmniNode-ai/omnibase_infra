"""Health Alert Model.

Strongly-typed model for health monitoring alerts.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


class ModelHealthAlert(BaseModel):
    """Model for health monitoring alerts."""

    alert_id: UUID = Field(
        description="Unique identifier for this alert"
    )

    component_name: str = Field(
        description="Name of the component that triggered the alert"
    )

    alert_type: str = Field(
        pattern="^(performance|availability|resource|security|configuration)$",
        description="Type of alert triggered"
    )

    severity: str = Field(
        pattern="^(low|medium|high|critical)$",
        description="Alert severity level"
    )

    status: str = Field(
        pattern="^(active|acknowledged|resolved|suppressed)$",
        description="Current status of the alert"
    )

    # Timing information
    triggered_at: datetime = Field(
        description="When the alert was first triggered"
    )

    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="When the alert was acknowledged"
    )

    resolved_at: Optional[datetime] = Field(
        default=None,
        description="When the alert was resolved"
    )

    # Alert details
    title: str = Field(
        max_length=200,
        description="Brief alert title"
    )

    description: str = Field(
        max_length=1000,
        description="Detailed alert description"
    )

    # Threshold information
    threshold_value: Optional[float] = Field(
        default=None,
        description="The threshold value that was breached"
    )

    current_value: Optional[float] = Field(
        default=None,
        description="The current value that triggered the alert"
    )

    metric_name: Optional[str] = Field(
        default=None,
        description="Name of the metric that triggered the alert"
    )

    metric_unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement for the metric"
    )

    # Impact assessment
    impact_level: str = Field(
        pattern="^(none|low|medium|high|severe)$",
        description="Assessed impact level of the issue"
    )

    affected_users_estimate: Optional[int] = Field(
        default=None,
        ge=0,
        description="Estimated number of affected users"
    )

    # Response information
    auto_resolve_available: bool = Field(
        default=False,
        description="Whether automatic resolution is available"
    )

    escalation_required: bool = Field(
        default=False,
        description="Whether escalation to human operators is required"
    )

    runbook_url: Optional[str] = Field(
        default=None,
        max_length=500,
        description="URL to relevant runbook or documentation"
    )

    # Tracking information
    acknowledged_by: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Username or system that acknowledged the alert"
    )

    resolved_by: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Username or system that resolved the alert"
    )

    resolution_notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Notes about how the alert was resolved"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }