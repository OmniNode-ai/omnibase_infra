"""Alert Model.

Shared model for infrastructure alerts and notifications.
Used across observability infrastructure for alert management.
"""

from datetime import datetime
from uuid import UUID

from omnibase_core.enums.enum_environment_type import EnumEnvironmentType
from pydantic import BaseModel, Field

from omnibase_infra.models.core.observability.model_alert_details import (
    ModelAlertDetails,
)


class ModelAlert(BaseModel):
    """Model for infrastructure alerts."""

    id: UUID = Field(
        description="Unique alert identifier",
    )

    name: str = Field(
        description="Alert name/title",
    )

    description: str = Field(
        description="Detailed alert description",
    )

    severity: str = Field(
        description="Alert severity level",
    )

    timestamp: datetime = Field(
        description="Alert creation timestamp",
    )

    source: str = Field(
        description="Source component that generated the alert",
    )

    resolved: bool = Field(
        default=False,
        description="Whether the alert has been resolved",
    )

    resolution_timestamp: datetime | None = Field(
        default=None,
        description="Alert resolution timestamp",
    )

    details: ModelAlertDetails | None = Field(
        default=None,
        description="Additional alert details and context",
    )

    environment: EnumEnvironmentType | None = Field(
        default=None,
        description="Environment where alert was generated",
    )

    threshold_value: float | None = Field(
        default=None,
        description="Threshold value that triggered the alert",
    )

    current_value: float | None = Field(
        default=None,
        description="Current metric value when alert was triggered",
    )

    alert_rule: str | None = Field(
        default=None,
        description="Alert rule that triggered this alert",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
