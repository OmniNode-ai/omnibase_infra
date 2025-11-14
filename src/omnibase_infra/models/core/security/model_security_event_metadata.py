"""Security event metadata model."""

from uuid import UUID

from pydantic import BaseModel, Field


class ModelSecurityEventMetadata(BaseModel):
    """Security event metadata."""

    correlation_id: UUID | None = Field(
        default=None, description="Request correlation ID",
    )
    tenant_id: UUID | None = Field(default=None, description="Tenant identifier")
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

    class Config:
        json_encoders = {
            UUID: lambda v: str(v),
        }
