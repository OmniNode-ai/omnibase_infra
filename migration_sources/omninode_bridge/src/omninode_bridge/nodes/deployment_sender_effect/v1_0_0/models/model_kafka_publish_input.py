"""
Kafka publish input model for deployment lifecycle events.

ONEX v2.0 Pydantic Model
"""

from typing import Any, ClassVar, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ModelKafkaPublishInput(BaseModel):
    """
    Input model for publish_transfer_event IO operation.

    Publishes deployment lifecycle events to Kafka topics.
    """

    event_type: str = Field(
        ...,
        description="Type of deployment event",
        min_length=1,
        max_length=64,
    )

    event_payload: dict[str, Any] = Field(
        ...,
        description="Event payload data",
    )

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for event tracking",
    )

    package_id: Optional[UUID] = Field(
        default=None,
        description="Package ID for package-related events",
    )

    container_name: Optional[str] = Field(
        default=None,
        description="Container name for deployment events",
        min_length=1,
        max_length=128,
    )

    image_tag: Optional[str] = Field(
        default=None,
        description="Image tag for deployment events",
        min_length=1,
        max_length=64,
    )

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event type."""
        allowed = [
            "BUILD_STARTED",
            "BUILD_COMPLETED",
            "TRANSFER_STARTED",
            "TRANSFER_COMPLETED",
            "DEPLOYMENT_FAILED",
        ]
        if v not in allowed:
            raise ValueError(f"Event type must be one of {allowed}, got: {v}")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict] = {
            "example": {
                "event_type": "BUILD_COMPLETED",
                "event_payload": {
                    "image_id": "sha256:abc123...",
                    "build_duration_ms": 12500,
                    "package_size_mb": 256.5,
                },
                "correlation_id": "123e4567-e89b-12d3-a456-426614174000",
                "package_id": "123e4567-e89b-12d3-a456-426614174000",
                "container_name": "omninode-bridge-orchestrator",
                "image_tag": "1.0.0",
            }
        }
