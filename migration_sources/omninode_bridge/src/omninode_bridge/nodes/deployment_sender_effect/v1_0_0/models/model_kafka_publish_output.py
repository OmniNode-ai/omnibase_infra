"""
Kafka publish output model for event publishing results.

ONEX v2.0 Pydantic Model
"""

from typing import ClassVar, Optional

from pydantic import BaseModel, Field


class ModelKafkaPublishOutput(BaseModel):
    """
    Output model for publish_transfer_event IO operation.

    Contains event publishing results and metadata.
    """

    success: bool = Field(
        ...,
        description="Whether the event publishing succeeded",
    )

    event_published: bool = Field(
        ...,
        description="Whether the event was successfully published to Kafka",
    )

    topic: Optional[str] = Field(
        default=None,
        description="Kafka topic where event was published",
        min_length=1,
        max_length=256,
    )

    partition: Optional[int] = Field(
        default=None,
        description="Kafka partition where event was published",
        ge=0,
    )

    offset: Optional[int] = Field(
        default=None,
        description="Kafka offset of published event",
        ge=0,
    )

    publish_duration_ms: Optional[int] = Field(
        default=None,
        description="Event publishing duration in milliseconds",
        ge=0,
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if operation failed",
        max_length=2048,
    )

    error_code: Optional[str] = Field(
        default=None,
        description="Error code for failed operations",
        max_length=64,
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict] = {
            "example": {
                "success": True,
                "event_published": True,
                "topic": "dev.omninode-bridge.deployment.build-completed.v1",
                "partition": 0,
                "offset": 12345,
                "publish_duration_ms": 25,
            }
        }
