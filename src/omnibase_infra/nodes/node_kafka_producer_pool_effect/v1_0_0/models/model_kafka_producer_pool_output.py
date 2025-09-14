"""Output model for Kafka producer pool EFFECT node."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ModelKafkaProducerPoolOutput(BaseModel):
    """Output model for Kafka producer pool operations."""

    success: bool = Field(
        description="Whether the operation completed successfully"
    )
    correlation_id: UUID = Field(
        description="Unique identifier for request tracing"
    )
    timestamp: datetime = Field(
        description="Timestamp of operation completion"
    )