"""Input model for Kafka producer pool EFFECT node."""

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class OperationType(str, Enum):
    """Kafka producer pool operation types."""
    SEND_MESSAGE = "send_message"
    GET_POOL_STATS = "get_pool_stats"
    GET_HEALTH = "get_health"


class ModelKafkaProducerPoolInput(BaseModel):
    """Input model for Kafka producer pool operations."""

    operation_type: OperationType = Field(
        description="Type of producer pool operation to perform"
    )
    correlation_id: UUID = Field(
        description="Unique identifier for request tracing"
    )