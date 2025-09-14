"""Get health output model for Kafka producer pool EFFECT node."""

from datetime import datetime

from pydantic import BaseModel, Field

from omnibase_infra.models.kafka.model_kafka_health_response import ModelKafkaHealthResponse


class ModelGetHealthOutput(BaseModel):
    """Output model for get_health operation."""

    success: bool = Field(
        description="Whether health check completed successfully"
    )
    health_data: ModelKafkaHealthResponse = Field(
        description="Comprehensive Kafka producer pool health data"
    )
    timestamp: datetime = Field(
        description="Health check timestamp"
    )