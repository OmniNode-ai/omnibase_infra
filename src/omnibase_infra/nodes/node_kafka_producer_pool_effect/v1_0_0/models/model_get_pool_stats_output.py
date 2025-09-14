"""Get pool stats output model for Kafka producer pool EFFECT node."""

from datetime import datetime

from pydantic import BaseModel, Field

from omnibase_infra.models.kafka.model_kafka_producer_pool_stats import ModelKafkaProducerPoolStats


class ModelGetPoolStatsOutput(BaseModel):
    """Output model for get_pool_stats operation."""

    success: bool = Field(
        description="Whether stats retrieval completed successfully"
    )
    stats: ModelKafkaProducerPoolStats = Field(
        description="Comprehensive Kafka producer pool statistics"
    )
    timestamp: datetime = Field(
        description="Stats collection timestamp"
    )