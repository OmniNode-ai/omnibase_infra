"""Kafka adapter output model - node-specific envelope for message bus responses."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from ....enums.enum_kafka_operation_type import EnumKafkaOperationType
from ....models.common.model_kafka_metadata import (
    ModelKafkaBrokerInfo,
    ModelKafkaOffsetInfo,
    ModelKafkaTopicInfo,
)
from ....models.common.model_request_context import ModelRequestContext
from ....models.kafka.model_kafka_health_response import ModelKafkaHealthResponse
from ....models.kafka.model_kafka_message import ModelKafkaMessage


class ModelKafkaAdapterOutput(BaseModel):
    """Kafka adapter output envelope for message bus responses."""

    operation_type: EnumKafkaOperationType = Field(
        description="Type of operation that was executed",
    )

    # Operation results (using shared models)
    messages: list[ModelKafkaMessage] = Field(
        default_factory=list,
        description="Consumed messages (for consume operations)",
    )
    topic_info: ModelKafkaTopicInfo | None = Field(
        default=None,
        description="Topic information (for topic operations)",
    )
    health_response: ModelKafkaHealthResponse | None = Field(
        default=None,
        description="Health check response payload",
    )

    # Operation status
    success: bool = Field(description="Whether the operation was successful")
    error_message: str | None = Field(
        default=None,
        description="Error message if operation failed",
    )
    error_code: str | None = Field(
        default=None,
        description="Error code for programmatic handling",
    )

    # Envelope metadata
    correlation_id: UUID = Field(description="Request correlation ID for tracing")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    execution_time_ms: float = Field(description="Total operation execution time in milliseconds")

    # Operation metrics
    record_count: int = Field(default=0, description="Number of records processed")
    bytes_processed: int = Field(default=0, description="Total bytes processed")
    partition_count: int | None = Field(
        default=None,
        description="Number of partitions involved (for topic operations)",
    )

    # Additional context
    context: ModelRequestContext | None = Field(
        default=None,
        description="Additional response context",
    )

    # Kafka-specific metrics
    offset_info: ModelKafkaOffsetInfo | None = Field(
        default=None,
        description="Kafka offset information (for produce/consume operations)",
    )
    broker_info: ModelKafkaBrokerInfo | None = Field(
        default=None,
        description="Broker information used for the operation",
    )
