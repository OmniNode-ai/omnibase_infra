"""Kafka message model for message streaming integration."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from ...enums.enum_kafka_message_format import EnumKafkaMessageFormat
from .model_kafka_message_payload import KafkaMessagePayload


class ModelKafkaMessage(BaseModel):
    """Kafka message model."""

    topic: str = Field(description="Kafka topic name")
    key: str | bytes | None = Field(default=None, description="Message key for partitioning")
    value: KafkaMessagePayload = Field(description="Message payload with strongly typed structure")
    headers: dict[str, str | bytes] = Field(
        default_factory=dict,
        description="Message headers",
    )
    partition: int | None = Field(default=None, description="Target partition (if specified)")
    timestamp: datetime | None = Field(default=None, description="Message timestamp")
    format: EnumKafkaMessageFormat = Field(
        default=EnumKafkaMessageFormat.JSON,
        description="Message format type",
    )
    correlation_id: UUID | None = Field(default=None, description="Message correlation ID")
    message_id: str | None = Field(default=None, description="Unique message identifier")
    schema_version: str | None = Field(default=None, description="Message schema version")
    compression_type: str | None = Field(default=None, description="Message compression type")
