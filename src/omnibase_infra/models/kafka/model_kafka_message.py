"""Kafka message model for message streaming integration."""

from typing import Dict, Optional, Union
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, Field

from ...enums.enum_kafka_message_format import EnumKafkaMessageFormat
from .model_kafka_message_payload import KafkaMessagePayload


class ModelKafkaMessage(BaseModel):
    """Kafka message model."""

    topic: str = Field(description="Kafka topic name")
    key: Optional[Union[str, bytes]] = Field(default=None, description="Message key for partitioning")
    value: KafkaMessagePayload = Field(description="Message payload with strongly typed structure")
    headers: Dict[str, Union[str, bytes]] = Field(
        default_factory=dict, 
        description="Message headers"
    )
    partition: Optional[int] = Field(default=None, description="Target partition (if specified)")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")
    format: EnumKafkaMessageFormat = Field(
        default=EnumKafkaMessageFormat.JSON, 
        description="Message format type"
    )
    correlation_id: Optional[UUID] = Field(default=None, description="Message correlation ID")
    message_id: Optional[str] = Field(default=None, description="Unique message identifier")
    schema_version: Optional[str] = Field(default=None, description="Message schema version")
    compression_type: Optional[str] = Field(default=None, description="Message compression type")