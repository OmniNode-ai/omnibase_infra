"""Strongly typed Kafka message payload models for ONEX compliance."""

from typing import Union, Dict
from pydantic import BaseModel, Field


class ModelKafkaJsonPayload(BaseModel):
    """JSON payload data structure for Kafka messages."""
    
    data: Dict[str, Union[str, int, float, bool]] = Field(
        description="JSON payload data with strongly typed values"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata for the payload"
    )


class ModelKafkaEventPayload(BaseModel):
    """Event payload structure for Kafka event messages."""
    
    event_type: str = Field(description="Type of the event")
    event_data: Dict[str, Union[str, int, float, bool]] = Field(
        description="Event-specific data with strongly typed values"
    )
    event_version: str = Field(default="1.0", description="Event schema version")
    event_source: str = Field(description="Source system of the event")


class ModelKafkaTransactionPayload(BaseModel):
    """Transaction payload structure for transactional Kafka messages."""
    
    transaction_id: str = Field(description="Unique transaction identifier")
    operation_type: str = Field(description="Type of database operation")
    table_name: str = Field(description="Target table name")
    operation_data: Dict[str, Union[str, int, float, bool]] = Field(
        description="Operation data with strongly typed values"
    )


# Union type for all supported Kafka message payload types
KafkaMessagePayload = Union[
    ModelKafkaJsonPayload,
    ModelKafkaEventPayload, 
    ModelKafkaTransactionPayload,
    str,  # String payload
    bytes  # Binary payload
]