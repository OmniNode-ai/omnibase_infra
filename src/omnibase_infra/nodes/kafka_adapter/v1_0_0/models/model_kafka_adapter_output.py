"""Kafka adapter output model - node-specific envelope for message bus responses."""

from typing import List, Optional, Any
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, Field

from ....enums.enum_kafka_operation_type import EnumKafkaOperationType
from ....models.kafka.model_kafka_message import ModelKafkaMessage
from ....models.kafka.model_kafka_health_response import ModelKafkaHealthResponse


class ModelKafkaAdapterOutput(BaseModel):
    """Kafka adapter output envelope for message bus responses."""

    operation_type: EnumKafkaOperationType = Field(
        description="Type of operation that was executed"
    )
    
    # Operation results (using shared models)
    messages: List[ModelKafkaMessage] = Field(
        default_factory=list,
        description="Consumed messages (for consume operations)"
    )
    topic_info: Optional[dict[str, Any]] = Field(
        default=None,
        description="Topic information (for topic operations)"
    )
    health_response: Optional[ModelKafkaHealthResponse] = Field(
        default=None,
        description="Health check response payload"
    )
    
    # Operation status
    success: bool = Field(description="Whether the operation was successful")
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if operation failed"
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Error code for programmatic handling"
    )
    
    # Envelope metadata
    correlation_id: UUID = Field(description="Request correlation ID for tracing")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    execution_time_ms: float = Field(description="Total operation execution time in milliseconds")
    
    # Operation metrics
    record_count: int = Field(default=0, description="Number of records processed")
    bytes_processed: int = Field(default=0, description="Total bytes processed")
    partition_count: Optional[int] = Field(
        default=None,
        description="Number of partitions involved (for topic operations)"
    )
    
    # Additional context
    context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional response context"
    )
    
    # Kafka-specific metrics
    offset_info: Optional[dict[str, Any]] = Field(
        default=None,
        description="Kafka offset information (for produce/consume operations)"
    )
    broker_info: Optional[dict[str, Any]] = Field(
        default=None,
        description="Broker information used for the operation"
    )