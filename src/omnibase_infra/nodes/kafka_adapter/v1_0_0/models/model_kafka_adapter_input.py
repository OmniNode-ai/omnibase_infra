"""Kafka adapter input model - node-specific envelope for message bus integration."""

from typing import Optional
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, Field

from ....enums.enum_kafka_operation_type import EnumKafkaOperationType
from ....models.kafka.model_kafka_message import ModelKafkaMessage
from ....models.kafka.model_kafka_topic_config import ModelKafkaTopicConfig
from ....models.kafka.model_kafka_producer_config import ModelKafkaProducerConfig
from ....models.kafka.model_kafka_consumer_config import ModelKafkaConsumerConfig
from ....models.common.model_request_context import ModelRequestContext


class ModelKafkaAdapterInput(BaseModel):
    """Kafka adapter input envelope for message bus operations."""

    operation_type: EnumKafkaOperationType = Field(
        description="Type of Kafka operation to perform"
    )
    
    # Operation-specific payloads (using shared models)
    message: Optional[ModelKafkaMessage] = Field(
        default=None, 
        description="Kafka message payload (for produce operations)"
    )
    topic_config: Optional[ModelKafkaTopicConfig] = Field(
        default=None,
        description="Topic configuration (for topic operations)"
    )
    producer_config: Optional[ModelKafkaProducerConfig] = Field(
        default=None,
        description="Producer configuration (for produce operations)"
    )
    consumer_config: Optional[ModelKafkaConsumerConfig] = Field(
        default=None,
        description="Consumer configuration (for consume operations)"
    )
    
    # Envelope metadata
    correlation_id: UUID = Field(description="Request correlation ID for tracing")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    context: Optional[ModelRequestContext] = Field(
        default=None, 
        description="Additional request context"
    )
    
    # Operation-specific parameters
    timeout_seconds: Optional[float] = Field(
        default=30.0,
        description="Operation timeout in seconds"
    )
    retry_count: int = Field(default=3, description="Number of retries on failure")
    
    def model_post_init(self, __context: Optional[dict]) -> None:
        """Validate operation-specific payload requirements."""
        if self.operation_type == EnumKafkaOperationType.PRODUCE:
            if not self.message:
                raise ValueError("message is required for produce operations")
        elif self.operation_type == EnumKafkaOperationType.CONSUME:
            if not self.consumer_config:
                raise ValueError("consumer_config is required for consume operations")
        elif self.operation_type in (EnumKafkaOperationType.TOPIC_CREATE, EnumKafkaOperationType.TOPIC_DELETE):
            if not self.topic_config:
                raise ValueError("topic_config is required for topic operations")