"""Kafka shared models package for ONEX infrastructure.

This package contains shared Kafka models following the DRY pattern:
- Reusable models for Kafka operations (produce, consume, topic management)
- Referenced as dependencies by Kafka adapter and other Kafka-related nodes
- One model per file with proper Pydantic inheritance
"""

from .model_kafka_message import ModelKafkaMessage
from .model_kafka_message_payload import (
    ModelKafkaJsonPayload,
    ModelKafkaEventPayload,
    ModelKafkaTransactionPayload,
    KafkaMessagePayload
)
from .model_kafka_security_config import (
    ModelKafkaSSLConfig,
    ModelKafkaSASLConfig,
    ModelKafkaSecurityConfig
)
from .model_kafka_topic_config import ModelKafkaTopicConfig
from .model_kafka_producer_config import ModelKafkaProducerConfig
from .model_kafka_consumer_config import ModelKafkaConsumerConfig
from .model_kafka_health_response import ModelKafkaHealthResponse

__all__ = [
    "ModelKafkaMessage",
    "ModelKafkaJsonPayload",
    "ModelKafkaEventPayload", 
    "ModelKafkaTransactionPayload",
    "KafkaMessagePayload",
    "ModelKafkaSSLConfig",
    "ModelKafkaSASLConfig",
    "ModelKafkaSecurityConfig",
    "ModelKafkaTopicConfig", 
    "ModelKafkaProducerConfig",
    "ModelKafkaConsumerConfig",
    "ModelKafkaHealthResponse",
]