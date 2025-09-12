"""Kafka adapter node-specific models.

This package contains models specific to the Kafka adapter node:
- Input/output envelope models for message bus integration
- Configuration models specific to this adapter node

Shared Kafka models are located in omnibase_infra.models.kafka.
"""

from .model_kafka_adapter_input import ModelKafkaAdapterInput
from .model_kafka_adapter_output import ModelKafkaAdapterOutput

__all__ = [
    "ModelKafkaAdapterInput",
    "ModelKafkaAdapterOutput",
]