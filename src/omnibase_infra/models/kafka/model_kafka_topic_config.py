"""Kafka topic configuration model."""

from typing import Optional
from pydantic import BaseModel, Field

from .model_kafka_topic_overrides import ModelKafkaTopicOverrides


class ModelKafkaTopicConfig(BaseModel):
    """Kafka topic configuration model."""

    topic_name: str = Field(description="Name of the Kafka topic")
    num_partitions: int = Field(default=1, description="Number of partitions for the topic")
    replication_factor: int = Field(default=1, description="Replication factor for the topic")
    config_overrides: Optional[ModelKafkaTopicOverrides] = Field(
        default=None,
        description="Topic configuration overrides (e.g., retention.ms, cleanup.policy)"
    )
    cleanup_policy: Optional[str] = Field(
        default="delete", 
        description="Topic cleanup policy (delete, compact, or delete,compact)"
    )
    retention_ms: Optional[int] = Field(
        default=604800000,  # 7 days
        description="Message retention time in milliseconds"
    )
    segment_ms: Optional[int] = Field(
        default=604800000,  # 7 days
        description="Log segment time in milliseconds"
    )
    max_message_bytes: Optional[int] = Field(
        default=1000000,  # 1MB
        description="Maximum message size in bytes"
    )
    min_in_sync_replicas: Optional[int] = Field(
        default=1,
        description="Minimum number of in-sync replicas"
    )