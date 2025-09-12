"""Kafka producer configuration model."""

from typing import Optional
from pydantic import BaseModel, Field

from .model_kafka_security_config import ModelKafkaSecurityConfig


class ModelKafkaProducerConfig(BaseModel):
    """Kafka producer configuration model."""

    bootstrap_servers: str = Field(description="Kafka bootstrap servers (comma-separated)")
    client_id: Optional[str] = Field(default=None, description="Producer client ID")
    acks: str = Field(default="1", description="Acknowledgment level (0, 1, all)")
    retries: int = Field(default=3, description="Number of retries on failure")
    batch_size: int = Field(default=16384, description="Batch size in bytes")
    linger_ms: int = Field(default=0, description="Time to wait for batching")
    buffer_memory: int = Field(default=33554432, description="Total memory buffer size")
    compression_type: Optional[str] = Field(
        default="none", 
        description="Compression type (none, gzip, snappy, lz4, zstd)"
    )
    max_request_size: int = Field(
        default=1048576,  # 1MB
        description="Maximum request size in bytes"
    )
    request_timeout_ms: int = Field(
        default=30000,  # 30 seconds
        description="Request timeout in milliseconds"
    )
    delivery_timeout_ms: int = Field(
        default=120000,  # 2 minutes
        description="Delivery timeout in milliseconds"
    )
    max_in_flight_requests_per_connection: int = Field(
        default=5,
        description="Maximum unacknowledged requests per connection"
    )
    enable_idempotence: bool = Field(
        default=True,
        description="Enable idempotent producer"
    )
    security_config: Optional[ModelKafkaSecurityConfig] = Field(
        default=None,
        description="Security configuration (SSL, SASL, etc.)"
    )