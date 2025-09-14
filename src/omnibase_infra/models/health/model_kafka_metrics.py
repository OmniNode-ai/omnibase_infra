"""Kafka Metrics Model.

Strongly-typed model for Kafka component health metrics.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ModelKafkaMetrics(BaseModel):
    """Model for Kafka component health metrics."""

    # Producer metrics
    producer_active_count: int = Field(
        ge=0,
        description="Number of active Kafka producers"
    )

    producer_pool_size: int = Field(
        ge=0,
        description="Total producer pool size"
    )

    producer_success_rate: float = Field(
        ge=0.0,
        le=100.0,
        description="Producer success rate percentage"
    )

    messages_sent_total: int = Field(
        ge=0,
        description="Total number of messages sent"
    )

    messages_per_second: float = Field(
        ge=0.0,
        description="Average messages sent per second"
    )

    # Consumer metrics (if applicable)
    consumer_active_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of active Kafka consumers"
    )

    messages_consumed_total: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total number of messages consumed"
    )

    consumer_lag_total: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total consumer lag across all partitions"
    )

    # Topic metrics
    topics_count: int = Field(
        ge=0,
        description="Number of topics being used"
    )

    partitions_count: int = Field(
        ge=0,
        description="Total number of partitions across all topics"
    )

    # Performance metrics
    avg_send_latency_ms: float = Field(
        ge=0.0,
        description="Average message send latency in milliseconds"
    )

    avg_batch_size: float = Field(
        ge=0.0,
        description="Average batch size for producer operations"
    )

    compression_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Message compression ratio percentage"
    )

    # Error metrics
    send_failures_total: int = Field(
        ge=0,
        description="Total number of send failures"
    )

    connection_errors: int = Field(
        ge=0,
        description="Number of Kafka connection errors"
    )

    timeout_errors: int = Field(
        ge=0,
        description="Number of timeout errors"
    )

    serialization_errors: int = Field(
        ge=0,
        description="Number of message serialization errors"
    )

    # Resource utilization
    memory_usage_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Kafka client memory usage in megabytes"
    )

    buffer_memory_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Producer buffer memory usage percentage"
    )

    # Connection health
    broker_connections: int = Field(
        ge=0,
        description="Number of active broker connections"
    )

    metadata_age_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Age of metadata cache in milliseconds"
    )

    # Topic-specific metrics
    active_topics: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of topics with active message traffic"
    )

    high_throughput_topics: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of topics with high message throughput"
    )

    # Throughput metrics
    bytes_sent_per_second: float = Field(
        ge=0.0,
        description="Average bytes sent per second"
    )

    bytes_received_per_second: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Average bytes received per second (for consumers)"
    )