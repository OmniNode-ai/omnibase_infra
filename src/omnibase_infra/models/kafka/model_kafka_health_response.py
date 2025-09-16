"""Kafka health check response model."""

from datetime import datetime

from pydantic import BaseModel, Field


class ModelKafkaHealthResponse(BaseModel):
    """Kafka health check response model."""

    is_healthy: bool = Field(description="Whether Kafka cluster is healthy")
    cluster_id: str | None = Field(default=None, description="Kafka cluster ID")
    broker_count: int = Field(default=0, description="Number of available brokers")
    broker_ids: list[int] = Field(default_factory=list, description="List of broker IDs")
    topic_count: int = Field(default=0, description="Total number of topics")
    partition_count: int = Field(default=0, description="Total number of partitions")
    under_replicated_partitions: int = Field(
        default=0,
        description="Number of under-replicated partitions",
    )
    offline_partitions: int = Field(default=0, description="Number of offline partitions")
    controller_id: int | None = Field(default=None, description="Current controller broker ID")
    response_time_ms: float = Field(description="Health check response time in milliseconds")
    timestamp: datetime = Field(description="Health check timestamp")
    version: str | None = Field(default=None, description="Kafka version")
    errors: list[str] = Field(default_factory=list, description="Any health check errors")
    broker_details: dict[int, dict[str, str]] = Field(
        default_factory=dict,
        description="Detailed broker information (broker_id -> details)",
    )
    lag_info: dict[str, int] | None = Field(
        default=None,
        description="Consumer group lag information",
    )
