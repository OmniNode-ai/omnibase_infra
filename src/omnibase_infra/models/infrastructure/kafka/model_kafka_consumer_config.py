"""Kafka consumer configuration model."""

from pydantic import BaseModel, Field

from omnibase_infra.models.infrastructure.kafka.model_kafka_security_config import (
    ModelKafkaSecurityConfig,
)


class ModelKafkaConsumerConfig(BaseModel):
    """Kafka consumer configuration model."""

    bootstrap_servers: str = Field(
        description="Kafka bootstrap servers (comma-separated)",
    )
    group_id: str = Field(description="Consumer group ID")
    client_id: str | None = Field(default=None, description="Consumer client ID")
    topics: list[str] = Field(description="List of topics to subscribe to")
    auto_offset_reset: str = Field(
        default="latest",
        description="Offset reset policy (earliest, latest, none)",
    )
    enable_auto_commit: bool = Field(
        default=True,
        description="Enable automatic offset commits",
    )
    auto_commit_interval_ms: int = Field(
        default=5000,  # 5 seconds
        description="Auto commit interval in milliseconds",
    )
    session_timeout_ms: int = Field(
        default=10000,  # 10 seconds
        description="Session timeout in milliseconds",
    )
    heartbeat_interval_ms: int = Field(
        default=3000,  # 3 seconds
        description="Heartbeat interval in milliseconds",
    )
    max_poll_records: int = Field(
        default=500,
        description="Maximum number of records per poll",
    )
    max_poll_interval_ms: int = Field(
        default=300000,  # 5 minutes
        description="Maximum poll interval in milliseconds",
    )
    fetch_min_bytes: int = Field(
        default=1,
        description="Minimum bytes to fetch per request",
    )
    fetch_max_wait_ms: int = Field(
        default=500,
        description="Maximum wait time for fetch requests",
    )
    max_partition_fetch_bytes: int = Field(
        default=1048576,  # 1MB
        description="Maximum bytes per partition to fetch",
    )
    isolation_level: str = Field(
        default="read_uncommitted",
        description="Transaction isolation level (read_uncommitted, read_committed)",
    )
    security_config: ModelKafkaSecurityConfig = Field(
        default_factory=ModelKafkaSecurityConfig,
        description="Strongly typed security configuration (SSL, SASL, etc.)",
    )
