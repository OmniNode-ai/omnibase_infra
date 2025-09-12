"""Kafka consumer configuration model."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class ModelKafkaConsumerConfig(BaseModel):
    """Kafka consumer configuration model."""

    bootstrap_servers: str = Field(description="Kafka bootstrap servers (comma-separated)")
    group_id: str = Field(description="Consumer group ID")
    client_id: Optional[str] = Field(default=None, description="Consumer client ID")
    topics: List[str] = Field(description="List of topics to subscribe to")
    auto_offset_reset: str = Field(
        default="latest", 
        description="Offset reset policy (earliest, latest, none)"
    )
    enable_auto_commit: bool = Field(
        default=True,
        description="Enable automatic offset commits"
    )
    auto_commit_interval_ms: int = Field(
        default=5000,  # 5 seconds
        description="Auto commit interval in milliseconds"
    )
    session_timeout_ms: int = Field(
        default=10000,  # 10 seconds
        description="Session timeout in milliseconds"
    )
    heartbeat_interval_ms: int = Field(
        default=3000,  # 3 seconds
        description="Heartbeat interval in milliseconds"
    )
    max_poll_records: int = Field(
        default=500,
        description="Maximum number of records per poll"
    )
    max_poll_interval_ms: int = Field(
        default=300000,  # 5 minutes
        description="Maximum poll interval in milliseconds"
    )
    fetch_min_bytes: int = Field(
        default=1,
        description="Minimum bytes to fetch per request"
    )
    fetch_max_wait_ms: int = Field(
        default=500,
        description="Maximum wait time for fetch requests"
    )
    max_partition_fetch_bytes: int = Field(
        default=1048576,  # 1MB
        description="Maximum bytes per partition to fetch"
    )
    isolation_level: str = Field(
        default="read_uncommitted",
        description="Transaction isolation level (read_uncommitted, read_committed)"
    )
    security_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Security configuration (SSL, SASL, etc.)"
    )