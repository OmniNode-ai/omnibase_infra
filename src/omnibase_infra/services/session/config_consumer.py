"""Configuration for session event consumers.

Loads from environment variables with OMNIBASE_INFRA_SESSION_CONSUMER_ prefix.

Moved from omniclaude as part of OMN-1526 architectural cleanup.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigSessionConsumer(BaseSettings):
    """Configuration for the Claude session event Kafka consumer.

    Environment variables use the OMNIBASE_INFRA_SESSION_CONSUMER_ prefix.
    Example: OMNIBASE_INFRA_SESSION_CONSUMER_BOOTSTRAP_SERVERS=kafka.example.com:9092
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_SESSION_CONSUMER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka connection
    bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers. Set via OMNIBASE_INFRA_SESSION_CONSUMER_BOOTSTRAP_SERVERS env var for production.",
    )
    group_id: str = Field(
        default="omnibase-infra-session-consumer",
        description="Consumer group ID",
    )

    # Topics to subscribe
    topics: list[str] = Field(
        default=[
            "dev.omniclaude.session.started.v1",
            "dev.omniclaude.session.ended.v1",
            "dev.omniclaude.prompt.submitted.v1",
            "dev.omniclaude.tool.executed.v1",
        ],
        description="Kafka topics to consume",
    )

    # Consumer behavior
    auto_offset_reset: str = Field(
        default="earliest",
        description="Where to start consuming if no offset exists",
    )
    enable_auto_commit: bool = Field(
        default=False,
        description="Disable auto-commit for at-least-once delivery",
    )
    max_poll_records: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum records per poll",
    )

    # Processing
    batch_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Timeout for batch processing in milliseconds",
    )

    # Circuit breaker
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Failures before circuit opens",
    )
    circuit_breaker_timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Time before circuit half-opens",
    )
