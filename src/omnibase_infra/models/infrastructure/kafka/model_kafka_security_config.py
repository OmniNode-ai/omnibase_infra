"""Kafka security configuration model."""

from pydantic import BaseModel, Field

from .model_kafka_sasl_config import ModelKafkaSASLConfig
from .model_kafka_ssl_config import ModelKafkaSSLConfig


class ModelKafkaSecurityConfig(BaseModel):
    """Kafka security configuration model."""

    security_protocol: str = Field(
        default="PLAINTEXT",
        description="Security protocol (PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL)",
    )
    ssl_config: ModelKafkaSSLConfig | None = Field(
        default=None, description="SSL/TLS configuration",
    )
    sasl_config: ModelKafkaSASLConfig | None = Field(
        default=None, description="SASL authentication configuration",
    )
    enable_auto_commit: bool = Field(
        default=True, description="Enable automatic offset commits",
    )
    auto_commit_interval_ms: int = Field(
        default=5000, description="Auto commit interval in milliseconds",
    )
    session_timeout_ms: int = Field(
        default=10000, description="Session timeout in milliseconds",
    )
    heartbeat_interval_ms: int = Field(
        default=3000, description="Heartbeat interval in milliseconds",
    )
    max_poll_interval_ms: int = Field(
        default=300000, description="Maximum poll interval in milliseconds",
    )
    connections_max_idle_ms: int = Field(
        default=540000, description="Connection max idle time in milliseconds",
    )
    request_timeout_ms: int = Field(
        default=30000, description="Request timeout in milliseconds",
    )
    retry_backoff_ms: int = Field(
        default=100, description="Retry backoff time in milliseconds",
    )
    reconnect_backoff_ms: int = Field(
        default=50, description="Reconnect backoff time in milliseconds",
    )
    reconnect_backoff_max_ms: int = Field(
        default=1000, description="Maximum reconnect backoff time in milliseconds",
    )
