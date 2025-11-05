"""Consumer health status model."""

from datetime import datetime

from pydantic import BaseModel, Field


class ModelConsumerHealthStatus(BaseModel):
    """Kafka consumer health status and diagnostics.

    Provides comprehensive health information for a Kafka consumer instance,
    including connectivity, lag, and performance metrics.
    """

    consumer_id: str = Field(description="Unique consumer identifier")
    group_id: str = Field(description="Consumer group identifier")
    status: str = Field(
        description="Health status: healthy, degraded, unhealthy, disconnected",
    )

    # Connection state
    is_connected: bool = Field(description="Whether consumer is connected to broker")
    broker_connections: int = Field(ge=0, description="Number of active broker connections")

    # Consumption metrics
    messages_consumed: int = Field(ge=0, default=0, description="Total messages consumed")
    bytes_consumed: int = Field(ge=0, default=0, description="Total bytes consumed")
    average_throughput_mps: float = Field(
        ge=0,
        default=0.0,
        description="Average throughput in messages per second",
    )

    # Lag metrics
    current_lag: int = Field(ge=0, default=0, description="Current consumer lag")
    max_lag_threshold: int = Field(
        ge=0,
        default=10000,
        description="Maximum acceptable lag threshold",
    )
    is_lagging: bool = Field(
        default=False,
        description="Whether consumer lag exceeds threshold",
    )

    # Error tracking
    error_count: int = Field(ge=0, default=0, description="Number of errors encountered")
    consecutive_errors: int = Field(
        ge=0,
        default=0,
        description="Number of consecutive errors",
    )
    last_error: str | None = Field(default=None, description="Last error message")
    last_error_time: datetime | None = Field(
        default=None,
        description="Timestamp of last error",
    )

    # Timing
    last_poll_time: datetime | None = Field(
        default=None,
        description="Last successful poll timestamp",
    )
    last_commit_time: datetime | None = Field(
        default=None,
        description="Last offset commit timestamp",
    )
    uptime_seconds: int = Field(ge=0, default=0, description="Consumer uptime in seconds")

    # Assignment
    assigned_partitions: list[str] = Field(
        default_factory=list,
        description="Currently assigned topic-partitions",
    )
    partition_count: int = Field(ge=0, default=0, description="Number of assigned partitions")

    def determine_health_status(self) -> str:
        """Determine overall health status based on metrics.

        Returns:
            Health status string: healthy, degraded, unhealthy, disconnected
        """
        if not self.is_connected:
            return "disconnected"

        if self.consecutive_errors >= 5 or self.error_count > 100:
            return "unhealthy"

        if self.is_lagging or self.consecutive_errors > 0:
            return "degraded"

        return "healthy"

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"
