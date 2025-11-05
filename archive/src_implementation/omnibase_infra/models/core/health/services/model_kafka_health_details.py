"""Kafka health details model implementing ProtocolHealthDetails."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from omnibase_spi.protocols.types.core_types import HealthStatus

from omnibase_infra.enums import EnumHealthStatus


class ModelKafkaHealthDetails(BaseModel):
    """Kafka-specific health details with self-assessment capability."""

    kafka_producer_count: int | None = Field(
        default=None,
        ge=0,
        description="Current Kafka producer count",
    )

    kafka_consumer_count: int | None = Field(
        default=None,
        ge=0,
        description="Current Kafka consumer count",
    )

    kafka_last_error: str | None = Field(
        default=None,
        max_length=500,
        description="Last Kafka error message",
    )

    producer_lag_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Average producer lag in milliseconds",
    )

    consumer_lag_messages: int | None = Field(
        default=None,
        ge=0,
        description="Consumer lag in number of messages",
    )

    broker_connectivity: bool | None = Field(
        default=None,
        description="Whether brokers are reachable",
    )

    topic_partition_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of topic partitions available",
    )

    def get_health_status(self) -> "HealthStatus":
        """Assess Kafka health status based on service metrics."""
        # Critical failures
        if self.kafka_last_error:
            return EnumHealthStatus.UNHEALTHY

        if self.broker_connectivity is False:
            return EnumHealthStatus.CRITICAL

        # Warning conditions
        if self.consumer_lag_messages and self.consumer_lag_messages > 10000:
            return EnumHealthStatus.WARNING

        if self.producer_lag_ms and self.producer_lag_ms > 1000:  # 1+ second lag
            return EnumHealthStatus.WARNING

        # Performance degradation
        if self.consumer_lag_messages and self.consumer_lag_messages > 1000:
            return EnumHealthStatus.DEGRADED

        return EnumHealthStatus.HEALTHY

    def is_healthy(self) -> bool:
        """Return True if Kafka is considered healthy."""
        return self.get_health_status() == EnumHealthStatus.HEALTHY

    def get_health_summary(self) -> str:
        """Generate human-readable Kafka health summary."""
        status = self.get_health_status()

        if status == EnumHealthStatus.UNHEALTHY:
            return f"Kafka Error: {self.kafka_last_error}"

        if status == EnumHealthStatus.CRITICAL:
            return "Kafka Critical: Brokers unreachable"

        if status == EnumHealthStatus.WARNING:
            if self.consumer_lag_messages and self.consumer_lag_messages > 10000:
                return f"Kafka Warning: High consumer lag ({self.consumer_lag_messages} messages)"
            if self.producer_lag_ms and self.producer_lag_ms > 1000:
                return f"Kafka Warning: High producer lag ({self.producer_lag_ms}ms)"

        if status == EnumHealthStatus.DEGRADED:
            return f"Kafka Degraded: Consumer lag ({self.consumer_lag_messages} messages)"

        summary_parts = []
        if self.kafka_producer_count is not None:
            summary_parts.append(f"{self.kafka_producer_count} producers")
        if self.kafka_consumer_count is not None:
            summary_parts.append(f"{self.kafka_consumer_count} consumers")

        if summary_parts:
            return f"Kafka Healthy: {', '.join(summary_parts)}"

        return "Kafka connections healthy"
