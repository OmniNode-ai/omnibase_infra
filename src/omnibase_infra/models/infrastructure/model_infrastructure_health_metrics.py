"""Infrastructure Health Metrics Model.

Pydantic model for aggregated infrastructure health metrics, extracted from
infrastructure_health_monitor.py for shared usage across ONEX nodes.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from omnibase_infra.models.circuit_breaker.model_circuit_breaker_metrics import (
    ModelCircuitBreakerMetrics,
)
from omnibase_infra.models.kafka.model_kafka_producer_pool_stats import (
    ModelKafkaProducerPoolStats,
)
from omnibase_infra.models.postgres.model_postgres_performance_metrics import (
    ModelPostgresPerformanceMetrics,
)


class ModelInfrastructureHealthMetrics(BaseModel):
    """Model for aggregated infrastructure health metrics."""

    # Overall health
    overall_status: str = Field(
        description="Overall health status: healthy, degraded, unhealthy",
    )

    timestamp: float = Field(
        description="Unix timestamp of health check",
    )

    environment: str = Field(
        description="Target environment name",
    )

    # Component statuses
    postgres_healthy: bool = Field(
        description="PostgreSQL connection health status",
    )

    kafka_healthy: bool = Field(
        description="Kafka producer health status",
    )

    circuit_breaker_healthy: bool = Field(
        description="Circuit breaker health status",
    )

    # Detailed metrics - using strongly typed models per ONEX standards
    postgres_metrics: ModelPostgresPerformanceMetrics = Field(
        description="Detailed PostgreSQL performance metrics",
    )

    kafka_metrics: ModelKafkaProducerPoolStats = Field(
        description="Detailed Kafka producer pool statistics",
    )

    circuit_breaker_metrics: ModelCircuitBreakerMetrics = Field(
        description="Detailed circuit breaker metrics",
    )

    # Aggregate statistics
    total_connections: int = Field(
        ge=0,
        description="Total number of active connections",
    )

    total_messages_processed: int = Field(
        ge=0,
        description="Total messages processed",
    )

    total_events_queued: int = Field(
        ge=0,
        description="Total events in queues",
    )

    error_rate_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Error rate percentage",
    )

    # Performance indicators
    avg_db_response_time_ms: float = Field(
        ge=0.0,
        description="Average database response time in milliseconds",
    )

    avg_kafka_throughput_mps: float = Field(
        ge=0.0,
        description="Average Kafka throughput in messages per second",
    )

    circuit_breaker_success_rate: float = Field(
        ge=0.0,
        le=100.0,
        description="Circuit breaker success rate percentage",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
