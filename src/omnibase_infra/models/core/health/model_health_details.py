"""Health Status Details Model.

DEPRECATED: This model is being replaced by service-specific health models
that implement ProtocolHealthDetails from omnibase_spi for better encapsulation
and self-contained health assessment logic.

Use the service-specific models instead:
- ModelPostgresHealthDetails for PostgreSQL health
- ModelKafkaHealthDetails for Kafka health
- ModelCircuitBreakerHealthDetails for circuit breaker health
- ModelSystemHealthDetails for general system health

These models provide self-assessment capabilities and follow the protocol-based
architecture pattern for better maintainability and service isolation.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from omnibase_spi.protocols.types.core_types import HealthStatus

from omnibase_infra.enums import EnumCircuitBreakerState, EnumHealthStatus
from omnibase_infra.models.core.health.services import (
    ModelCircuitBreakerHealthDetails,
    ModelKafkaHealthDetails,
    ModelPostgresHealthDetails,
    ModelSystemHealthDetails,
)


class ModelHealthDetails(BaseModel):
    """
    DEPRECATED: Composite health details model for backward compatibility.

    This model now delegates to service-specific health models that implement
    ProtocolHealthDetails for proper health assessment and reporting.

    New code should use the service-specific models directly.
    """

    # Service-specific health details (preferred approach)
    postgres_health: ModelPostgresHealthDetails | None = Field(
        default=None,
        description="PostgreSQL service health details",
    )

    kafka_health: ModelKafkaHealthDetails | None = Field(
        default=None,
        description="Kafka service health details",
    )

    circuit_breaker_health: ModelCircuitBreakerHealthDetails | None = Field(
        default=None,
        description="Circuit breaker health details",
    )

    system_health: ModelSystemHealthDetails | None = Field(
        default=None,
        description="System-level health details",
    )

    # Legacy fields (maintained for backward compatibility, will be removed)
    postgres_connection_count: int | None = Field(
        default=None,
        ge=0,
        description="DEPRECATED: Use postgres_health.postgres_connection_count",
    )

    postgres_last_error: str | None = Field(
        default=None,
        max_length=500,
        description="DEPRECATED: Use postgres_health.postgres_last_error",
    )

    kafka_producer_count: int | None = Field(
        default=None,
        ge=0,
        description="DEPRECATED: Use kafka_health.kafka_producer_count",
    )

    kafka_last_error: str | None = Field(
        default=None,
        max_length=500,
        description="DEPRECATED: Use kafka_health.kafka_last_error",
    )

    circuit_breaker_state: EnumCircuitBreakerState | None = Field(
        default=None,
        description="DEPRECATED: Use circuit_breaker_health.circuit_breaker_state",
    )

    circuit_breaker_failure_count: int | None = Field(
        default=None,
        ge=0,
        description="DEPRECATED: Use circuit_breaker_health.circuit_breaker_failure_count",
    )

    # Legacy system metrics (maintained for backward compatibility, will be removed)
    peak_memory_usage_mb: float | None = Field(
        default=None,
        ge=0.0,
        description="DEPRECATED: Use system_health.peak_memory_usage_mb",
    )

    average_cpu_usage_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="DEPRECATED: Use system_health.average_cpu_usage_percent",
    )

    disk_space_available_gb: float | None = Field(
        default=None,
        ge=0.0,
        description="DEPRECATED: Use system_health.disk_space_available_gb",
    )

    network_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="DEPRECATED: Use system_health.network_latency_ms",
    )

    external_service_count: int | None = Field(
        default=None,
        ge=0,
        description="DEPRECATED: Use system_health.external_service_count",
    )

    external_services_healthy: int | None = Field(
        default=None,
        ge=0,
        description="DEPRECATED: Use system_health.external_services_healthy",
    )

    environment_variables_loaded: int | None = Field(
        default=None,
        ge=0,
        description="DEPRECATED: Use system_health.environment_variables_loaded",
    )

    configuration_files_loaded: int | None = Field(
        default=None,
        ge=0,
        description="DEPRECATED: Use system_health.configuration_files_loaded",
    )

    # Legacy security and tracking fields (maintained for backward compatibility, will be removed)
    ssl_certificates_valid: bool | None = Field(
        default=None,
        description="DEPRECATED: Use dedicated security health model",
    )

    ssl_certificates_expire_days: int | None = Field(
        default=None,
        ge=0,
        description="DEPRECATED: Use dedicated security health model",
    )

    recent_errors: list[str] | None = Field(
        default=None,
        max_items=10,
        description="DEPRECATED: Use service-specific health models for error tracking",
    )

    warning_messages: list[str] | None = Field(
        default=None,
        max_items=10,
        description="DEPRECATED: Use service-specific health models for warning tracking",
    )

    health_check_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="DEPRECATED: Use service-specific health models for check duration",
    )

    components_checked: int | None = Field(
        default=None,
        ge=0,
        description="DEPRECATED: Use service-specific health models for component tracking",
    )

    components_healthy: int | None = Field(
        default=None,
        ge=0,
        description="DEPRECATED: Use service-specific health models for component tracking",
    )

    def get_overall_health_status(self) -> "HealthStatus":
        """
        Get overall health status by aggregating service-specific health details.

        Returns the worst health status among all service health models.
        """
        statuses = []

        if self.postgres_health:
            statuses.append(self.postgres_health.get_health_status())

        if self.kafka_health:
            statuses.append(self.kafka_health.get_health_status())

        if self.circuit_breaker_health:
            statuses.append(self.circuit_breaker_health.get_health_status())

        if self.system_health:
            statuses.append(self.system_health.get_health_status())

        if not statuses:
            return EnumHealthStatus.UNKNOWN

        # Priority order: CRITICAL > UNHEALTHY > WARNING > DEGRADED > HEALTHY
        status_priority = {
            EnumHealthStatus.CRITICAL: 0,
            EnumHealthStatus.UNHEALTHY: 1,
            EnumHealthStatus.WARNING: 2,
            EnumHealthStatus.DEGRADED: 3,
            EnumHealthStatus.HEALTHY: 4,
            EnumHealthStatus.UNKNOWN: 5,
        }

        return min(statuses, key=lambda s: status_priority.get(s, 99))

    def get_health_summary(self) -> str:
        """Generate comprehensive health summary from all service health models."""
        summaries = []

        if self.postgres_health:
            summaries.append(f"PostgreSQL: {self.postgres_health.get_health_summary()}")

        if self.kafka_health:
            summaries.append(f"Kafka: {self.kafka_health.get_health_summary()}")

        if self.circuit_breaker_health:
            summaries.append(f"Circuit Breaker: {self.circuit_breaker_health.get_health_summary()}")

        if self.system_health:
            summaries.append(f"System: {self.system_health.get_health_summary()}")

        if not summaries:
            return "No service health data available"

        return " | ".join(summaries)
