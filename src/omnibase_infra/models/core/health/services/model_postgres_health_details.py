"""PostgreSQL health details model implementing ProtocolHealthDetails."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from omnibase_spi.protocols.types.core_types import HealthStatus

from omnibase_infra.enums import EnumHealthStatus


class ModelPostgresHealthDetails(BaseModel):
    """PostgreSQL-specific health details with self-assessment capability."""

    postgres_connection_count: int | None = Field(
        default=None,
        ge=0,
        description="Current PostgreSQL connection count",
    )

    max_connections: int | None = Field(
        default=None,
        ge=1,
        description="Maximum allowed PostgreSQL connections",
    )

    postgres_last_error: str | None = Field(
        default=None,
        max_length=500,
        description="Last PostgreSQL error message",
    )

    connection_pool_size: int | None = Field(
        default=None,
        ge=0,
        description="Current connection pool size",
    )

    active_queries: int | None = Field(
        default=None,
        ge=0,
        description="Number of currently active queries",
    )

    average_query_time_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Average query execution time in milliseconds",
    )

    def get_health_status(self) -> "HealthStatus":
        """Assess PostgreSQL health status based on service metrics."""
        # Critical failures
        if self.postgres_last_error:
            return EnumHealthStatus.UNHEALTHY

        # Warning conditions
        if self.postgres_connection_count and self.max_connections:
            connection_usage = self.postgres_connection_count / self.max_connections
            if connection_usage > 0.9:
                return EnumHealthStatus.WARNING
            if connection_usage > 0.95:
                return EnumHealthStatus.CRITICAL

        # Performance degradation
        if self.average_query_time_ms and self.average_query_time_ms > 5000:  # 5+ seconds
            return EnumHealthStatus.DEGRADED

        return EnumHealthStatus.HEALTHY

    def is_healthy(self) -> bool:
        """Return True if PostgreSQL is considered healthy."""
        return self.get_health_status() == EnumHealthStatus.HEALTHY

    def get_health_summary(self) -> str:
        """Generate human-readable PostgreSQL health summary."""
        status = self.get_health_status()

        if status == EnumHealthStatus.UNHEALTHY:
            return f"PostgreSQL Error: {self.postgres_last_error}"

        if status == EnumHealthStatus.CRITICAL:
            return f"PostgreSQL Critical: {self.postgres_connection_count}/{self.max_connections} connections (>95%)"

        if status == EnumHealthStatus.WARNING:
            return f"PostgreSQL Warning: {self.postgres_connection_count}/{self.max_connections} connections (>90%)"

        if status == EnumHealthStatus.DEGRADED:
            return f"PostgreSQL Degraded: Average query time {self.average_query_time_ms}ms"

        if self.postgres_connection_count and self.max_connections:
            return f"PostgreSQL Healthy: {self.postgres_connection_count}/{self.max_connections} connections"

        return "PostgreSQL connections healthy"
