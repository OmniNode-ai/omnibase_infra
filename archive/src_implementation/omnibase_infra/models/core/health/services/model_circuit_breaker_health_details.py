"""Circuit breaker health details model implementing ProtocolHealthDetails."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from omnibase_spi.protocols.types.core_types import HealthStatus

from omnibase_infra.enums import EnumCircuitBreakerState, EnumHealthStatus


class ModelCircuitBreakerHealthDetails(BaseModel):
    """Circuit breaker health details with self-assessment capability."""

    circuit_breaker_state: EnumCircuitBreakerState | None = Field(
        default=None,
        description="Current circuit breaker state",
    )

    circuit_breaker_failure_count: int | None = Field(
        default=None,
        ge=0,
        description="Circuit breaker failure count",
    )

    circuit_breaker_success_count: int | None = Field(
        default=None,
        ge=0,
        description="Circuit breaker success count",
    )

    failure_threshold: int | None = Field(
        default=None,
        ge=1,
        description="Failure threshold for opening circuit breaker",
    )

    timeout_duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Circuit breaker timeout duration in milliseconds",
    )

    last_failure_time: str | None = Field(
        default=None,
        description="ISO timestamp of last failure",
    )

    def get_health_status(self) -> "HealthStatus":
        """Assess circuit breaker health status based on state metrics."""
        if self.circuit_breaker_state == EnumCircuitBreakerState.OPEN:
            return EnumHealthStatus.CRITICAL

        if self.circuit_breaker_state == EnumCircuitBreakerState.HALF_OPEN:
            return EnumHealthStatus.WARNING

        # Check failure rate if we have the data
        if (self.circuit_breaker_failure_count is not None and
            self.circuit_breaker_success_count is not None and
            self.failure_threshold is not None):

            total_calls = self.circuit_breaker_failure_count + self.circuit_breaker_success_count
            if total_calls > 0:
                failure_rate = self.circuit_breaker_failure_count / total_calls
                threshold_rate = self.failure_threshold / max(total_calls, self.failure_threshold)

                if failure_rate >= threshold_rate * 0.8:  # 80% of threshold
                    return EnumHealthStatus.WARNING

        return EnumHealthStatus.HEALTHY

    def is_healthy(self) -> bool:
        """Return True if circuit breaker is considered healthy."""
        return self.get_health_status() == EnumHealthStatus.HEALTHY

    def get_health_summary(self) -> str:
        """Generate human-readable circuit breaker health summary."""
        status = self.get_health_status()

        if status == EnumHealthStatus.CRITICAL:
            return f"Circuit Breaker OPEN: {self.circuit_breaker_failure_count} failures"

        if status == EnumHealthStatus.WARNING:
            if self.circuit_breaker_state == EnumCircuitBreakerState.HALF_OPEN:
                return "Circuit Breaker HALF_OPEN: Testing recovery"
            return f"Circuit Breaker Warning: High failure rate ({self.circuit_breaker_failure_count} failures)"

        if self.circuit_breaker_state == EnumCircuitBreakerState.CLOSED:
            success_count = self.circuit_breaker_success_count or 0
            failure_count = self.circuit_breaker_failure_count or 0
            return f"Circuit Breaker CLOSED: {success_count} successes, {failure_count} failures"

        return "Circuit breaker healthy"
