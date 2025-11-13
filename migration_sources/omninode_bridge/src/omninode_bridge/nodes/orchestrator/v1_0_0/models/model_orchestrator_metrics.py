"""
Orchestrator Performance Metrics Model.

Tracks orchestrator performance metrics for event-driven coordination
and workflow orchestration.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelOrchestratorMetrics
- Strong typing with Pydantic v2
- Comprehensive performance tracking

Wave 5 Refactor - Event-Driven Orchestration
"""

from typing import ClassVar

from pydantic import BaseModel, Field


class ModelOrchestratorMetrics(BaseModel):
    """
    Performance metrics for orchestrator node.

    Tracks latency, success/failure rates, timeouts, and retries
    for both event-driven and legacy execution modes.

    Attributes:
        orchestration_latency_ms: Average orchestration latency in milliseconds
        orchestration_success_total: Total number of successful orchestrations
        orchestration_failure_total: Total number of failed orchestrations
        orchestration_timeout_total: Total number of timeout errors
        orchestration_retry_total: Total number of retry attempts
        event_wait_time_ms: Average time waiting for reducer events
        event_driven_executions: Count of event-driven workflow executions
        legacy_executions: Count of legacy synchronous executions

    Example:
        >>> metrics = ModelOrchestratorMetrics(
        ...     orchestration_latency_ms=125.5,
        ...     orchestration_success_total=100,
        ...     orchestration_failure_total=5,
        ...     orchestration_timeout_total=2,
        ...     orchestration_retry_total=8,
        ...     event_wait_time_ms=45.3,
        ...     event_driven_executions=95,
        ...     legacy_executions=10,
        ... )
        >>> assert metrics.success_rate == 0.95  # 100 / (100 + 5)
    """

    orchestration_latency_ms: float = Field(
        default=0.0,
        description="Average orchestration latency in milliseconds",
        ge=0.0,
    )
    orchestration_success_total: int = Field(
        default=0,
        description="Total number of successful orchestrations",
        ge=0,
    )
    orchestration_failure_total: int = Field(
        default=0,
        description="Total number of failed orchestrations",
        ge=0,
    )
    orchestration_timeout_total: int = Field(
        default=0,
        description="Total number of timeout errors",
        ge=0,
    )
    orchestration_retry_total: int = Field(
        default=0,
        description="Total number of retry attempts (DAG policy)",
        ge=0,
    )
    event_wait_time_ms: float = Field(
        default=0.0,
        description="Average time waiting for reducer events in milliseconds",
        ge=0.0,
    )
    event_driven_executions: int = Field(
        default=0,
        description="Count of event-driven workflow executions",
        ge=0,
    )
    legacy_executions: int = Field(
        default=0,
        description="Count of legacy synchronous workflow executions",
        ge=0,
    )

    @property
    def total_executions(self) -> int:
        """Calculate total number of executions."""
        return self.orchestration_success_total + self.orchestration_failure_total

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        total = self.total_executions
        if total == 0:
            return 0.0
        return self.orchestration_success_total / total

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate (0.0 to 1.0)."""
        total = self.total_executions
        if total == 0:
            return 0.0
        return self.orchestration_failure_total / total

    @property
    def timeout_rate(self) -> float:
        """Calculate timeout rate (0.0 to 1.0)."""
        total = self.total_executions
        if total == 0:
            return 0.0
        return self.orchestration_timeout_total / total

    @property
    def retry_rate(self) -> float:
        """Calculate retry rate (retries per failure)."""
        if self.orchestration_failure_total == 0:
            return 0.0
        return self.orchestration_retry_total / self.orchestration_failure_total

    @property
    def event_driven_percentage(self) -> float:
        """Calculate percentage of event-driven executions (0.0 to 1.0)."""
        total = self.event_driven_executions + self.legacy_executions
        if total == 0:
            return 0.0
        return self.event_driven_executions / total

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict] = {
            "example": {
                "orchestration_latency_ms": 125.5,
                "orchestration_success_total": 100,
                "orchestration_failure_total": 5,
                "orchestration_timeout_total": 2,
                "orchestration_retry_total": 8,
                "event_wait_time_ms": 45.3,
                "event_driven_executions": 95,
                "legacy_executions": 10,
            }
        }
