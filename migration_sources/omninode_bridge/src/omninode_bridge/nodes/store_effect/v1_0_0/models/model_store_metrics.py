"""
Store Effect Node Metrics Model.

Tracks performance and operational metrics for the Store Effect Node.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelStoreEffectMetrics
- Strong typing with Pydantic models
- Comprehensive metrics tracking

Pure Reducer Refactor - Wave 4, Workstream 4A
Reference: docs/planning/PURE_REDUCER_REFACTOR_PLAN.md
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelStoreEffectMetrics(BaseModel):
    """
    Metrics for Store Effect Node operations.

    Tracks all persistence operations, conflicts, errors, and performance metrics.
    Used for monitoring, alerting, and optimization.

    Performance Targets:
    - state_commits_total: >95% success rate
    - state_conflicts_total: <5% of total operations
    - persist_errors_total: <1% of total operations
    - avg_persist_latency_ms: <10ms (p95)

    Attributes:
        state_commits_total: Total successful state commits
        state_conflicts_total: Total version conflicts (optimistic lock failures)
        fsm_transitions_total: Total FSM state transitions processed
        persist_errors_total: Total persistence errors encountered
        avg_persist_latency_ms: Average persistence latency in milliseconds
        events_published_total: Total events published to Kafka
        events_failed_total: Total failed event publications
        current_workflows: Number of active workflows being tracked
    """

    state_commits_total: int = Field(
        default=0,
        description="Total successful state commits",
        ge=0,
    )

    state_conflicts_total: int = Field(
        default=0,
        description="Total version conflicts (optimistic lock failures)",
        ge=0,
    )

    fsm_transitions_total: int = Field(
        default=0,
        description="Total FSM state transitions processed",
        ge=0,
    )

    persist_errors_total: int = Field(
        default=0,
        description="Total persistence errors encountered",
        ge=0,
    )

    avg_persist_latency_ms: float = Field(
        default=0.0,
        description="Average persistence latency in milliseconds",
        ge=0.0,
    )

    events_published_total: int = Field(
        default=0,
        description="Total events published to Kafka",
        ge=0,
    )

    events_failed_total: int = Field(
        default=0,
        description="Total failed event publications",
        ge=0,
    )

    current_workflows: int = Field(
        default=0,
        description="Number of active workflows being tracked",
        ge=0,
    )

    # Latency tracking internals (not exposed in JSON schema)
    _latency_samples: list[float] = []
    _max_samples: int = 1000  # Keep last 1000 samples for rolling average

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "state_commits_total": 1234,
                "state_conflicts_total": 56,
                "fsm_transitions_total": 789,
                "persist_errors_total": 12,
                "avg_persist_latency_ms": 8.5,
                "events_published_total": 2468,
                "events_failed_total": 3,
                "current_workflows": 45,
            }
        }
    )

    def record_commit_success(self, latency_ms: float) -> None:
        """Record successful state commit with latency."""
        self.state_commits_total += 1
        self._record_latency(latency_ms)

    def record_conflict(self, latency_ms: float) -> None:
        """Record version conflict with latency."""
        self.state_conflicts_total += 1
        self._record_latency(latency_ms)

    def record_fsm_transition(self) -> None:
        """Record FSM state transition."""
        self.fsm_transitions_total += 1

    def record_error(self) -> None:
        """Record persistence error."""
        self.persist_errors_total += 1

    def record_event_published(self) -> None:
        """Record successful event publication."""
        self.events_published_total += 1

    def record_event_failed(self) -> None:
        """Record failed event publication."""
        self.events_failed_total += 1

    def increment_workflow_count(self) -> None:
        """Increment active workflow count."""
        self.current_workflows += 1

    def decrement_workflow_count(self) -> None:
        """Decrement active workflow count."""
        if self.current_workflows > 0:
            self.current_workflows -= 1

    def _record_latency(self, latency_ms: float) -> None:
        """
        Record latency sample and update rolling average.

        Maintains a rolling window of the last N samples to compute average.
        This prevents memory growth while providing recent performance data.
        """
        self._latency_samples.append(latency_ms)

        # Keep only last N samples
        if len(self._latency_samples) > self._max_samples:
            self._latency_samples = self._latency_samples[-self._max_samples :]

        # Update average
        if self._latency_samples:
            self.avg_persist_latency_ms = sum(self._latency_samples) / len(
                self._latency_samples
            )

    def get_success_rate(self) -> float:
        """
        Calculate success rate for state commits.

        Returns:
            Success rate as percentage (0.0-100.0)
        """
        total_attempts = self.state_commits_total + self.state_conflicts_total
        if total_attempts == 0:
            return 0.0

        return (self.state_commits_total / total_attempts) * 100.0

    def get_conflict_rate(self) -> float:
        """
        Calculate conflict rate for state commits.

        Returns:
            Conflict rate as percentage (0.0-100.0)
        """
        total_attempts = self.state_commits_total + self.state_conflicts_total
        if total_attempts == 0:
            return 0.0

        return (self.state_conflicts_total / total_attempts) * 100.0

    def get_error_rate(self) -> float:
        """
        Calculate error rate for all operations.

        Returns:
            Error rate as percentage (0.0-100.0)
        """
        total_operations = (
            self.state_commits_total
            + self.state_conflicts_total
            + self.persist_errors_total
        )
        if total_operations == 0:
            return 0.0

        return (self.persist_errors_total / total_operations) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metrics to dictionary for Prometheus export.

        Returns:
            Dictionary with all metric values
        """
        return {
            "state_commits_total": self.state_commits_total,
            "state_conflicts_total": self.state_conflicts_total,
            "fsm_transitions_total": self.fsm_transitions_total,
            "persist_errors_total": self.persist_errors_total,
            "avg_persist_latency_ms": self.avg_persist_latency_ms,
            "events_published_total": self.events_published_total,
            "events_failed_total": self.events_failed_total,
            "current_workflows": self.current_workflows,
            "success_rate_pct": self.get_success_rate(),
            "conflict_rate_pct": self.get_conflict_rate(),
            "error_rate_pct": self.get_error_rate(),
        }
