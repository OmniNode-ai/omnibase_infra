# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Dispatch Metrics Model.

Aggregate metrics for the message dispatch engine including dispatch counts,
latency histograms, and per-dispatcher/per-category breakdowns.

Design Pattern:
    ModelDispatchMetrics is a comprehensive metrics container that aggregates:
    - Overall dispatch statistics (counts, latency)
    - Per-dispatcher metrics (via ModelDispatcherMetrics)
    - Per-category metrics (event, command, intent, projection counts)
    - Latency histogram buckets for distribution analysis

    Copy-on-Write Pattern:
        Unlike most ONEX models, this model is NOT frozen (no `frozen=True` in
        model_config) to allow Pydantic validation of nested dictionary updates.
        However, update methods like `record_dispatch()` return NEW instances
        rather than mutating in place, following the copy-on-write pattern:

        ```python
        # Copy-on-write: returns new instance, does not mutate
        metrics = metrics.record_dispatch(duration_ms=45.2, success=True)
        ```

        This provides the benefits of immutability (predictable state, thread-safe
        sharing of snapshots) while still allowing the model to be non-frozen for
        Pydantic's internal validation requirements.

Thread Safety:
    Individual ModelDispatchMetrics instances are safe to share across threads
    since update operations return new instances. The MessageDispatchEngine
    uses a lock (`_metrics_lock`) to ensure atomic read-modify-write cycles
    when updating the shared metrics reference.

Example:
    >>> from omnibase_infra.models.dispatch import ModelDispatchMetrics
    >>> from omnibase_infra.enums import EnumMessageCategory
    >>>
    >>> metrics = ModelDispatchMetrics()
    >>> metrics = metrics.record_dispatch(
    ...     duration_ms=45.2,
    ...     success=True,
    ...     category=EnumMessageCategory.EVENT,
    ...     dispatcher_id="user-dispatcher",
    ... )
    >>> print(f"Success rate: {metrics.success_rate:.1%}")

See Also:
    omnibase_infra.models.dispatch.ModelDispatcherMetrics: Per-dispatcher metrics
    omnibase_core.runtime.MessageDispatchEngine: Uses these for observability
"""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatcher_metrics import (
    ModelDispatcherMetrics,
)

# Histogram bucket boundaries in milliseconds
LATENCY_HISTOGRAM_BUCKETS: tuple[float, ...] = (
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
)


class ModelDispatchMetrics(BaseModel):
    """
    Aggregate metrics for the message dispatch engine.

    Provides comprehensive observability including:
    - Overall dispatch counts and success/error rates
    - Latency statistics (average, min, max)
    - Latency histogram for distribution analysis
    - Per-dispatcher metrics breakdown
    - Per-category metrics breakdown

    Memory Considerations:
        The dispatcher_metrics dictionary size is bounded by the number
        of registered dispatchers. Under the freeze-after-init pattern,
        dispatchers are registered during initialization and then frozen,
        ensuring predictable memory usage.

        Specifically:
        - dispatcher_metrics: Bounded by registered dispatcher count (fixed after freeze)
        - category_metrics: Fixed 4 keys (event, command, intent, projection)
        - latency_histogram: Fixed 13 buckets

        The freeze-after-init pattern guarantees that after engine.freeze()
        is called, no new dispatchers can be registered. This means the
        dispatcher_metrics dictionary has a known maximum size equal to
        the number of dispatchers registered during initialization.

        For production deployments, typical dispatcher counts are:
        - Small systems: 5-20 dispatchers
        - Medium systems: 20-100 dispatchers
        - Large systems: 100-500 dispatchers

        Each ModelDispatcherMetrics instance is approximately 200-500 bytes,
        so even large systems with 500 dispatchers use under 250KB for
        dispatcher metrics.

    Attributes:
        total_dispatches: Total number of dispatch operations.
        successful_dispatches: Number of successful dispatches.
        failed_dispatches: Number of failed dispatches.
        no_dispatcher_count: Dispatches with no matching dispatcher.
        category_mismatch_count: Dispatches with category validation failures.
        total_latency_ms: Cumulative latency across all dispatches.
        min_latency_ms: Minimum observed dispatch latency.
        max_latency_ms: Maximum observed dispatch latency.
        latency_histogram: Histogram buckets for latency distribution.
        dispatcher_metrics: Per-dispatcher metrics keyed by dispatcher_id.
        category_metrics: Per-category dispatch counts.

    Example:
        >>> metrics = ModelDispatchMetrics()
        >>> print(f"Total dispatches: {metrics.total_dispatches}")
        >>> print(f"Success rate: {metrics.success_rate:.1%}")
    """

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        validate_assignment=True,
    )

    # ---- Dispatch Counts ----
    total_dispatches: int = Field(
        default=0,
        description="Total number of dispatch operations.",
        ge=0,
    )
    successful_dispatches: int = Field(
        default=0,
        description="Number of successful dispatches.",
        ge=0,
    )
    failed_dispatches: int = Field(
        default=0,
        description="Number of failed dispatches.",
        ge=0,
    )
    no_dispatcher_count: int = Field(
        default=0,
        description="Dispatches with no matching dispatcher.",
        ge=0,
    )
    category_mismatch_count: int = Field(
        default=0,
        description="Dispatches with category validation failures.",
        ge=0,
    )

    # ---- Dispatcher Execution Counts ----
    dispatcher_execution_count: int = Field(
        default=0,
        description="Total number of dispatcher executions (may exceed dispatch count for fan-out).",
        ge=0,
    )
    dispatcher_error_count: int = Field(
        default=0,
        description="Total number of dispatcher execution failures.",
        ge=0,
    )
    routes_matched_count: int = Field(
        default=0,
        description="Total number of route matches.",
        ge=0,
    )

    # ---- Latency Statistics ----
    total_latency_ms: float = Field(
        default=0.0,
        description="Cumulative latency across all dispatches in milliseconds.",
        ge=0,
    )
    min_latency_ms: float | None = Field(
        default=None,
        description="Minimum observed dispatch latency in milliseconds.",
    )
    max_latency_ms: float | None = Field(
        default=None,
        description="Maximum observed dispatch latency in milliseconds.",
    )

    # ---- Latency Histogram ----
    latency_histogram: dict[str, int] = Field(
        default_factory=lambda: {
            "le_1ms": 0,
            "le_5ms": 0,
            "le_10ms": 0,
            "le_25ms": 0,
            "le_50ms": 0,
            "le_100ms": 0,
            "le_250ms": 0,
            "le_500ms": 0,
            "le_1000ms": 0,
            "le_2500ms": 0,
            "le_5000ms": 0,
            "le_10000ms": 0,
            "gt_10000ms": 0,
        },
        description="Histogram buckets for latency distribution.",
    )

    # ---- Per-Dispatcher Metrics ----
    dispatcher_metrics: dict[str, ModelDispatcherMetrics] = Field(
        default_factory=dict,
        description=(
            "Per-dispatcher metrics keyed by dispatcher_id. "
            "Memory is bounded by freeze-after-init pattern: only dispatchers "
            "registered before freeze() will have metrics entries, ensuring "
            "no unbounded growth after initialization. The dictionary size equals "
            "the number of registered dispatchers (typically 5-500 depending on "
            "system scale). For long-running services, consider periodically "
            "exporting metrics to an external system (Prometheus, StatsD) and "
            "using reset_metrics() or create_empty() to prevent counter overflow "
            "in extreme cases. Note: counter overflow is theoretical (requires "
            ">9 quintillion dispatches for int64), but periodic metrics export "
            "enables historical trend analysis and reduces memory pressure from "
            "accumulated per-dispatcher statistics."
        ),
    )

    # ---- Per-Category Metrics ----
    category_metrics: dict[str, int] = Field(
        default_factory=lambda: {
            "event": 0,
            "command": 0,
            "intent": 0,
            "projection": 0,
        },
        description="Per-category dispatch counts.",
    )

    @property
    def avg_latency_ms(self) -> float:
        """
        Calculate average latency across all dispatches.

        Returns:
            Average latency in milliseconds, or 0.0 if no dispatches.
        """
        if self.total_dispatches == 0:
            return 0.0
        return self.total_latency_ms / self.total_dispatches

    @property
    def success_rate(self) -> float:
        """
        Calculate success rate as a fraction (0.0 to 1.0).

        Returns:
            Success rate as a decimal, or 1.0 if no dispatches.
        """
        if self.total_dispatches == 0:
            return 1.0
        return self.successful_dispatches / self.total_dispatches

    @property
    def error_rate(self) -> float:
        """
        Calculate error rate as a fraction (0.0 to 1.0).

        Returns:
            Error rate as a decimal, or 0.0 if no dispatches.
        """
        if self.total_dispatches == 0:
            return 0.0
        return self.failed_dispatches / self.total_dispatches

    def _get_histogram_bucket(self, duration_ms: float) -> str:
        """Get the histogram bucket key for a given latency.

        Uses LATENCY_HISTOGRAM_BUCKETS to ensure consistency between
        bucket thresholds and bucket key generation.
        """
        # Iterate through bucket thresholds in order
        for threshold in LATENCY_HISTOGRAM_BUCKETS:
            if duration_ms <= threshold:
                return f"le_{int(threshold)}ms"
        # Exceeds all bucket thresholds
        return f"gt_{int(LATENCY_HISTOGRAM_BUCKETS[-1])}ms"

    def record_dispatch(
        self,
        duration_ms: float,
        success: bool,
        category: EnumMessageCategory | None = None,
        dispatcher_id: str | None = None,
        no_dispatcher: bool = False,
        category_mismatch: bool = False,
        handler_error: bool = False,
        routes_matched: int = 0,
        topic: str | None = None,
        error_message: str | None = None,
    ) -> "ModelDispatchMetrics":
        """
        Record a dispatch operation and return updated metrics.

        Creates a new ModelDispatchMetrics instance with updated statistics.

        Args:
            duration_ms: Dispatch duration in milliseconds.
            success: Whether the dispatch was successful.
            category: Optional message category for per-category metrics.
            dispatcher_id: Optional dispatcher ID for per-dispatcher metrics.
            no_dispatcher: Whether no dispatcher was found.
            category_mismatch: Whether category validation failed.
            handler_error: Whether a dispatcher execution error occurred.
            routes_matched: Number of routes that matched.
            topic: Optional topic for dispatcher metrics.
            error_message: Optional error message for dispatcher metrics.

        Returns:
            New ModelDispatchMetrics with updated statistics.
        """
        # Update latency statistics
        new_min = (
            duration_ms
            if self.min_latency_ms is None
            else min(self.min_latency_ms, duration_ms)
        )
        new_max = (
            duration_ms
            if self.max_latency_ms is None
            else max(self.max_latency_ms, duration_ms)
        )

        # Update histogram
        new_histogram = dict(self.latency_histogram)
        bucket = self._get_histogram_bucket(duration_ms)
        new_histogram[bucket] = new_histogram.get(bucket, 0) + 1

        # Update category metrics
        new_category_metrics = dict(self.category_metrics)
        if category is not None:
            category_key = category.value.lower()
            new_category_metrics[category_key] = (
                new_category_metrics.get(category_key, 0) + 1
            )

        # Update dispatcher metrics
        new_dispatcher_metrics = dict(self.dispatcher_metrics)
        if dispatcher_id is not None:
            existing = new_dispatcher_metrics.get(dispatcher_id)
            if existing is None:
                existing = ModelDispatcherMetrics(dispatcher_id=dispatcher_id)
            new_dispatcher_metrics[dispatcher_id] = existing.record_execution(
                duration_ms=duration_ms,
                success=success and not handler_error,
                topic=topic,
                error_message=error_message,
            )

        # Use model_copy to prevent field drift when new fields are added
        return self.model_copy(
            update={
                "total_dispatches": self.total_dispatches + 1,
                "successful_dispatches": self.successful_dispatches
                + (1 if success else 0),
                "failed_dispatches": self.failed_dispatches + (0 if success else 1),
                "no_dispatcher_count": self.no_dispatcher_count
                + (1 if no_dispatcher else 0),
                "category_mismatch_count": self.category_mismatch_count
                + (1 if category_mismatch else 0),
                "dispatcher_execution_count": self.dispatcher_execution_count
                + (1 if dispatcher_id else 0),
                "dispatcher_error_count": self.dispatcher_error_count
                + (1 if handler_error else 0),
                "routes_matched_count": self.routes_matched_count + routes_matched,
                "total_latency_ms": self.total_latency_ms + duration_ms,
                "min_latency_ms": new_min,
                "max_latency_ms": new_max,
                "latency_histogram": new_histogram,
                "dispatcher_metrics": new_dispatcher_metrics,
                "category_metrics": new_category_metrics,
            }
        )

    def get_dispatcher_metrics(
        self, dispatcher_id: str
    ) -> ModelDispatcherMetrics | None:
        """
        Get metrics for a specific dispatcher.

        Args:
            dispatcher_id: The dispatcher's unique identifier.

        Returns:
            ModelDispatcherMetrics for the dispatcher, or None if not found.
        """
        return self.dispatcher_metrics.get(dispatcher_id)

    def update_dispatcher_metrics(
        self,
        dispatcher_id: str,
        metrics: ModelDispatcherMetrics,
    ) -> "ModelDispatchMetrics":
        """
        Efficiently update a single dispatcher's metrics.

        Uses model_copy() for efficient copy-on-write update instead of
        reconstructing the entire object with all parameters. This is
        significantly more efficient when only updating dispatcher metrics
        without changing aggregate statistics.

        Args:
            dispatcher_id: The dispatcher ID to update metrics for.
            metrics: The new metrics for the dispatcher.

        Returns:
            New ModelDispatchMetrics with updated dispatcher metrics.

        Example:
            >>> old_metrics = ModelDispatchMetrics()
            >>> dispatcher_metrics = ModelDispatcherMetrics(
            ...     dispatcher_id="my-dispatcher"
            ... )
            >>> new_metrics = old_metrics.update_dispatcher_metrics(
            ...     "my-dispatcher",
            ...     dispatcher_metrics
            ... )
        """
        new_dict = dict(self.dispatcher_metrics)
        new_dict[dispatcher_id] = metrics
        return self.model_copy(update={"dispatcher_metrics": new_dict})

    def get_category_count(self, category: EnumMessageCategory) -> int:
        """
        Get dispatch count for a specific category.

        Args:
            category: The message category.

        Returns:
            Number of dispatches for this category.
        """
        category_key = category.value.lower()
        return self.category_metrics.get(category_key, 0)

    def to_dict(self) -> dict[str, object]:
        """
        Convert to dictionary with computed properties included.

        Returns:
            Dictionary with all metrics including computed properties.
        """
        return {
            "total_dispatches": self.total_dispatches,
            "successful_dispatches": self.successful_dispatches,
            "failed_dispatches": self.failed_dispatches,
            "no_dispatcher_count": self.no_dispatcher_count,
            "category_mismatch_count": self.category_mismatch_count,
            "dispatcher_execution_count": self.dispatcher_execution_count,
            "dispatcher_error_count": self.dispatcher_error_count,
            "routes_matched_count": self.routes_matched_count,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "total_latency_ms": self.total_latency_ms,
            "latency_histogram": self.latency_histogram,
            "category_metrics": self.category_metrics,
            "dispatcher_metrics": {
                dispatcher_id: metrics.to_dict()
                for dispatcher_id, metrics in self.dispatcher_metrics.items()
            },
        }

    @classmethod
    def create_empty(cls) -> "ModelDispatchMetrics":
        """
        Create a new empty metrics instance.

        Returns:
            New ModelDispatchMetrics with all counters at zero.
        """
        return cls()


__all__ = ["LATENCY_HISTOGRAM_BUCKETS", "ModelDispatchMetrics"]
