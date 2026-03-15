# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding resolution metrics model for observability.

Captures per-operation and aggregate metrics for binding resolution,
including resolution time histograms, failure counts by error code,
and per-dispatch binding statistics.

Design Pattern:
    ModelBindingResolutionMetrics follows the same copy-on-write pattern
    as ModelDispatchMetrics: update methods return NEW instances rather
    than mutating in place. This provides thread-safe snapshot sharing.

Thread Safety:
    Individual instances are safe to share across threads since update
    operations return new instances. The caller (e.g., OperationBindingResolver
    or MessageDispatchEngine) is responsible for atomic read-modify-write
    cycles when updating a shared metrics reference.

.. versionadded:: 0.2.8
    Created as part of OMN-1644 - Observability for operation bindings.

See Also:
    omnibase_infra.runtime.binding_resolver.OperationBindingResolver
    omnibase_infra.models.dispatch.model_dispatch_metrics.ModelDispatchMetrics
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Histogram bucket boundaries in milliseconds for binding resolution latency.
# Binding resolution is typically sub-millisecond, so buckets are finer-grained
# than dispatch latency histograms.
BINDING_LATENCY_HISTOGRAM_BUCKETS: tuple[float, ...] = (
    0.1,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
)


class ModelBindingResolutionMetrics(BaseModel):
    """Aggregate metrics for binding resolution observability.

    Tracks resolution counts, latency distribution, error breakdowns by
    error code, and per-operation statistics. Compatible with Prometheus
    and OpenTelemetry metric export conventions.

    Metric Mapping (Prometheus/OpenTelemetry):
        - ``total_resolutions`` -> Counter ``onex_binding_resolutions_total``
        - ``successful_resolutions`` -> Counter ``onex_binding_resolutions_total{status="success"}``
        - ``failed_resolutions`` -> Counter ``onex_binding_resolutions_total{status="error"}``
        - ``latency_histogram`` -> Histogram ``onex_binding_resolution_duration_ms``
        - ``error_counts_by_code`` -> Counter ``onex_binding_resolution_errors_total{code="..."}``
        - ``bindings_resolved_count`` -> Counter ``onex_bindings_resolved_total``
        - ``per_operation_resolutions`` -> Counter ``onex_binding_resolutions_total{operation="..."}``

    Attributes:
        total_resolutions: Total binding resolution attempts.
        successful_resolutions: Resolutions that completed without error.
        failed_resolutions: Resolutions that raised BindingResolutionError.
        bindings_resolved_count: Total individual bindings resolved across all operations.
        total_latency_ms: Cumulative resolution latency in milliseconds.
        min_latency_ms: Minimum observed resolution latency.
        max_latency_ms: Maximum observed resolution latency.
        latency_histogram: Histogram buckets for resolution latency distribution.
        error_counts_by_code: Failure counts keyed by BINDING_LOADER_0xx error code.
        per_operation_resolutions: Resolution counts keyed by operation name.
        per_operation_errors: Error counts keyed by operation name.
        missing_context_path_warnings: Count of missing additional_context_paths warnings.

    Example:
        >>> metrics = ModelBindingResolutionMetrics()
        >>> metrics = metrics.record_resolution(
        ...     duration_ms=0.45,
        ...     success=True,
        ...     operation="db.query",
        ...     bindings_resolved=3,
        ... )
        >>> print(f"Success rate: {metrics.success_rate:.1%}")
        Success rate: 100.0%

    .. versionadded:: 0.2.8
    """

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        validate_assignment=True,
    )

    # ---- Resolution Counts ----
    total_resolutions: int = Field(
        default=0,
        description="Total binding resolution attempts.",
        ge=0,
    )
    successful_resolutions: int = Field(
        default=0,
        description="Resolutions that completed without error.",
        ge=0,
    )
    failed_resolutions: int = Field(
        default=0,
        description="Resolutions that raised BindingResolutionError.",
        ge=0,
    )
    bindings_resolved_count: int = Field(
        default=0,
        description="Total individual bindings resolved across all operations.",
        ge=0,
    )

    # ---- Latency Statistics ----
    total_latency_ms: float = Field(
        default=0.0,
        description="Cumulative resolution latency in milliseconds.",
        ge=0,
    )
    min_latency_ms: float | None = Field(
        default=None,
        description="Minimum observed resolution latency in milliseconds.",
    )
    max_latency_ms: float | None = Field(
        default=None,
        description="Maximum observed resolution latency in milliseconds.",
    )

    # ---- Latency Histogram ----
    latency_histogram: dict[str, int] = Field(
        default_factory=lambda: {
            "le_0.1ms": 0,
            "le_0.5ms": 0,
            "le_1ms": 0,
            "le_2.5ms": 0,
            "le_5ms": 0,
            "le_10ms": 0,
            "le_25ms": 0,
            "le_50ms": 0,
            "le_100ms": 0,
            "gt_100ms": 0,
        },
        description="Histogram buckets for resolution latency distribution.",
    )

    # ---- Error Breakdown ----
    error_counts_by_code: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Failure counts keyed by error code (e.g., BINDING_LOADER_010). "
            "Only populated on resolution failures with typed error codes."
        ),
    )

    # ---- Per-Operation Metrics ----
    per_operation_resolutions: dict[str, int] = Field(
        default_factory=dict,
        description="Resolution counts keyed by operation name.",
    )
    per_operation_errors: dict[str, int] = Field(
        default_factory=dict,
        description="Error counts keyed by operation name.",
    )

    # ---- Context Path Warnings ----
    missing_context_path_warnings: int = Field(
        default=0,
        description=(
            "Count of warnings emitted when declared additional_context_paths "
            "are not provided in the dispatch context."
        ),
        ge=0,
    )

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average resolution latency.

        Returns:
            Average latency in milliseconds, or 0.0 if no resolutions.
        """
        if self.total_resolutions == 0:
            return 0.0
        return self.total_latency_ms / self.total_resolutions

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a fraction (0.0 to 1.0).

        Returns:
            Success rate, or 1.0 if no resolutions.
        """
        if self.total_resolutions == 0:
            return 1.0
        return self.successful_resolutions / self.total_resolutions

    @property
    def error_rate(self) -> float:
        """Calculate error rate as a fraction (0.0 to 1.0).

        Returns:
            Error rate, or 0.0 if no resolutions.
        """
        if self.total_resolutions == 0:
            return 0.0
        return self.failed_resolutions / self.total_resolutions

    def _get_histogram_bucket(self, duration_ms: float) -> str:
        """Get the histogram bucket key for a given latency."""
        for threshold in BINDING_LATENCY_HISTOGRAM_BUCKETS:
            if duration_ms <= threshold:
                if threshold < 1.0:
                    return f"le_{threshold}ms"
                return f"le_{int(threshold)}ms"
        return f"gt_{int(BINDING_LATENCY_HISTOGRAM_BUCKETS[-1])}ms"

    def record_resolution(
        self,
        duration_ms: float,
        success: bool,
        operation: str,
        bindings_resolved: int = 0,
    ) -> ModelBindingResolutionMetrics:
        """Record a binding resolution attempt and return updated metrics.

        Creates a new ModelBindingResolutionMetrics instance with updated
        statistics (copy-on-write pattern).

        Args:
            duration_ms: Resolution duration in milliseconds.
            success: Whether the resolution succeeded.
            operation: Operation name (e.g., "db.query").
            bindings_resolved: Number of individual bindings resolved.

        Returns:
            New ModelBindingResolutionMetrics with updated statistics.

        Example:
            >>> metrics = ModelBindingResolutionMetrics()
            >>> metrics = metrics.record_resolution(
            ...     duration_ms=0.3,
            ...     success=True,
            ...     operation="db.query",
            ...     bindings_resolved=3,
            ... )
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

        # Update per-operation metrics
        new_per_op = dict(self.per_operation_resolutions)
        new_per_op[operation] = new_per_op.get(operation, 0) + 1

        new_per_op_errors = dict(self.per_operation_errors)
        if not success:
            new_per_op_errors[operation] = new_per_op_errors.get(operation, 0) + 1

        return self.model_copy(
            update={
                "total_resolutions": self.total_resolutions + 1,
                "successful_resolutions": self.successful_resolutions
                + (1 if success else 0),
                "failed_resolutions": self.failed_resolutions + (0 if success else 1),
                "bindings_resolved_count": self.bindings_resolved_count
                + bindings_resolved,
                "total_latency_ms": self.total_latency_ms + duration_ms,
                "min_latency_ms": new_min,
                "max_latency_ms": new_max,
                "latency_histogram": new_histogram,
                "per_operation_resolutions": new_per_op,
                "per_operation_errors": new_per_op_errors,
            },
        )

    def record_error_code(self, error_code: str) -> ModelBindingResolutionMetrics:
        """Record a binding resolution error by error code.

        Use after ``record_resolution(success=False)`` to track the
        specific error code (e.g., ``BINDING_LOADER_010``).

        Args:
            error_code: Error code string (e.g., "BINDING_LOADER_010").

        Returns:
            New instance with incremented error code count.

        .. versionadded:: 0.2.8
        """
        new_error_counts = dict(self.error_counts_by_code)
        new_error_counts[error_code] = new_error_counts.get(error_code, 0) + 1
        return self.model_copy(
            update={"error_counts_by_code": new_error_counts},
        )

    def record_missing_context_path_warning(self) -> ModelBindingResolutionMetrics:
        """Record a missing additional_context_paths warning.

        Returns:
            New instance with incremented warning count.
        """
        return self.model_copy(
            update={
                "missing_context_path_warnings": self.missing_context_path_warnings + 1,
            },
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary with computed properties included.

        Returns:
            Dictionary suitable for JSON serialization or metrics export.
        """
        return {
            "total_resolutions": self.total_resolutions,
            "successful_resolutions": self.successful_resolutions,
            "failed_resolutions": self.failed_resolutions,
            "bindings_resolved_count": self.bindings_resolved_count,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "total_latency_ms": self.total_latency_ms,
            "latency_histogram": self.latency_histogram,
            "error_counts_by_code": self.error_counts_by_code,
            "per_operation_resolutions": self.per_operation_resolutions,
            "per_operation_errors": self.per_operation_errors,
            "missing_context_path_warnings": self.missing_context_path_warnings,
        }

    @classmethod
    def create_empty(cls) -> ModelBindingResolutionMetrics:
        """Create a new empty metrics instance.

        Returns:
            New ModelBindingResolutionMetrics with all counters at zero.
        """
        return cls()


__all__ = [
    "BINDING_LATENCY_HISTOGRAM_BUCKETS",
    "ModelBindingResolutionMetrics",
]
