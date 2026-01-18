# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Prometheus metrics sink implementation for ONEX observability.

This module provides a thread-safe Prometheus metrics sink that implements
the ProtocolHotPathMetricsSink protocol. It enforces cardinality policies
to prevent high-cardinality label explosions that can overwhelm metrics backends.

Key Features:
    - Thread-safe metric registry with lazy metric creation
    - Cardinality enforcement via ModelMetricsPolicy
    - Forbidden high-cardinality labels blocked (envelope_id, correlation_id, etc.)
    - Label value length enforcement
    - Configurable violation handling (raise, warn_and_drop, drop_silent, warn_and_strip)

Priority Metrics Support:
    - Circuit breaker: state gauge, trip counter, rejection counter
    - Handler execution: latency histogram, success/failure counters
    - Retry: attempts counter, final failure counter

Thread-Safety:
    This implementation is fully thread-safe. Metric objects are created lazily
    and cached using a threading.Lock to prevent duplicate metric registration
    errors that would occur with concurrent first-access to the same metric.

Usage Example:
    >>> from omnibase_core.models.observability import ModelMetricsPolicy
    >>> from omnibase_infra.observability.sinks import SinkMetricsPrometheus
    >>>
    >>> # Create with default policy (warns and drops on violation)
    >>> sink = SinkMetricsPrometheus()
    >>>
    >>> # Or with custom policy
    >>> policy = ModelMetricsPolicy(
    ...     on_violation=EnumMetricsPolicyViolationAction.RAISE,
    ... )
    >>> sink = SinkMetricsPrometheus(policy=policy)
    >>>
    >>> # Use in hot path code
    >>> sink.increment_counter(
    ...     "http_requests_total",
    ...     {"method": "POST", "status": "200", "handler": "create_user"},
    ... )

See Also:
    - ProtocolHotPathMetricsSink: Protocol definition in omnibase_spi
    - ModelMetricsPolicy: Cardinality policy model in omnibase_core
    - docs/patterns/circuit_breaker_implementation.md: Circuit breaker metrics
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, cast

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

# Dependency validation for prometheus_client with clear error message
_PROMETHEUS_CLIENT_AVAILABLE: bool = False
_PROMETHEUS_IMPORT_ERROR: str | None = None

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROMETHEUS_CLIENT_AVAILABLE = True
except ImportError as e:
    _PROMETHEUS_IMPORT_ERROR = str(e)
    # Provide stubs for type checking when prometheus_client is not installed
    Counter = None  # type: ignore[assignment, misc]
    Gauge = None  # type: ignore[assignment, misc]
    Histogram = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from omnibase_core.models.observability import ModelMetricsPolicy

_logger = logging.getLogger(__name__)

# Import constants from dedicated constants module (ONEX pattern: constants in constants_*.py)
# Note: DEFAULT_HISTOGRAM_BUCKETS is also re-exported from this module for backwards compatibility
from omnibase_infra.observability.constants_metrics import (
    DEFAULT_HISTOGRAM_BUCKETS,
)


class SinkMetricsPrometheus:
    """Thread-safe Prometheus metrics sink with cardinality policy enforcement.

    This class implements ProtocolHotPathMetricsSink for Prometheus metric collection
    while enforcing cardinality policies to prevent metric explosion. All metrics
    are created lazily on first access and cached for subsequent calls.

    Cardinality Enforcement:
        The sink validates all labels against the configured ModelMetricsPolicy before
        recording any metric. By default, the following high-cardinality labels are
        forbidden:
            - envelope_id: Unique per-message identifier
            - correlation_id: Request correlation identifier
            - node_id: Node instance identifier
            - runtime_id: Runtime instance identifier

        Attempts to use these labels will be handled according to the policy's
        on_violation setting (raise, warn_and_drop, drop_silent, warn_and_strip).

    Thread-Safety:
        All metric operations are thread-safe. The internal metric registry uses
        a threading.Lock to ensure that metric objects are created exactly once
        even under concurrent access. Individual metric operations (increment,
        set, observe) are inherently thread-safe in prometheus_client.

    Metric Naming:
        Metric names should follow Prometheus naming conventions:
            - Use lowercase with underscores (snake_case)
            - Include units in the name (e.g., _seconds, _bytes, _total)
            - Use _total suffix for counters
            - Be descriptive but not too long

    Attributes:
        _policy: The cardinality policy governing label validation.
        _counters: Cache of Counter metric objects by name.
        _gauges: Cache of Gauge metric objects by name.
        _histograms: Cache of Histogram metric objects by name.
        _lock: Threading lock for thread-safe metric creation.

    Example:
        >>> sink = SinkMetricsPrometheus()
        >>>
        >>> # Record handler latency
        >>> sink.observe_histogram(
        ...     "handler_execution_seconds",
        ...     {"handler_type": "http_rest", "operation": "create_user"},
        ...     value=0.042,
        ... )
        >>>
        >>> # Track circuit breaker state
        >>> sink.set_gauge(
        ...     "circuit_breaker_state",
        ...     {"service": "kafka", "transport": "event_bus"},
        ...     value=0.0,  # 0=closed, 1=half_open, 2=open
        ... )
        >>>
        >>> # Count retries
        >>> sink.increment_counter(
        ...     "retry_attempts_total",
        ...     {"service": "postgres", "operation": "execute_query"},
        ... )
    """

    def __init__(
        self,
        policy: ModelMetricsPolicy | None = None,
        histogram_buckets: tuple[float, ...] | None = None,
        metric_prefix: str = "",
    ) -> None:
        """Initialize the Prometheus metrics sink.

        Args:
            policy: Cardinality policy for label validation. If None, a default
                policy is created that forbids high-cardinality labels and warns
                on violations while dropping the offending metric.
            histogram_buckets: Custom histogram bucket boundaries. If None,
                DEFAULT_HISTOGRAM_BUCKETS are used (Prometheus conventions).
            metric_prefix: Optional prefix to add to all metric names. Useful for
                namespacing metrics by service or component. The prefix will be
                joined with the metric name using an underscore.

        Raises:
            ProtocolConfigurationError: If prometheus_client is not installed.
                Install with: pip install prometheus-client

        Example:
            >>> # Default policy - warns and drops on violation
            >>> sink = SinkMetricsPrometheus()
            >>>
            >>> # Strict policy - raises on any violation
            >>> from omnibase_core.models.observability import ModelMetricsPolicy
            >>> from omnibase_core.enums import EnumMetricsPolicyViolationAction
            >>> strict_policy = ModelMetricsPolicy(
            ...     on_violation=EnumMetricsPolicyViolationAction.RAISE,
            ... )
            >>> sink = SinkMetricsPrometheus(policy=strict_policy)
            >>>
            >>> # Namespaced metrics
            >>> sink = SinkMetricsPrometheus(metric_prefix="onex_infra")
        """
        # Validate dependency is available with clear error message
        if not _PROMETHEUS_CLIENT_AVAILABLE:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="initialize_prometheus_sink",
            )
            raise ProtocolConfigurationError(
                "prometheus_client is required for SinkMetricsPrometheus but is not installed. "
                f"Install with: pip install prometheus-client. "
                f"Original error: {_PROMETHEUS_IMPORT_ERROR}",
                context=context,
            )

        # Import here to avoid circular imports and allow TYPE_CHECKING
        from omnibase_core.models.observability import ModelMetricsPolicy as _Policy

        self._policy: ModelMetricsPolicy = policy if policy is not None else _Policy()
        self._histogram_buckets = (
            histogram_buckets if histogram_buckets else DEFAULT_HISTOGRAM_BUCKETS
        )
        self._metric_prefix = metric_prefix

        # Thread-safe metric caches
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}

        # Lock for thread-safe metric creation
        self._lock = threading.Lock()

        _logger.debug(
            "SinkMetricsPrometheus initialized",
            extra={
                "policy_on_violation": self._policy.on_violation.value,
                "forbidden_keys": list(self._policy.forbidden_label_keys),
                "max_label_value_length": self._policy.max_label_value_length,
                "metric_prefix": self._metric_prefix,
            },
        )

    def _get_prefixed_name(self, name: str) -> str:
        """Return metric name with optional prefix.

        Args:
            name: Base metric name.

        Returns:
            Prefixed metric name if prefix is set, otherwise original name.
        """
        if self._metric_prefix:
            return f"{self._metric_prefix}_{name}"
        return name

    def _enforce_labels(self, labels: dict[str, str]) -> dict[str, str] | None:
        """Validate and enforce label policy.

        Delegates to ModelMetricsPolicy.enforce_labels() which handles all
        violation actions (raise, warn_and_drop, drop_silent, warn_and_strip).

        Args:
            labels: Label key-value pairs to validate.

        Returns:
            Validated/sanitized labels if allowed, None if metric should be dropped.

        Raises:
            ModelOnexError: If policy on_violation is RAISE and violations found.
        """
        # enforce_labels returns dict[str, str] | None directly
        # - Returns labels (possibly stripped) if allowed
        # - Returns None if metric should be dropped
        # - Raises ModelOnexError if on_violation=RAISE
        # Cast needed because omnibase_core lacks type stubs
        return cast("dict[str, str] | None", self._policy.enforce_labels(labels))

    def _get_or_create_counter(
        self,
        name: str,
        label_names: tuple[str, ...],
        documentation: str = "",
    ) -> Counter:
        """Get or create a Counter metric.

        Thread-safe lazy creation of Counter metrics. The counter is created
        on first access and cached for subsequent calls.

        Thread-Safety:
            Uses double-check locking pattern for optimal performance:
            1. Fast path: Check cache without lock (common case)
            2. Slow path: Acquire lock and verify before creation
            This avoids lock contention for the common cache-hit case while
            ensuring exactly-once initialization under concurrent access.

        Label Drift Protection:
            If a metric already exists with different label names, a warning
            is logged and the existing metric is returned. This prevents
            Prometheus errors from label set mismatches.

        Args:
            name: Metric name (without prefix).
            label_names: Tuple of label key names for this metric.
            documentation: Human-readable metric description.

        Returns:
            Counter metric object.
        """
        prefixed_name = self._get_prefixed_name(name)

        # Fast path: check cache without lock (common case)
        if prefixed_name in self._counters:
            existing = self._counters[prefixed_name]
            # Validate label names match (guard against label drift)
            existing_labels = tuple(existing._labelnames)
            if existing_labels != label_names:
                _logger.warning(
                    "Label set drift detected for counter metric",
                    extra={
                        "metric_name": prefixed_name,
                        "existing_labels": existing_labels,
                        "requested_labels": label_names,
                    },
                )
            return existing

        # Slow path: acquire lock and double-check before creation
        with self._lock:
            # Double-check after acquiring lock
            if prefixed_name in self._counters:
                existing = self._counters[prefixed_name]
                existing_labels = tuple(existing._labelnames)
                if existing_labels != label_names:
                    _logger.warning(
                        "Label set drift detected for counter metric",
                        extra={
                            "metric_name": prefixed_name,
                            "existing_labels": existing_labels,
                            "requested_labels": label_names,
                        },
                    )
                return existing

            self._counters[prefixed_name] = Counter(
                prefixed_name,
                documentation or f"Counter metric: {name}",
                labelnames=label_names,
            )
            _logger.debug(
                "Created Prometheus Counter",
                extra={"metric_name": prefixed_name, "labels": label_names},
            )
            return self._counters[prefixed_name]

    def _get_or_create_gauge(
        self,
        name: str,
        label_names: tuple[str, ...],
        documentation: str = "",
    ) -> Gauge:
        """Get or create a Gauge metric.

        Thread-safe lazy creation of Gauge metrics. The gauge is created
        on first access and cached for subsequent calls.

        Thread-Safety:
            Uses double-check locking pattern for optimal performance:
            1. Fast path: Check cache without lock (common case)
            2. Slow path: Acquire lock and verify before creation
            This avoids lock contention for the common cache-hit case while
            ensuring exactly-once initialization under concurrent access.

        Label Drift Protection:
            If a metric already exists with different label names, a warning
            is logged and the existing metric is returned. This prevents
            Prometheus errors from label set mismatches.

        Args:
            name: Metric name (without prefix).
            label_names: Tuple of label key names for this metric.
            documentation: Human-readable metric description.

        Returns:
            Gauge metric object.
        """
        prefixed_name = self._get_prefixed_name(name)

        # Fast path: check cache without lock (common case)
        if prefixed_name in self._gauges:
            existing = self._gauges[prefixed_name]
            # Validate label names match (guard against label drift)
            existing_labels = tuple(existing._labelnames)
            if existing_labels != label_names:
                _logger.warning(
                    "Label set drift detected for gauge metric",
                    extra={
                        "metric_name": prefixed_name,
                        "existing_labels": existing_labels,
                        "requested_labels": label_names,
                    },
                )
            return existing

        # Slow path: acquire lock and double-check before creation
        with self._lock:
            # Double-check after acquiring lock
            if prefixed_name in self._gauges:
                existing = self._gauges[prefixed_name]
                existing_labels = tuple(existing._labelnames)
                if existing_labels != label_names:
                    _logger.warning(
                        "Label set drift detected for gauge metric",
                        extra={
                            "metric_name": prefixed_name,
                            "existing_labels": existing_labels,
                            "requested_labels": label_names,
                        },
                    )
                return existing

            self._gauges[prefixed_name] = Gauge(
                prefixed_name,
                documentation or f"Gauge metric: {name}",
                labelnames=label_names,
            )
            _logger.debug(
                "Created Prometheus Gauge",
                extra={"metric_name": prefixed_name, "labels": label_names},
            )
            return self._gauges[prefixed_name]

    def _get_or_create_histogram(
        self,
        name: str,
        label_names: tuple[str, ...],
        documentation: str = "",
    ) -> Histogram:
        """Get or create a Histogram metric.

        Thread-safe lazy creation of Histogram metrics. The histogram is created
        on first access and cached for subsequent calls.

        Thread-Safety:
            Uses double-check locking pattern for optimal performance:
            1. Fast path: Check cache without lock (common case)
            2. Slow path: Acquire lock and verify before creation
            This avoids lock contention for the common cache-hit case while
            ensuring exactly-once initialization under concurrent access.

        Label Drift Protection:
            If a metric already exists with different label names, a warning
            is logged and the existing metric is returned. This prevents
            Prometheus errors from label set mismatches.

        Args:
            name: Metric name (without prefix).
            label_names: Tuple of label key names for this metric.
            documentation: Human-readable metric description.

        Returns:
            Histogram metric object.
        """
        prefixed_name = self._get_prefixed_name(name)

        # Fast path: check cache without lock (common case)
        if prefixed_name in self._histograms:
            existing = self._histograms[prefixed_name]
            # Validate label names match (guard against label drift)
            existing_labels = tuple(existing._labelnames)
            if existing_labels != label_names:
                _logger.warning(
                    "Label set drift detected for histogram metric",
                    extra={
                        "metric_name": prefixed_name,
                        "existing_labels": existing_labels,
                        "requested_labels": label_names,
                    },
                )
            return existing

        # Slow path: acquire lock and double-check before creation
        with self._lock:
            # Double-check after acquiring lock
            if prefixed_name in self._histograms:
                existing = self._histograms[prefixed_name]
                existing_labels = tuple(existing._labelnames)
                if existing_labels != label_names:
                    _logger.warning(
                        "Label set drift detected for histogram metric",
                        extra={
                            "metric_name": prefixed_name,
                            "existing_labels": existing_labels,
                            "requested_labels": label_names,
                        },
                    )
                return existing

            self._histograms[prefixed_name] = Histogram(
                prefixed_name,
                documentation or f"Histogram metric: {name}",
                labelnames=label_names,
                buckets=self._histogram_buckets,
            )
            _logger.debug(
                "Created Prometheus Histogram",
                extra={
                    "metric_name": prefixed_name,
                    "labels": label_names,
                    "buckets": self._histogram_buckets,
                },
            )
            return self._histograms[prefixed_name]

    def increment_counter(
        self,
        name: str,
        labels: dict[str, str],
        increment: int = 1,
    ) -> None:
        """Increment a counter metric by the specified amount.

        Counters are monotonically increasing values that reset only on process
        restart. Use counters for events, requests processed, errors, etc.

        Early Return Optimization:
            Non-positive increments (<=0) result in an immediate return without
            label validation. This avoids unnecessary policy enforcement overhead
            for no-op operations.

        Cardinality Enforcement:
            Labels are validated against the policy before recording. If any
            forbidden labels (envelope_id, correlation_id, node_id, runtime_id)
            are present or values exceed max length, the behavior depends on
            the policy's on_violation setting.

        Args:
            name: Metric name following Prometheus naming conventions.
                Should be lowercase with underscores (e.g., "http_requests_total").
            labels: Label key-value pairs. Keys and values must be strings.
                Label cardinality should be bounded (avoid high-cardinality
                values like user IDs or request IDs).
            increment: Amount to add to the counter. Defaults to 1.
                Must be positive; negative values are silently ignored.

        Returns:
            None. This method has no return value.

        Example:
            >>> sink.increment_counter(
            ...     "http_requests_total",
            ...     {"method": "POST", "status": "200", "handler": "create_user"},
            ... )
            >>>
            >>> # Increment by more than 1
            >>> sink.increment_counter(
            ...     "bytes_processed_total",
            ...     {"stream": "events"},
            ...     increment=len(payload),
            ... )
        """
        # Early return for non-positive increments (skip label validation for no-op)
        if increment < 1:
            _logger.debug(
                "Ignoring non-positive counter increment",
                extra={"metric_name": name, "increment": increment},
            )
            return

        # Validate labels against policy (only for positive increments)
        validated_labels = self._enforce_labels(labels)
        if validated_labels is None:
            # Policy dropped the metric
            return

        # Get label names as sorted tuple for consistent ordering
        label_names = tuple(sorted(validated_labels.keys()))

        # Get or create counter
        counter = self._get_or_create_counter(name, label_names)

        # Increment with labels in sorted order
        label_values = {k: validated_labels[k] for k in label_names}
        counter.labels(**label_values).inc(increment)

    def set_gauge(
        self,
        name: str,
        labels: dict[str, str],
        value: float,
    ) -> None:
        """Set a gauge metric to the specified value.

        Gauges represent point-in-time values that can increase or decrease.
        Use gauges for queue depths, active connections, memory usage, etc.

        Cardinality Enforcement:
            Labels are validated against the policy before recording. If any
            forbidden labels are present or values exceed max length, the
            behavior depends on the policy's on_violation setting.

        Args:
            name: Metric name following Prometheus naming conventions.
                Should be lowercase with underscores (e.g., "queue_depth").
            labels: Label key-value pairs. Keys and values must be strings.
            value: Current value of the gauge. Can be any float including
                negative values, zero, infinity, or NaN.

        Returns:
            None. This method has no return value.

        Example:
            >>> sink.set_gauge(
            ...     "active_connections",
            ...     {"pool": "database", "region": "us-west"},
            ...     value=42.0,
            ... )
            >>>
            >>> # Circuit breaker state (0=closed, 1=half_open, 2=open)
            >>> sink.set_gauge(
            ...     "circuit_breaker_state",
            ...     {"service": "kafka", "transport": "event_bus"},
            ...     value=0.0,
            ... )
        """
        # Validate labels against policy
        validated_labels = self._enforce_labels(labels)
        if validated_labels is None:
            # Policy dropped the metric
            return

        # Get label names as sorted tuple for consistent ordering
        label_names = tuple(sorted(validated_labels.keys()))

        # Get or create gauge
        gauge = self._get_or_create_gauge(name, label_names)

        # Set with labels in sorted order
        label_values = {k: validated_labels[k] for k in label_names}
        gauge.labels(**label_values).set(value)

    def observe_histogram(
        self,
        name: str,
        labels: dict[str, str],
        value: float,
    ) -> None:
        """Record an observation in a histogram metric.

        Histograms track the distribution of values, typically latencies or
        sizes. Values are bucketed for efficient storage and aggregation.

        Cardinality Enforcement:
            Labels are validated against the policy before recording. If any
            forbidden labels are present or values exceed max length, the
            behavior depends on the policy's on_violation setting.

        Args:
            name: Metric name following Prometheus naming conventions.
                Should be lowercase with underscores and include units
                (e.g., "request_duration_seconds", "response_size_bytes").
            labels: Label key-value pairs. Keys and values must be strings.
            value: Observed value to record. Should be non-negative for
                most use cases (durations, sizes).

        Returns:
            None. This method has no return value.

        Example:
            >>> import time
            >>>
            >>> start = time.perf_counter()
            >>> result = process_request(request)
            >>> duration = time.perf_counter() - start
            >>>
            >>> sink.observe_histogram(
            ...     "request_duration_seconds",
            ...     {"handler": "process_request", "status": "success"},
            ...     value=duration,
            ... )
        """
        # Validate labels against policy
        validated_labels = self._enforce_labels(labels)
        if validated_labels is None:
            # Policy dropped the metric
            return

        # Get label names as sorted tuple for consistent ordering
        label_names = tuple(sorted(validated_labels.keys()))

        # Get or create histogram
        histogram = self._get_or_create_histogram(name, label_names)

        # Observe with labels in sorted order
        label_values = {k: validated_labels[k] for k in label_names}
        histogram.labels(**label_values).observe(value)

    def get_policy(self) -> ModelMetricsPolicy:
        """Retrieve the metrics policy governing this sink.

        The policy defines constraints and configuration for metrics collection,
        including forbidden label keys, maximum label value length, and
        violation handling behavior.

        Returns:
            ModelMetricsPolicy: The active metrics policy. This is immutable
                (frozen=True in Pydantic config) and remains consistent
                throughout the sink's lifetime.

        Thread-Safety:
            This method is thread-safe. The policy is immutable after creation.

        Example:
            >>> policy = sink.get_policy()
            >>>
            >>> # Check forbidden labels
            >>> if "correlation_id" in policy.forbidden_label_keys:
            ...     logger.debug("correlation_id is forbidden as a metric label")
            >>>
            >>> # Check violation behavior
            >>> if policy.on_violation == EnumMetricsPolicyViolationAction.RAISE:
            ...     logger.warning("Strict policy - violations will raise errors")
        """
        return self._policy


__all__ = ["SinkMetricsPrometheus", "DEFAULT_HISTOGRAM_BUCKETS"]
