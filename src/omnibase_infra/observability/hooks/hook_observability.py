# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pipeline hook for cross-cutting observability concerns.

This module provides the HookObservability class, a pipeline hook that enables
cross-cutting observability instrumentation for infrastructure components.
The hook tracks operation timing, emits metrics, and maintains execution context
across async boundaries.

CRITICAL: Concurrency Safety via contextvars
--------------------------------------------
This implementation uses contextvars exclusively for all timing and operation
state. This is a CRITICAL design decision to prevent concurrency bugs in async
code. Each async task gets its own isolated context, preventing race conditions
where multiple concurrent operations would corrupt shared timing state.

Why NOT use instance variables:
    # WRONG - Race condition in async code!
    class BadHook:
        def __init__(self):
            self._start_time = 0.0  # Shared across all concurrent operations!

        def before_operation(self, operation: str):
            self._start_time = time.perf_counter()  # Overwrites previous!

        def after_operation(self):
            return time.perf_counter() - self._start_time  # Wrong value!

Why contextvars ARE correct:
    # CORRECT - Each async task has isolated state
    _start_time: ContextVar[float | None] = ContextVar("start_time", default=None)

    class GoodHook:
        def before_operation(self, operation: str):
            _start_time.set(time.perf_counter())  # Isolated per-task

        def after_operation(self):
            return time.perf_counter() - _start_time.get()  # Correct per-task

Thread-Safety Guarantees
------------------------
This class is thread-safe with the following guarantees:

1. **Timing State**: All timing and operation state is stored in contextvars,
   which provide per-task isolation in async code and per-thread isolation in
   threaded code. Concurrent operations NEVER share timing state.

2. **Metrics Sink**: The metrics_sink is accessed via a read-only property
   from multiple async tasks. The metrics_sink implementation MUST be
   thread-safe. The built-in SinkMetricsPrometheus satisfies this requirement
   via internal locking. If using a custom sink, ensure it is thread-safe.
   The metrics_sink is set once during __init__ and exposed as read-only to
   prevent accidental modification.

3. **Class Constants**: _HIGH_CARDINALITY_KEYS is a frozenset (immutable),
   ensuring thread-safe read access.

4. **Instance Variables**: The only instance variable (__metrics_sink) is set
   once during __init__ and exposed via a read-only property, ensuring
   thread-safe reads and preventing modification after construction.

5. **Module-Level Singleton**: The global singleton hook instance is protected
   by a threading.Lock for thread-safe initialization across threads.

Singleton Support
-----------------
This module provides optional singleton support via get_global_hook() and
configure_global_hook(). The singleton pattern is useful when:

- Multiple components need to share the same observability infrastructure
- You want centralized metrics collection across the application
- Resource efficiency is important

Important: When a singleton already exists, calling configure_global_hook()
with different configuration will log a warning and return the existing
instance. Use clear_global_hook() to reset the singleton if reconfiguration
is needed.

Usage Example:
    ```python
    from omnibase_infra.observability.hooks import HookObservability
    from omnibase_spi.protocols.observability import ProtocolHotPathMetricsSink

    # Create hook with optional metrics sink
    sink: ProtocolHotPathMetricsSink = get_metrics_sink()
    hook = HookObservability(metrics_sink=sink)

    # Use in handler execution
    hook.before_operation("handler.execute", correlation_id="abc-123")
    try:
        result = await handler.execute(payload)
        hook.record_success()
    except Exception as e:
        hook.record_failure(str(type(e).__name__))
        raise
    finally:
        duration_ms = hook.after_operation()
        logger.info(f"Operation took {duration_ms:.2f}ms")

    # Or use the global singleton
    from omnibase_infra.observability.hooks import get_global_hook
    hook = get_global_hook()  # Returns singleton, creates if needed
    ```

See Also:
    - ProtocolHotPathMetricsSink: Metrics collection interface
    - correlation.py: Correlation ID context management pattern
    - docs/patterns/observability_patterns.md: Observability guidelines
"""

from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING
from uuid import UUID

_logger = logging.getLogger(__name__)

# =============================================================================
# MODULE-LEVEL SINGLETON STATE
# =============================================================================
#
# Thread-safe singleton support for global hook instance. Protected by
# _global_hook_lock for safe initialization from multiple threads.
#
# =============================================================================

# Forward reference resolved by `from __future__ import annotations`
_global_hook_instance: HookObservability | None = None
_global_hook_lock = threading.Lock()

if TYPE_CHECKING:
    from types import TracebackType

    from omnibase_spi.protocols.observability import ProtocolHotPathMetricsSink

# =============================================================================
# CONTEXT VARIABLES FOR CONCURRENCY-SAFE OPERATION TRACKING
# =============================================================================
#
# These ContextVars provide per-async-task isolation for operation state.
# Each concurrent operation gets its own isolated copy of these values,
# preventing race conditions in high-concurrency environments.
#
# DO NOT convert these to instance variables - that would break concurrency!
# =============================================================================

# Operation start time in perf_counter units (high-resolution monotonic clock)
_start_time: ContextVar[float | None] = ContextVar("hook_start_time", default=None)

# Current operation name (e.g., "handler.execute", "retry.attempt")
_operation_name: ContextVar[str | None] = ContextVar(
    "hook_operation_name", default=None
)

# Correlation ID for distributed tracing (propagated from request context)
_correlation_id: ContextVar[str | None] = ContextVar(
    "hook_correlation_id", default=None
)

# Additional operation labels for metrics (e.g., handler name, status)
# Note: ContextVar doesn't support default_factory, so we use None and handle it
_operation_labels: ContextVar[dict[str, str] | None] = ContextVar(
    "hook_operation_labels", default=None
)


class HookObservability:
    """Pipeline hook for cross-cutting observability instrumentation.

    This hook provides timing, metrics, and context management for infrastructure
    operations. It uses contextvars for all state to ensure concurrency safety
    in async code paths.

    Key Features:
        - Concurrency-safe timing via contextvars (NOT instance variables)
        - Metrics emission via ProtocolHotPathMetricsSink
        - Operation context propagation across async boundaries
        - Support for nested operation tracking via context manager
        - High-cardinality label filtering to prevent metric explosion

    Thread-Safety Guarantees:
        This class is safe for concurrent use from multiple async tasks.
        All timing and operation state is stored in contextvars, which provide
        per-task isolation. The metrics sink (if provided) MUST be thread-safe.

        Specific guarantees:
        1. Timing state (start_time, operation_name, etc.) is isolated per-task
        2. The metrics_sink is set once in __init__ and never modified
        3. _HIGH_CARDINALITY_KEYS is immutable (frozenset)
        4. No mutable shared state exists between concurrent operations

    High-Cardinality Label Filtering:
        Labels containing high-cardinality keys (correlation_id, request_id,
        trace_id, span_id, session_id, user_id) are automatically filtered
        from metrics to prevent cardinality explosion. These values remain
        available via get_current_context() for logging and tracing.

        When labels are filtered, a debug log is emitted to aid troubleshooting.
        Metrics are NEVER dropped entirely - only high-cardinality labels are
        removed from the label set.

    Metrics Emitted:
        - `operation_started_total`: Counter incremented when operation starts
        - `operation_completed_total`: Counter incremented when operation completes
        - `operation_failed_total`: Counter incremented on failure
        - `operation_duration_seconds`: Histogram of operation durations
        - `retry_attempt_total`: Counter for retry attempts
        - `circuit_breaker_state_change_total`: Counter for circuit state changes

    Attributes:
        metrics_sink: Read-only property for the metrics sink. Returns None if
            no sink was provided. The sink MUST be thread-safe for concurrent
            access from multiple async tasks.

    Example:
        ```python
        hook = HookObservability(metrics_sink=sink)

        # Manual instrumentation
        hook.before_operation("db.query", correlation_id="req-123")
        try:
            result = await db.execute(query)
            hook.record_success()
        except Exception:
            hook.record_failure("DatabaseError")
            raise
        finally:
            duration = hook.after_operation()

        # Context manager for automatic timing
        with hook.operation_context("http.request", correlation_id="req-456"):
            response = await http_client.get(url)
        ```
    """

    # Use __slots__ to prevent accidental attribute addition and improve memory
    # _metrics_lock provides defense-in-depth thread safety for metrics operations
    __slots__ = ("__metrics_sink", "_metrics_lock")

    def __init__(
        self,
        metrics_sink: ProtocolHotPathMetricsSink | None = None,
    ) -> None:
        """Initialize the observability hook.

        Args:
            metrics_sink: Optional metrics sink for emitting observability data.
                If None, the hook operates in no-op mode for metrics (timing
                is still tracked). This allows the hook to be used even when
                metrics infrastructure is not available.

        Thread-Safety Requirements:
            The metrics_sink (if provided) MUST be thread-safe for concurrent
            access from multiple async tasks. The built-in SinkMetricsPrometheus
            satisfies this requirement. If using a custom sink implementation,
            ensure all methods (increment_counter, set_gauge, observe_histogram)
            are thread-safe.

        Note:
            The metrics_sink is stored as a private instance variable and
            exposed via a read-only property. This prevents accidental
            modification after construction, ensuring thread-safe reads.
            This is intentionally different from timing state which MUST be
            in contextvars for per-task isolation.

        Thread-Safety Implementation:
            A threading.Lock (_metrics_lock) is used for defense-in-depth protection
            of all metrics sink operations. While the metrics_sink is expected to be
            thread-safe, this lock ensures atomic operation sequences and protects
            against potential subtle thread-safety issues in sink implementations.
        """
        # Use name mangling (__) to prevent external modification
        # Access via the metrics_sink property
        self.__metrics_sink = metrics_sink
        # Defense-in-depth lock for metrics operations
        # Ensures atomic counter increments and metric emissions even if sink
        # has subtle thread-safety issues
        self._metrics_lock = threading.Lock()

    @property
    def metrics_sink(self) -> ProtocolHotPathMetricsSink | None:
        """Read-only access to the metrics sink.

        Returns:
            The metrics sink provided during construction, or None if no
            sink was provided.

        Thread-Safety:
            This property is thread-safe. The underlying value is set once
            during __init__ and never modified.
        """
        return self.__metrics_sink

    # =========================================================================
    # CORE TIMING API
    # =========================================================================

    def before_operation(
        self,
        operation: str,
        correlation_id: str | UUID | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Mark the start of an operation for timing.

        Sets up the timing context for the current async task. This method
        MUST be called before after_operation() to establish the start time.

        Concurrency Safety:
            Uses contextvars to store timing state, ensuring each async task
            has its own isolated start time. Multiple concurrent operations
            will not interfere with each other.

        Args:
            operation: Name of the operation being tracked. Should follow
                a dotted naming convention (e.g., "handler.execute",
                "db.query", "http.request"). This is used as the metric name.
            correlation_id: Optional correlation ID for distributed tracing.
                Can be a string or UUID. If UUID, it will be converted to string.
            labels: Optional additional labels to attach to metrics. Keys and
                values must be strings. Common labels: "handler", "status".

        Side Effects:
            - Sets _start_time contextvar to current perf_counter value
            - Sets _operation_name contextvar to operation parameter
            - Sets _correlation_id contextvar if provided
            - Sets _operation_labels contextvar if provided
            - Increments "operation_started_total" counter if metrics sink present

        Example:
            ```python
            hook.before_operation(
                "handler.process",
                correlation_id="abc-123",
                labels={"handler": "UserHandler"},
            )
            ```
        """
        # Store timing state in contextvars for concurrency safety
        _start_time.set(time.perf_counter())
        _operation_name.set(operation)

        # Convert UUID to string if needed
        if correlation_id is not None:
            _correlation_id.set(str(correlation_id))
        else:
            _correlation_id.set(None)

        # Store labels (or empty dict if none provided)
        _operation_labels.set(labels.copy() if labels else {})

        # Emit start metric if sink is available
        # Lock ensures atomic counter increment across concurrent calls
        if self.metrics_sink is not None:
            metric_labels = self._build_metric_labels(operation)
            with self._metrics_lock:
                self.metrics_sink.increment_counter(
                    name="operation_started_total",
                    labels=metric_labels,
                    increment=1,
                )

    def after_operation(self) -> float:
        """Mark the end of an operation and calculate duration.

        Calculates the elapsed time since before_operation() was called and
        optionally emits the duration as a histogram observation.

        Concurrency Safety:
            Reads timing state from contextvars, which are isolated per async
            task. The returned duration is specific to the current task's
            operation timing.

        Returns:
            Duration in milliseconds since before_operation() was called.
            Returns 0.0 if before_operation() was not called (start_time is None).

        Side Effects:
            - Observes "operation_duration_seconds" histogram if metrics sink present
            - Clears _start_time contextvar (sets to None)
            - Clears _operation_name contextvar (sets to None)
            - Does NOT clear correlation_id (may be needed for error handling)

        Example:
            ```python
            hook.before_operation("db.query")
            result = await db.execute(query)
            duration_ms = hook.after_operation()  # e.g., 42.5
            logger.info(f"Query took {duration_ms:.2f}ms")
            ```
        """
        start = _start_time.get()
        operation = _operation_name.get()

        # Handle case where before_operation was not called
        if start is None:
            return 0.0

        # Calculate duration
        end = time.perf_counter()
        duration_seconds = end - start
        duration_ms = duration_seconds * 1000.0

        # Emit duration metric if sink is available
        # Lock ensures atomic histogram observation across concurrent calls
        if self.metrics_sink is not None and operation is not None:
            metric_labels = self._build_metric_labels(operation)
            with self._metrics_lock:
                self.metrics_sink.observe_histogram(
                    name="operation_duration_seconds",
                    labels=metric_labels,
                    value=duration_seconds,
                )

        # Clear timing state (but keep correlation_id for potential error handling)
        _start_time.set(None)
        _operation_name.set(None)
        _operation_labels.set(None)

        return duration_ms

    def get_current_context(self) -> dict[str, str | None]:
        """Get the current operation context from contextvars.

        Returns the current operation context including operation name,
        correlation ID, and any additional labels. Useful for logging
        and debugging.

        Concurrency Safety:
            Reads from contextvars, returning context specific to the
            current async task.

        Returns:
            Dictionary containing:
                - "operation": Current operation name or None
                - "correlation_id": Current correlation ID or None
                - Plus any additional labels from _operation_labels

        Example:
            ```python
            hook.before_operation("handler.process", correlation_id="abc-123")
            ctx = hook.get_current_context()
            # ctx = {"operation": "handler.process", "correlation_id": "abc-123"}
            logger.info("Processing", extra=ctx)
            ```
        """
        result: dict[str, str | None] = {
            "operation": _operation_name.get(),
            "correlation_id": _correlation_id.get(),
        }

        # Add any additional labels (they're already string -> string)
        labels = _operation_labels.get()
        if labels is not None:
            for key, value in labels.items():
                result[key] = value

        return result

    # =========================================================================
    # SUCCESS/FAILURE TRACKING
    # =========================================================================

    def record_success(self, labels: dict[str, str] | None = None) -> None:
        """Record a successful operation completion.

        Increments the operation_completed_total counter with success status.
        Should be called after the operation completes successfully, before
        after_operation().

        Args:
            labels: Optional additional labels to merge with operation labels.

        Side Effects:
            - Increments "operation_completed_total" counter with status="success"

        Example:
            ```python
            hook.before_operation("handler.process")
            result = await handler.execute()
            hook.record_success()
            duration = hook.after_operation()
            ```
        """
        if self.metrics_sink is None:
            return

        operation = _operation_name.get()
        if operation is None:
            return

        metric_labels = self._build_metric_labels(operation, labels)
        metric_labels["status"] = "success"

        # Lock ensures atomic counter increment across concurrent calls
        with self._metrics_lock:
            self.metrics_sink.increment_counter(
                name="operation_completed_total",
                labels=metric_labels,
                increment=1,
            )

    def record_failure(
        self,
        error_type: str,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a failed operation.

        Increments the operation_failed_total counter with the error type.
        Should be called when an operation fails, before after_operation().

        Args:
            error_type: Type/class name of the error that occurred.
                Should be a stable identifier (e.g., "TimeoutError",
                "DatabaseConnectionError"), not the error message.
            labels: Optional additional labels to merge with operation labels.

        Side Effects:
            - Increments "operation_failed_total" counter with error_type label

        Example:
            ```python
            hook.before_operation("db.query")
            try:
                result = await db.execute(query)
                hook.record_success()
            except DatabaseError as e:
                hook.record_failure("DatabaseError")
                raise
            finally:
                hook.after_operation()
            ```
        """
        if self.metrics_sink is None:
            return

        operation = _operation_name.get()
        if operation is None:
            return

        metric_labels = self._build_metric_labels(operation, labels)
        metric_labels["status"] = "failure"
        metric_labels["error_type"] = error_type

        # Lock ensures atomic counter increment across concurrent calls
        with self._metrics_lock:
            self.metrics_sink.increment_counter(
                name="operation_failed_total",
                labels=metric_labels,
                increment=1,
            )

    # =========================================================================
    # SPECIALIZED TRACKING METHODS
    # =========================================================================

    def record_retry_attempt(
        self,
        attempt_number: int,
        reason: str,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a retry attempt for an operation.

        Tracks retry attempts with attempt number and reason. Useful for
        monitoring retry behavior and identifying flaky operations.

        Args:
            attempt_number: The current attempt number (1-based). First attempt
                is 1, first retry is 2, etc.
            reason: Reason for the retry (e.g., "timeout", "connection_reset",
                "rate_limited"). Should be a stable identifier.
            labels: Optional additional labels to merge with operation labels.

        Side Effects:
            - Increments "retry_attempt_total" counter

        Example:
            ```python
            for attempt in range(1, max_retries + 1):
                try:
                    result = await operation()
                    break
                except RetryableError as e:
                    hook.record_retry_attempt(attempt, "transient_error")
                    if attempt == max_retries:
                        raise
            ```
        """
        if self.metrics_sink is None:
            return

        operation = _operation_name.get() or "unknown"
        metric_labels = self._build_metric_labels(operation, labels)
        metric_labels["attempt"] = str(attempt_number)
        metric_labels["reason"] = reason

        # Lock ensures atomic counter increment across concurrent calls
        with self._metrics_lock:
            self.metrics_sink.increment_counter(
                name="retry_attempt_total",
                labels=metric_labels,
                increment=1,
            )

    def record_circuit_breaker_state_change(
        self,
        service_name: str,
        from_state: str,
        to_state: str,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a circuit breaker state transition.

        Tracks circuit breaker state changes for monitoring circuit health
        and identifying unstable services.

        Args:
            service_name: Name of the service protected by the circuit breaker.
            from_state: Previous circuit state (e.g., "CLOSED", "OPEN", "HALF_OPEN").
            to_state: New circuit state after transition.
            labels: Optional additional labels.

        Side Effects:
            - Increments "circuit_breaker_state_change_total" counter

        Example:
            ```python
            # In circuit breaker implementation
            hook.record_circuit_breaker_state_change(
                service_name="database",
                from_state="CLOSED",
                to_state="OPEN",
            )
            ```
        """
        if self.metrics_sink is None:
            return

        metric_labels: dict[str, str] = {
            "service": service_name,
            "from_state": from_state,
            "to_state": to_state,
        }

        # Merge additional labels
        if labels:
            for key, value in labels.items():
                if key not in metric_labels:  # Don't overwrite required labels
                    metric_labels[key] = value

        # Lock ensures atomic counter increment across concurrent calls
        with self._metrics_lock:
            self.metrics_sink.increment_counter(
                name="circuit_breaker_state_change_total",
                labels=metric_labels,
                increment=1,
            )

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric value.

        Convenience method for setting gauge values through the hook.
        Useful for tracking current state like queue depths or active connections.

        Note:
            Gauges can legitimately be negative (e.g., temperature, delta values).
            For buffer/queue metrics that should never be negative, use
            set_buffer_gauge() instead, which enforces non-negative values.

        Args:
            name: Metric name following Prometheus conventions.
            value: Current gauge value. Can be negative for appropriate metrics.
            labels: Optional labels for the metric.

        Side Effects:
            - Sets gauge metric via metrics sink

        Example:
            ```python
            hook.set_gauge(
                "active_handlers",
                value=len(active_handlers),
                labels={"handler_type": "http"},
            )
            ```
        """
        if self.metrics_sink is None:
            return

        # Lock ensures atomic gauge update across concurrent calls
        with self._metrics_lock:
            self.metrics_sink.set_gauge(
                name=name,
                labels=labels or {},
                value=value,
            )

    def set_buffer_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric value for buffer/queue metrics (non-negative).

        Similar to set_gauge(), but enforces non-negative values. Use this for
        metrics that represent counts, sizes, or capacities that logically
        cannot be negative (e.g., queue depth, buffer size, connection pool size).

        The value is clamped to 0.0 if negative, preventing invalid metric values
        that could occur from race conditions or calculation errors. A warning
        is logged when clamping occurs to aid debugging.

        Args:
            name: Metric name following Prometheus conventions.
            value: Current buffer/queue value. Clamped to 0.0 if negative.
            labels: Optional labels for the metric.

        Side Effects:
            - Sets gauge metric via metrics sink with max(0.0, value)
            - Logs warning if negative value is clamped

        Example:
            ```python
            # Queue depth can't go negative even with concurrent updates
            hook.set_buffer_gauge(
                "message_queue_depth",
                value=queue.qsize(),
                labels={"queue_name": "events"},
            )

            # Safe even if calculation error produces negative
            hook.set_buffer_gauge(
                "buffer_available_slots",
                value=total_slots - used_slots,  # Clamped to 0 if oversubscribed
                labels={"buffer": "write"},
            )
            ```
        """
        if self.metrics_sink is None:
            return

        # Enforce non-negative values for buffer/count metrics
        if value < 0.0:
            _logger.warning(
                "Buffer gauge received negative value; clamping to 0.0",
                extra={
                    "metric_name": name,
                    "original_value": value,
                    "labels": labels,
                },
            )
            safe_value = 0.0
        else:
            safe_value = value

        # Lock ensures atomic gauge update across concurrent calls
        with self._metrics_lock:
            self.metrics_sink.set_gauge(
                name=name,
                labels=labels or {},
                value=safe_value,
            )

    # =========================================================================
    # CONTEXT MANAGER SUPPORT
    # =========================================================================

    def operation_context(
        self,
        operation: str,
        correlation_id: str | UUID | None = None,
        labels: dict[str, str] | None = None,
    ) -> OperationScope:
        """Create a context manager for operation timing.

        Returns a context manager that automatically calls before_operation()
        on entry and after_operation() on exit. This is the recommended way
        to instrument operations as it ensures proper cleanup even on exceptions.

        Args:
            operation: Name of the operation to track.
            correlation_id: Optional correlation ID for tracing.
            labels: Optional additional labels for metrics.

        Returns:
            A context manager that yields the duration in milliseconds on exit.

        Example:
            ```python
            # Basic usage
            with hook.operation_context("handler.process") as ctx:
                result = await handler.execute()
            print(f"Operation took {ctx.duration_ms:.2f}ms")

            # With correlation ID
            with hook.operation_context(
                "db.query",
                correlation_id=request_id,
                labels={"table": "users"},
            ):
                rows = await db.fetch_all(query)
            ```
        """
        return OperationScope(
            hook=self,
            operation=operation,
            correlation_id=correlation_id,
            labels=labels,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    # High-cardinality label keys that must be excluded from metrics.
    # These would cause metrics cardinality explosion and be dropped by
    # ModelMetricsPolicy when on_violation is WARN_AND_DROP or DROP_SILENT.
    _HIGH_CARDINALITY_KEYS: frozenset[str] = frozenset(
        {
            "correlation_id",
            "request_id",
            "trace_id",
            "span_id",
            "session_id",
            "user_id",
        }
    )

    def _build_metric_labels(
        self,
        operation: str,
        extra_labels: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build the complete label set for a metric.

        Combines operation name, stored operation labels, and any extra labels
        into a single label dictionary. High-cardinality keys are automatically
        filtered out to prevent metrics from being dropped by the policy.

        CRITICAL - Metric Recording Guarantee:
            This method NEVER drops or suppresses metric recording. When called,
            the metric WILL be recorded with the returned labels. The only
            filtering that occurs is removal of high-cardinality label KEYS
            from the label dictionary - the metric itself is ALWAYS recorded.

            Example with correlation_id:
                - Input: operation="db.query", labels={"correlation_id": "abc-123"}
                - Output: {"operation": "db.query"}  # correlation_id filtered
                - Metric: RECORDED with {"operation": "db.query"}

            The correlation_id and other high-cardinality values remain available
            via the _correlation_id contextvar for structured logging and
            distributed tracing - they are just excluded from Prometheus labels
            to prevent cardinality explosion.

        Why Filter High-Cardinality Labels:
            High-cardinality values (correlation_id, request_id, trace_id, etc.)
            are unique per request. Including them in Prometheus labels would:
            1. Create millions of unique time series (cardinality explosion)
            2. Cause metrics to be dropped by ModelMetricsPolicy (WARN_AND_DROP)
            3. Overwhelm Prometheus storage and query performance

            By filtering these keys early, we ensure metrics are ALWAYS recorded
            with stable, low-cardinality labels.

        Args:
            operation: Operation name to include. Always present in output.
            extra_labels: Optional additional labels to merge.

        Returns:
            Complete label dictionary for metric emission, with high-cardinality
            keys filtered out. GUARANTEED to contain at least {"operation": operation}.
            Never returns empty dict. Never returns None.
        """
        # INVARIANT: labels always contains at least {"operation": operation}
        # This ensures metrics are NEVER dropped due to empty labels
        labels: dict[str, str] = {"operation": operation}
        filtered_keys: list[str] = []

        # Merge stored operation labels, filtering out high-cardinality keys
        # Note: stored_labels dict is a copy made in before_operation(), safe to iterate
        stored_labels = _operation_labels.get()
        if stored_labels is not None:
            for key, value in stored_labels.items():
                if key in self._HIGH_CARDINALITY_KEYS:
                    filtered_keys.append(key)
                else:
                    labels[key] = value

        # Merge extra labels (overrides stored if same key), filtering high-cardinality
        if extra_labels:
            for key, value in extra_labels.items():
                if key in self._HIGH_CARDINALITY_KEYS:
                    # Only add to filtered_keys if not already tracked
                    if key not in filtered_keys:
                        filtered_keys.append(key)
                else:
                    labels[key] = value

        # Log when high-cardinality keys are filtered (debug level)
        # IMPORTANT: This is informational logging only - the metric IS being recorded
        # We only remove specific label KEYS, NOT the entire metric
        # The log includes correlation_id for tracing even though it's filtered from labels
        if filtered_keys:
            # Include correlation_id in log for tracing purposes
            correlation_id = _correlation_id.get()
            _logger.debug(
                "Removed high-cardinality keys from metric labels - "
                "metric WILL be recorded with %d remaining labels (keys removed: %s)",
                len(labels),
                filtered_keys,
                extra={
                    "operation": operation,
                    "filtered_keys": filtered_keys,
                    "remaining_labels": list(labels.keys()),
                    "correlation_id": correlation_id,  # Available for log correlation
                },
            )

        # CRITICAL: Defense-in-depth guarantee that metrics are NEVER dropped.
        # The invariant at line 889 ensures "operation" is always present, but this
        # explicit check provides runtime safety if the invariant is ever violated.
        # This protects against data loss from unexpected edge cases.
        if "operation" not in labels:
            _logger.error(
                "BUG: operation key missing from labels after filtering; "
                "restoring to prevent metric data loss",
                extra={"operation": operation, "labels_keys": list(labels.keys())},
            )
            labels["operation"] = operation

        # GUARANTEE: This method ALWAYS returns a non-empty dict containing at least
        # {"operation": operation}. Metrics are NEVER dropped - only high-cardinality
        # label keys (correlation_id, request_id, etc.) are removed from the label set.
        # The metric itself is ALWAYS recorded with the remaining labels.
        return labels


class OperationScope:
    """Context manager for scoped operation timing.

    This internal class provides context manager support for HookObservability.
    It automatically calls before_operation() on entry and after_operation()
    on exit, ensuring proper cleanup even when exceptions occur.

    Attributes:
        duration_ms: Duration of the operation in milliseconds, available
            after the context exits. Will be 0.0 if accessed before exit.

    Note:
        This class stores tokens for contextvar restoration, enabling proper
        nesting of operation contexts. Each context saves and restores the
        previous contextvar values.
    """

    def __init__(
        self,
        hook: HookObservability,
        operation: str,
        correlation_id: str | UUID | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Initialize the context manager.

        Args:
            hook: The HookObservability instance to use.
            operation: Operation name to track.
            correlation_id: Optional correlation ID.
            labels: Optional additional labels.
        """
        self._hook = hook
        self._operation = operation
        self._correlation_id = correlation_id
        self._labels = labels
        self.duration_ms: float = 0.0

        # Tokens for restoring previous contextvar values (for nesting support)
        self._start_time_token: Token[float | None] | None = None
        self._operation_name_token: Token[str | None] | None = None
        self._correlation_id_token: Token[str | None] | None = None
        self._labels_token: Token[dict[str, str] | None] | None = None

    def __enter__(self) -> OperationScope:
        """Enter the operation context.

        Saves current contextvar values and calls before_operation().

        Returns:
            Self, for accessing duration_ms after exit.
        """
        # Save current values for restoration on exit (nesting support)
        self._start_time_token = _start_time.set(_start_time.get())
        self._operation_name_token = _operation_name.set(_operation_name.get())
        self._correlation_id_token = _correlation_id.set(_correlation_id.get())
        current_labels = _operation_labels.get()
        # NOTE: Shallow copy is sufficient here because:
        # 1. Labels dict is typed as dict[str, str] (string keys and values)
        # 2. Strings are immutable in Python, so no aliasing issues can occur
        # 3. We only need isolation of the dict structure, not deep cloning of values
        self._labels_token = _operation_labels.set(
            current_labels.copy() if current_labels is not None else None
        )

        # Now start the new operation
        self._hook.before_operation(
            operation=self._operation,
            correlation_id=self._correlation_id,
            labels=self._labels,
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the operation context.

        Calls after_operation() and restores previous contextvar values.
        Records success or failure based on whether an exception occurred.

        Concurrency Safety:
            Uses try/finally to ensure contextvar tokens are ALWAYS restored,
            even if record_failure/record_success/after_operation raise exceptions.
            This prevents contextvar state leakage in error scenarios.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Traceback if an exception was raised.
        """
        try:
            # Record success or failure before timing
            if exc_type is not None:
                self._hook.record_failure(exc_type.__name__)
            else:
                self._hook.record_success()

            # Get duration
            self.duration_ms = self._hook.after_operation()
        finally:
            # CRITICAL: Always restore previous contextvar values (for nesting support)
            # This must happen even if the above code raises an exception to prevent
            # contextvar state leakage.
            if self._start_time_token is not None:
                _start_time.reset(self._start_time_token)
            if self._operation_name_token is not None:
                _operation_name.reset(self._operation_name_token)
            if self._correlation_id_token is not None:
                _correlation_id.reset(self._correlation_id_token)
            if self._labels_token is not None:
                _operation_labels.reset(self._labels_token)


# =============================================================================
# MODULE-LEVEL SINGLETON FUNCTIONS
# =============================================================================
#
# These functions provide optional singleton access to a global HookObservability
# instance. The singleton is thread-safe and lazily initialized.
#
# =============================================================================


def get_global_hook(
    metrics_sink: ProtocolHotPathMetricsSink | None = None,
) -> HookObservability:
    """Get or create the global singleton HookObservability instance.

    Returns the cached singleton hook if one exists, otherwise creates a new
    one with the provided configuration and caches it. This is the recommended
    way to access a shared observability hook across the application.

    Thread Safety:
        This function is thread-safe. Multiple concurrent calls will receive
        the same singleton instance. A threading.Lock ensures only one thread
        creates the singleton.

    Note:
        The metrics_sink parameter is only used on first call when the singleton
        is created. Subsequent calls ignore this parameter and return the
        existing instance. A warning is logged when configuration is provided
        but ignored due to an existing singleton.

        To reconfigure the singleton, call clear_global_hook() first.

    Args:
        metrics_sink: Optional metrics sink for the hook. Only used if no
            singleton exists yet. Subsequent calls ignore this parameter.

    Returns:
        The global singleton HookObservability instance.

    Example:
        ```python
        # First call creates the singleton with the provided sink
        hook1 = get_global_hook(metrics_sink=my_sink)

        # Subsequent calls return the same instance
        hook2 = get_global_hook()  # metrics_sink parameter ignored
        assert hook1 is hook2

        # Reconfigure if needed
        clear_global_hook()
        hook3 = get_global_hook(metrics_sink=new_sink)  # New singleton
        ```
    """
    global _global_hook_instance  # noqa: PLW0603 - Standard singleton pattern

    with _global_hook_lock:
        if _global_hook_instance is None:
            _global_hook_instance = HookObservability(metrics_sink=metrics_sink)
            _logger.debug(
                "Created global singleton HookObservability",
                extra={"has_metrics_sink": metrics_sink is not None},
            )
        elif metrics_sink is not None:
            # Log warning when config is provided but ignored for existing singleton
            _logger.warning(
                "Global HookObservability singleton already exists; "
                "provided metrics_sink configuration ignored. "
                "Call clear_global_hook() first to reconfigure.",
                extra={
                    "existing_has_metrics_sink": (
                        _global_hook_instance.metrics_sink is not None
                    ),
                    "provided_metrics_sink_type": type(metrics_sink).__name__,
                },
            )
        return _global_hook_instance


def configure_global_hook(
    metrics_sink: ProtocolHotPathMetricsSink | None = None,
) -> HookObservability:
    """Configure the global singleton HookObservability instance.

    This is an alias for get_global_hook() that makes the intent clearer when
    you want to explicitly configure the singleton on first use.

    Args:
        metrics_sink: Optional metrics sink for the hook.

    Returns:
        The global singleton HookObservability instance.

    See Also:
        get_global_hook(): Primary singleton access function.
        clear_global_hook(): Reset the singleton for reconfiguration.
    """
    return get_global_hook(metrics_sink=metrics_sink)


def clear_global_hook() -> None:
    """Clear the global singleton HookObservability instance.

    Removes the reference to the cached singleton hook, allowing it to be
    garbage collected if no other references exist. Subsequent calls to
    get_global_hook() will create a new instance.

    Use Cases:
        - Testing: Reset singleton state between tests
        - Reconfiguration: Allow new singleton with different metrics_sink
        - Shutdown: Release resources before application exit

    Thread Safety:
        This function is thread-safe.

    Warning:
        Existing references to the old singleton remain valid. Only future
        get_global_hook() calls will create a new instance.

    Example:
        ```python
        hook1 = get_global_hook(metrics_sink=sink1)
        clear_global_hook()
        hook2 = get_global_hook(metrics_sink=sink2)
        assert hook1 is not hook2  # Different instances
        # hook1 still works but is no longer the singleton
        ```
    """
    global _global_hook_instance  # noqa: PLW0603 - Standard singleton pattern

    with _global_hook_lock:
        if _global_hook_instance is not None:
            _logger.debug("Cleared global HookObservability singleton")
        _global_hook_instance = None


def has_global_hook() -> bool:
    """Check if a global singleton HookObservability exists.

    Returns:
        True if a singleton hook has been created, False otherwise.

    Thread Safety:
        This function is thread-safe.
    """
    with _global_hook_lock:
        return _global_hook_instance is not None


__all__ = [
    "HookObservability",
    "get_global_hook",
    "configure_global_hook",
    "clear_global_hook",
    "has_global_hook",
]
