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
    ```

See Also:
    - ProtocolHotPathMetricsSink: Metrics collection interface
    - correlation.py: Correlation ID context management pattern
    - docs/patterns/observability_patterns.md: Observability guidelines
"""

from __future__ import annotations

import time
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING
from uuid import UUID

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

    Thread-Safety:
        This class is safe for concurrent use from multiple async tasks.
        All timing and operation state is stored in contextvars, which provide
        per-task isolation. The metrics sink (if provided) should also be
        thread-safe.

    Metrics Emitted:
        - `operation_started_total`: Counter incremented when operation starts
        - `operation_completed_total`: Counter incremented when operation completes
        - `operation_failed_total`: Counter incremented on failure
        - `operation_duration_seconds`: Histogram of operation durations
        - `retry_attempt_total`: Counter for retry attempts
        - `circuit_breaker_state_change_total`: Counter for circuit state changes

    Attributes:
        metrics_sink: Optional metrics sink for emitting observability data.
            If None, metrics emission is skipped (no-op).

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

        Note:
            The metrics_sink is stored as an instance variable because it is
            a shared resource, not per-operation state. This is intentionally
            different from the timing state which MUST be in contextvars.
        """
        self._metrics_sink = metrics_sink

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
        if self._metrics_sink is not None:
            metric_labels = self._build_metric_labels(operation)
            self._metrics_sink.increment_counter(
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
        if self._metrics_sink is not None and operation is not None:
            metric_labels = self._build_metric_labels(operation)
            self._metrics_sink.observe_histogram(
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
        if self._metrics_sink is None:
            return

        operation = _operation_name.get()
        if operation is None:
            return

        metric_labels = self._build_metric_labels(operation, labels)
        metric_labels["status"] = "success"

        self._metrics_sink.increment_counter(
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
        if self._metrics_sink is None:
            return

        operation = _operation_name.get()
        if operation is None:
            return

        metric_labels = self._build_metric_labels(operation, labels)
        metric_labels["status"] = "failure"
        metric_labels["error_type"] = error_type

        self._metrics_sink.increment_counter(
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
        if self._metrics_sink is None:
            return

        operation = _operation_name.get() or "unknown"
        metric_labels = self._build_metric_labels(operation, labels)
        metric_labels["attempt"] = str(attempt_number)
        metric_labels["reason"] = reason

        self._metrics_sink.increment_counter(
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
        if self._metrics_sink is None:
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

        self._metrics_sink.increment_counter(
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
        if self._metrics_sink is None:
            return

        self._metrics_sink.set_gauge(
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
        that could occur from race conditions or calculation errors.

        Args:
            name: Metric name following Prometheus conventions.
            value: Current buffer/queue value. Clamped to 0.0 if negative.
            labels: Optional labels for the metric.

        Side Effects:
            - Sets gauge metric via metrics sink with max(0.0, value)

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
        if self._metrics_sink is None:
            return

        # Enforce non-negative values for buffer/count metrics
        safe_value = max(0.0, value)

        self._metrics_sink.set_gauge(
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

        Note:
            High-cardinality values (correlation_id, request_id, trace_id, etc.)
            are intentionally EXCLUDED from metric labels. These are unique per
            request and would cause metrics to be dropped when ModelMetricsPolicy's
            on_violation is set to WARN_AND_DROP or DROP_SILENT. The correlation_id
            remains available via _correlation_id contextvar for structured logging
            and distributed tracing purposes.

        Args:
            operation: Operation name to include.
            extra_labels: Optional additional labels to merge.

        Returns:
            Complete label dictionary for metric emission, with high-cardinality
            keys filtered out.
        """
        labels: dict[str, str] = {"operation": operation}

        # Merge stored operation labels, filtering out high-cardinality keys
        stored_labels = _operation_labels.get()
        if stored_labels is not None:
            for key, value in stored_labels.items():
                if key not in self._HIGH_CARDINALITY_KEYS:
                    labels[key] = value

        # Merge extra labels (overrides stored if same key), filtering high-cardinality
        if extra_labels:
            for key, value in extra_labels.items():
                if key not in self._HIGH_CARDINALITY_KEYS:
                    labels[key] = value

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


__all__ = ["HookObservability"]
