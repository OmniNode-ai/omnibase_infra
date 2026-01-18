# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HookObservability.

This module tests the observability hook implementation with critical focus on:
- Contextvars isolation (no shared state between async operations)
- before_operation/after_operation timing accuracy
- Metrics integration with ProtocolHotPathMetricsSink
- Context manager support for scoped timing
- Concurrent operations don't interfere with each other

CRITICAL: Contextvars Isolation
-------------------------------
The HookObservability class uses contextvars instead of instance variables for
all timing and operation state. This is a CRITICAL design decision for async
safety. These tests verify this behavior extensively.

Why NOT instance variables:
    # WRONG - Race condition in async code!
    class BadHook:
        def __init__(self):
            self._start_time = 0.0  # Shared across all concurrent operations!

    # In async code, multiple tasks share the same instance:
    hook.before_operation("op_a")  # Sets _start_time = 100
    # Task switches to another operation
    hook.before_operation("op_b")  # OVERWRITES _start_time = 200
    # Back to op_a, but _start_time is wrong!
    hook.after_operation()  # Returns wrong duration!

Why contextvars ARE correct:
    Each async task gets its own isolated context. Even with the same hook
    instance, concurrent operations don't interfere because contextvars
    provide per-task storage.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from omnibase_infra.observability.hooks import HookObservability

# =============================================================================
# CONTEXTVARS ISOLATION TESTS - CRITICAL
# =============================================================================


class TestContextvarsIsolation:
    """Test contextvars isolation ensures no shared state between async ops.

    These are the most critical tests for the hook. They verify that
    concurrent async operations using the same hook instance do not
    interfere with each other's timing or context.
    """

    @pytest.mark.asyncio
    async def test_concurrent_operations_isolated_timing(self) -> None:
        """Verify concurrent operations have isolated timing.

        This is the primary test for contextvars isolation. Two concurrent
        operations should each get their own correct duration, regardless
        of which completes first.
        """
        hook = HookObservability(metrics_sink=None)

        async def operation_a() -> float:
            """Long-running operation (100ms)."""
            hook.before_operation("op_a", correlation_id="corr-a")
            await asyncio.sleep(0.1)  # 100ms
            duration = hook.after_operation()
            return duration

        async def operation_b() -> float:
            """Short-running operation (50ms)."""
            hook.before_operation("op_b", correlation_id="corr-b")
            await asyncio.sleep(0.05)  # 50ms
            duration = hook.after_operation()
            return duration

        # Run concurrently - op_b finishes first
        results = await asyncio.gather(operation_a(), operation_b())

        duration_a = results[0]
        duration_b = results[1]

        # Each should have its own correct duration (with some tolerance)
        assert duration_a >= 95.0, f"op_a duration {duration_a}ms should be >= 95ms"
        assert duration_b >= 45.0, f"op_b duration {duration_b}ms should be >= 45ms"
        assert duration_a > duration_b, "op_a should take longer than op_b"

    @pytest.mark.asyncio
    async def test_concurrent_operations_isolated_operation_names(self) -> None:
        """Verify concurrent operations have isolated operation names."""
        hook = HookObservability(metrics_sink=None)
        contexts_during_operation: list[dict[str, str | None]] = []

        async def operation_alpha() -> None:
            """Operation with name 'alpha'."""
            hook.before_operation("alpha", labels={"type": "alpha"})
            ctx = hook.get_current_context()
            contexts_during_operation.append(ctx)
            await asyncio.sleep(0.05)
            hook.after_operation()

        async def operation_beta() -> None:
            """Operation with name 'beta'."""
            hook.before_operation("beta", labels={"type": "beta"})
            ctx = hook.get_current_context()
            contexts_during_operation.append(ctx)
            await asyncio.sleep(0.03)
            hook.after_operation()

        await asyncio.gather(operation_alpha(), operation_beta())

        # Each context should have its own operation name
        alpha_contexts = [
            c for c in contexts_during_operation if c.get("operation") == "alpha"
        ]
        beta_contexts = [
            c for c in contexts_during_operation if c.get("operation") == "beta"
        ]

        assert len(alpha_contexts) >= 1, "Should have alpha context"
        assert len(beta_contexts) >= 1, "Should have beta context"

    @pytest.mark.asyncio
    async def test_concurrent_operations_isolated_correlation_ids(self) -> None:
        """Verify concurrent operations have isolated correlation IDs."""
        hook = HookObservability(metrics_sink=None)
        correlation_ids_seen: list[str | None] = []

        async def operation_with_corr_id(corr_id: str) -> None:
            """Operation that records its correlation ID."""
            hook.before_operation("test_op", correlation_id=corr_id)
            ctx = hook.get_current_context()
            correlation_ids_seen.append(ctx.get("correlation_id"))
            await asyncio.sleep(0.02)
            hook.after_operation()

        # Run multiple operations with different correlation IDs
        corr_ids = [f"corr-{i}" for i in range(5)]
        await asyncio.gather(*[operation_with_corr_id(cid) for cid in corr_ids])

        # All unique correlation IDs should be present
        unique_ids = set(correlation_ids_seen)
        for cid in corr_ids:
            assert cid in unique_ids, f"Correlation ID {cid} should be seen"

    @pytest.mark.asyncio
    async def test_many_concurrent_operations_no_interference(self) -> None:
        """Verify many concurrent operations don't interfere.

        This stress test runs many concurrent operations and verifies
        that each gets a reasonable duration without timing corruption.
        """
        hook = HookObservability(metrics_sink=None)
        num_operations = 50
        sleep_base = 0.01  # 10ms base sleep

        async def timed_operation(op_id: int) -> tuple[int, float]:
            """Operation with known sleep duration."""
            sleep_duration = sleep_base + (op_id % 5) * 0.005  # 10-30ms
            hook.before_operation(f"op_{op_id}", correlation_id=str(op_id))
            await asyncio.sleep(sleep_duration)
            duration_ms = hook.after_operation()
            return op_id, duration_ms

        results = await asyncio.gather(
            *[timed_operation(i) for i in range(num_operations)]
        )

        # Verify each operation got a reasonable duration
        for op_id, duration_ms in results:
            expected_min_ms = (
                (sleep_base + (op_id % 5) * 0.005) * 1000 * 0.8
            )  # 80% of expected
            assert duration_ms >= expected_min_ms, (
                f"Op {op_id} duration {duration_ms}ms < expected {expected_min_ms}ms"
            )

    @pytest.mark.asyncio
    async def test_nested_context_managers_isolated(self) -> None:
        """Verify nested context managers maintain isolation."""
        hook = HookObservability(metrics_sink=None)

        async def outer_operation() -> tuple[float, float]:
            """Outer operation containing an inner operation."""
            with hook.operation_context(
                "outer", correlation_id="outer-corr"
            ) as outer_ctx:
                await asyncio.sleep(0.02)  # Some outer work

                # Nested inner operation
                with hook.operation_context(
                    "inner", correlation_id="inner-corr"
                ) as inner_ctx:
                    await asyncio.sleep(0.01)  # Inner work

                await asyncio.sleep(0.02)  # More outer work

            return outer_ctx.duration_ms, inner_ctx.duration_ms

        outer_duration, inner_duration = await outer_operation()

        # Inner should be ~10ms, outer should be ~40ms+ (includes inner)
        assert inner_duration >= 8.0, (
            f"Inner duration {inner_duration}ms should be >= 8ms"
        )
        assert outer_duration >= 35.0, (
            f"Outer duration {outer_duration}ms should be >= 35ms"
        )
        assert outer_duration > inner_duration, "Outer should include inner duration"


class TestNoSharedInstanceVariable:
    """Verify there is no shared _start_time instance variable.

    This test class specifically checks that the hook does NOT use
    instance variables for timing state, which would cause race
    conditions in async code.
    """

    def test_hook_has_no_start_time_instance_variable(self) -> None:
        """Verify hook does not have _start_time as instance variable."""
        hook = HookObservability(metrics_sink=None)

        # Check that _start_time is NOT an instance attribute
        assert not hasattr(hook, "_start_time"), (
            "Hook should NOT have _start_time instance variable - "
            "this would cause race conditions in async code!"
        )

    def test_hook_has_no_operation_name_instance_variable(self) -> None:
        """Verify hook does not have _operation_name as instance variable."""
        hook = HookObservability(metrics_sink=None)

        assert not hasattr(hook, "_operation_name"), (
            "Hook should NOT have _operation_name instance variable"
        )

    def test_hook_has_no_correlation_id_instance_variable(self) -> None:
        """Verify hook does not have _correlation_id as instance variable."""
        hook = HookObservability(metrics_sink=None)

        # Check it doesn't have a timing-related correlation_id attribute
        # (Note: it may have other attributes, but not timing state)
        instance_vars = [
            attr
            for attr in dir(hook)
            if not attr.startswith("__") and attr.startswith("_correlation")
        ]
        # Should not have any _correlation* instance vars used for timing
        assert "_correlation_id" not in vars(hook), (
            "Hook should NOT have _correlation_id instance variable for timing state"
        )


# =============================================================================
# TIMING TESTS
# =============================================================================


class TestTiming:
    """Test before_operation/after_operation timing accuracy."""

    def test_after_operation_returns_duration_in_milliseconds(self) -> None:
        """Verify after_operation returns duration in milliseconds."""
        hook = HookObservability(metrics_sink=None)

        hook.before_operation("test_op")
        time.sleep(0.05)  # 50ms
        duration = hook.after_operation()

        # Should be around 50ms (with some tolerance)
        assert 45.0 <= duration <= 100.0, f"Duration {duration}ms should be ~50ms"

    def test_after_operation_without_before_returns_zero(self) -> None:
        """Verify after_operation returns 0.0 if before_operation not called."""
        hook = HookObservability(metrics_sink=None)

        duration = hook.after_operation()

        assert duration == 0.0

    def test_timing_uses_perf_counter(self) -> None:
        """Verify timing uses high-resolution clock."""
        hook = HookObservability(metrics_sink=None)

        # Measure a very short duration
        hook.before_operation("short_op")
        time.sleep(0.001)  # 1ms
        duration = hook.after_operation()

        # Should measure at least ~1ms (perf_counter is high resolution)
        assert duration >= 0.5, f"Duration {duration}ms should measure sub-ms precision"

    def test_multiple_timing_cycles(self) -> None:
        """Verify multiple timing cycles work correctly."""
        hook = HookObservability(metrics_sink=None)

        durations: list[float] = []

        for i in range(5):
            hook.before_operation(f"cycle_{i}")
            time.sleep(0.01 * (i + 1))  # 10ms, 20ms, 30ms, 40ms, 50ms
            durations.append(hook.after_operation())

        # Each duration should be progressively longer
        for i in range(1, len(durations)):
            assert durations[i] > durations[i - 1], (
                f"Duration {i} ({durations[i]}ms) should be > "
                f"duration {i - 1} ({durations[i - 1]}ms)"
            )


# =============================================================================
# METRICS INTEGRATION TESTS
# =============================================================================


class TestMetricsIntegration:
    """Test metrics sink integration."""

    def test_before_operation_increments_started_counter(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify before_operation increments operation_started_total counter."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        hook.before_operation("test_op", correlation_id="test-corr")

        mock_metrics_sink.increment_counter.assert_called()
        call_args = mock_metrics_sink.increment_counter.call_args
        assert call_args[1]["name"] == "operation_started_total"

    def test_after_operation_observes_duration_histogram(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify after_operation observes operation_duration_seconds histogram."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        hook.before_operation("test_op")
        time.sleep(0.01)
        hook.after_operation()

        # Should have called observe_histogram
        mock_metrics_sink.observe_histogram.assert_called()
        call_args = mock_metrics_sink.observe_histogram.call_args
        assert call_args[1]["name"] == "operation_duration_seconds"

    def test_record_success_increments_completed_counter(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify record_success increments operation_completed_total counter."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        hook.before_operation("test_op")
        hook.record_success()

        # Find the call with operation_completed_total
        calls = mock_metrics_sink.increment_counter.call_args_list
        completed_calls = [
            c for c in calls if c[1].get("name") == "operation_completed_total"
        ]
        assert len(completed_calls) >= 1
        assert completed_calls[0][1]["labels"]["status"] == "success"

    def test_record_failure_increments_failed_counter(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify record_failure increments operation_failed_total counter."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        hook.before_operation("test_op")
        hook.record_failure("TestError")

        calls = mock_metrics_sink.increment_counter.call_args_list
        failed_calls = [
            c for c in calls if c[1].get("name") == "operation_failed_total"
        ]
        assert len(failed_calls) >= 1
        assert failed_calls[0][1]["labels"]["status"] == "failure"
        assert failed_calls[0][1]["labels"]["error_type"] == "TestError"

    def test_no_metrics_sink_operations_still_work(self) -> None:
        """Verify operations work without metrics sink (no-op mode)."""
        hook = HookObservability(metrics_sink=None)

        # Should not raise even without metrics sink
        hook.before_operation("test_op")
        hook.record_success()
        hook.record_failure("SomeError")
        duration = hook.after_operation()

        assert duration >= 0.0

    def test_record_retry_attempt(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify record_retry_attempt increments retry counter."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        hook.before_operation("retry_test")
        hook.record_retry_attempt(attempt_number=2, reason="timeout")

        calls = mock_metrics_sink.increment_counter.call_args_list
        retry_calls = [c for c in calls if c[1].get("name") == "retry_attempt_total"]
        assert len(retry_calls) >= 1
        assert retry_calls[0][1]["labels"]["attempt"] == "2"
        assert retry_calls[0][1]["labels"]["reason"] == "timeout"

    def test_record_circuit_breaker_state_change(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify record_circuit_breaker_state_change increments counter."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        hook.record_circuit_breaker_state_change(
            service_name="database",
            from_state="CLOSED",
            to_state="OPEN",
        )

        calls = mock_metrics_sink.increment_counter.call_args_list
        cb_calls = [
            c for c in calls if c[1].get("name") == "circuit_breaker_state_change_total"
        ]
        assert len(cb_calls) >= 1
        labels = cb_calls[0][1]["labels"]
        assert labels["service"] == "database"
        assert labels["from_state"] == "CLOSED"
        assert labels["to_state"] == "OPEN"

    def test_set_gauge_delegates_to_sink(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify set_gauge delegates to metrics sink."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        hook.set_gauge("active_handlers", value=5.0, labels={"type": "http"})

        mock_metrics_sink.set_gauge.assert_called_once_with(
            name="active_handlers",
            labels={"type": "http"},
            value=5.0,
        )


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestContextManager:
    """Test context manager support."""

    def test_context_manager_basic_usage(self) -> None:
        """Verify context manager provides correct duration."""
        hook = HookObservability(metrics_sink=None)

        with hook.operation_context("test_op") as ctx:
            time.sleep(0.02)

        assert ctx.duration_ms >= 15.0, (
            f"Duration {ctx.duration_ms}ms should be >= 15ms"
        )

    def test_context_manager_records_success_on_normal_exit(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify context manager records success when no exception."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        with hook.operation_context("test_op"):
            pass  # Normal completion

        calls = mock_metrics_sink.increment_counter.call_args_list
        completed_calls = [
            c for c in calls if c[1].get("name") == "operation_completed_total"
        ]
        assert len(completed_calls) >= 1
        assert completed_calls[0][1]["labels"]["status"] == "success"

    def test_context_manager_records_failure_on_exception(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify context manager records failure when exception occurs."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        with pytest.raises(ValueError):
            with hook.operation_context("test_op"):
                raise ValueError("Test error")

        calls = mock_metrics_sink.increment_counter.call_args_list
        failed_calls = [
            c for c in calls if c[1].get("name") == "operation_failed_total"
        ]
        assert len(failed_calls) >= 1
        assert failed_calls[0][1]["labels"]["status"] == "failure"
        assert failed_calls[0][1]["labels"]["error_type"] == "ValueError"

    def test_context_manager_with_correlation_id(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify context manager passes correlation ID correctly."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)
        test_corr_id = str(uuid.uuid4())

        with hook.operation_context("test_op", correlation_id=test_corr_id):
            ctx = hook.get_current_context()
            assert ctx["correlation_id"] == test_corr_id

    def test_context_manager_with_labels(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify context manager passes labels correctly."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        with hook.operation_context("test_op", labels={"handler": "test_handler"}):
            ctx = hook.get_current_context()
            assert ctx.get("handler") == "test_handler"

    @pytest.mark.asyncio
    async def test_context_manager_async_usage(self) -> None:
        """Verify context manager works correctly in async code."""
        hook = HookObservability(metrics_sink=None)

        with hook.operation_context("async_op") as ctx:
            await asyncio.sleep(0.02)

        assert ctx.duration_ms >= 15.0


# =============================================================================
# CONTEXT ACCESS TESTS
# =============================================================================


class TestContextAccess:
    """Test current context access."""

    def test_get_current_context_returns_operation_name(self) -> None:
        """Verify get_current_context includes operation name."""
        hook = HookObservability(metrics_sink=None)

        hook.before_operation("my_operation")
        ctx = hook.get_current_context()

        assert ctx["operation"] == "my_operation"

        hook.after_operation()

    def test_get_current_context_returns_correlation_id(self) -> None:
        """Verify get_current_context includes correlation ID."""
        hook = HookObservability(metrics_sink=None)
        test_corr_id = "test-correlation-123"

        hook.before_operation("test_op", correlation_id=test_corr_id)
        ctx = hook.get_current_context()

        assert ctx["correlation_id"] == test_corr_id

        hook.after_operation()

    def test_get_current_context_returns_labels(self) -> None:
        """Verify get_current_context includes custom labels."""
        hook = HookObservability(metrics_sink=None)

        hook.before_operation("test_op", labels={"handler": "test", "version": "1.0"})
        ctx = hook.get_current_context()

        assert ctx.get("handler") == "test"
        assert ctx.get("version") == "1.0"

        hook.after_operation()

    def test_get_current_context_with_uuid_correlation_id(self) -> None:
        """Verify UUID correlation IDs are converted to strings."""
        hook = HookObservability(metrics_sink=None)
        test_uuid = uuid.uuid4()

        hook.before_operation("test_op", correlation_id=test_uuid)
        ctx = hook.get_current_context()

        assert ctx["correlation_id"] == str(test_uuid)

        hook.after_operation()

    def test_get_current_context_when_no_operation(self) -> None:
        """Verify get_current_context returns None for operation when no active operation.

        Note: correlation_id may persist from previous operations because
        after_operation() intentionally preserves it for error handling scenarios.
        We only verify operation is None here.
        """
        hook = HookObservability(metrics_sink=None)

        # Ensure no active operation by calling after_operation if any
        # This clears the operation name but preserves correlation_id by design
        hook.after_operation()

        ctx = hook.get_current_context()

        # Operation should be None when no active operation
        assert ctx["operation"] is None
        # Note: correlation_id may be set from previous tests - this is expected
        # behavior because after_operation() preserves it for error handling


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_record_success_without_before_operation_no_error(self) -> None:
        """Verify record_success doesn't error if before_operation not called."""
        hook = HookObservability(metrics_sink=None)

        # Should not raise
        hook.record_success()

    def test_record_failure_without_before_operation_no_error(self) -> None:
        """Verify record_failure doesn't error if before_operation not called."""
        hook = HookObservability(metrics_sink=None)

        # Should not raise
        hook.record_failure("SomeError")

    def test_multiple_before_operation_overwrites_context(self) -> None:
        """Verify multiple before_operation calls overwrite context."""
        hook = HookObservability(metrics_sink=None)

        hook.before_operation("first_op", correlation_id="first")
        ctx1 = hook.get_current_context()
        assert ctx1["operation"] == "first_op"

        hook.before_operation("second_op", correlation_id="second")
        ctx2 = hook.get_current_context()
        assert ctx2["operation"] == "second_op"
        assert ctx2["correlation_id"] == "second"

        hook.after_operation()

    def test_after_operation_clears_timing_state(self) -> None:
        """Verify after_operation clears timing state."""
        hook = HookObservability(metrics_sink=None)

        hook.before_operation("test_op")
        hook.after_operation()

        # Subsequent after_operation should return 0 (no start time)
        duration = hook.after_operation()
        assert duration == 0.0

    def test_empty_labels_handled(self) -> None:
        """Verify empty labels dict is handled."""
        hook = HookObservability(metrics_sink=None)

        hook.before_operation("test_op", labels={})
        ctx = hook.get_current_context()

        assert ctx["operation"] == "test_op"

        hook.after_operation()

    def test_none_labels_handled(self) -> None:
        """Verify None labels is handled."""
        hook = HookObservability(metrics_sink=None)

        hook.before_operation("test_op", labels=None)
        ctx = hook.get_current_context()

        assert ctx["operation"] == "test_op"

        hook.after_operation()


# =============================================================================
# HIGH-CARDINALITY LABEL FILTERING TESTS
# =============================================================================


class TestHighCardinalityLabelFiltering:
    """Test that high-cardinality labels are filtered but metrics are still recorded.

    These tests verify the fix for PR #169 issue where metrics could be
    dropped when correlation_id or other high-cardinality labels were present.
    The correct behavior is to filter out high-cardinality labels while still
    recording the metric with remaining valid labels.
    """

    def test_correlation_id_in_labels_filtered_but_metric_recorded(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify correlation_id in labels is filtered but metric is still recorded."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        # Pass correlation_id as part of labels dict (not the separate parameter)
        hook.before_operation(
            "test_op",
            labels={"correlation_id": "should-be-filtered", "handler": "test_handler"},
        )
        hook.record_success()

        # Metric should be recorded
        calls = mock_metrics_sink.increment_counter.call_args_list
        completed_calls = [
            c for c in calls if c[1].get("name") == "operation_completed_total"
        ]
        assert len(completed_calls) >= 1, (
            "Metric should be recorded even with filtered labels"
        )

        # Labels should NOT contain correlation_id
        labels = completed_calls[0][1]["labels"]
        assert "correlation_id" not in labels, "correlation_id should be filtered"

        # Labels SHOULD contain handler (non-high-cardinality)
        assert labels.get("handler") == "test_handler", (
            "Valid labels should be preserved"
        )

        hook.after_operation()

    def test_multiple_high_cardinality_labels_all_filtered(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify all high-cardinality labels are filtered, others preserved."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        # Pass multiple high-cardinality labels
        hook.before_operation(
            "test_op",
            labels={
                "correlation_id": "corr-123",
                "request_id": "req-456",
                "trace_id": "trace-789",
                "session_id": "sess-abc",
                "user_id": "user-def",
                "handler": "valid_handler",  # This should NOT be filtered
                "status_code": "200",  # This should NOT be filtered
            },
        )
        hook.record_success()

        calls = mock_metrics_sink.increment_counter.call_args_list
        completed_calls = [
            c for c in calls if c[1].get("name") == "operation_completed_total"
        ]
        assert len(completed_calls) >= 1

        labels = completed_calls[0][1]["labels"]

        # All high-cardinality labels should be filtered
        assert "correlation_id" not in labels
        assert "request_id" not in labels
        assert "trace_id" not in labels
        assert "session_id" not in labels
        assert "user_id" not in labels

        # Valid labels should be preserved
        assert labels.get("handler") == "valid_handler"
        assert labels.get("status_code") == "200"
        assert labels.get("operation") == "test_op"

        hook.after_operation()

    def test_only_high_cardinality_labels_still_records_metric(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify metric is recorded even if ALL labels are high-cardinality."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        # Pass only high-cardinality labels - all will be filtered
        hook.before_operation(
            "test_op",
            labels={
                "correlation_id": "corr-123",
                "request_id": "req-456",
            },
        )
        hook.record_success()

        calls = mock_metrics_sink.increment_counter.call_args_list
        completed_calls = [
            c for c in calls if c[1].get("name") == "operation_completed_total"
        ]

        # Metric MUST still be recorded (this is the key fix)
        assert len(completed_calls) >= 1, (
            "Metric MUST be recorded even when all labels are filtered"
        )

        labels = completed_calls[0][1]["labels"]

        # Should have at least the operation label
        assert labels.get("operation") == "test_op"
        # Should NOT have high-cardinality labels
        assert "correlation_id" not in labels
        assert "request_id" not in labels

        hook.after_operation()

    def test_context_manager_with_high_cardinality_labels(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify context manager filters high-cardinality labels correctly."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        with hook.operation_context(
            "test_op",
            labels={"correlation_id": "ctx-123", "method": "GET"},
        ):
            pass

        calls = mock_metrics_sink.increment_counter.call_args_list
        completed_calls = [
            c for c in calls if c[1].get("name") == "operation_completed_total"
        ]
        assert len(completed_calls) >= 1

        labels = completed_calls[0][1]["labels"]
        assert "correlation_id" not in labels
        assert labels.get("method") == "GET"


# =============================================================================
# BUFFER GAUGE VALIDATION TESTS
# =============================================================================


class TestBufferGaugeValidation:
    """Test buffer gauge non-negative value enforcement.

    These tests verify the fix for PR #169 nitpick about enforcing
    non-negative buffer metrics with proper logging.
    """

    def test_negative_buffer_gauge_clamped_to_zero(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify negative buffer gauge values are clamped to 0.0."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        hook.set_buffer_gauge(
            "queue_depth",
            value=-5.0,  # Negative value
            labels={"queue": "test"},
        )

        mock_metrics_sink.set_gauge.assert_called_once_with(
            name="queue_depth",
            labels={"queue": "test"},
            value=0.0,  # Should be clamped to 0.0
        )

    def test_zero_buffer_gauge_allowed(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify zero buffer gauge values are passed through unchanged."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        hook.set_buffer_gauge(
            "queue_depth",
            value=0.0,
            labels={"queue": "test"},
        )

        mock_metrics_sink.set_gauge.assert_called_once_with(
            name="queue_depth",
            labels={"queue": "test"},
            value=0.0,
        )

    def test_positive_buffer_gauge_allowed(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify positive buffer gauge values are passed through unchanged."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        hook.set_buffer_gauge(
            "queue_depth",
            value=42.0,
            labels={"queue": "test"},
        )

        mock_metrics_sink.set_gauge.assert_called_once_with(
            name="queue_depth",
            labels={"queue": "test"},
            value=42.0,
        )

    def test_negative_buffer_gauge_logs_warning(
        self,
        mock_metrics_sink: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify negative buffer gauge values trigger a warning log."""
        import logging

        hook = HookObservability(metrics_sink=mock_metrics_sink)

        with caplog.at_level(logging.WARNING):
            hook.set_buffer_gauge(
                "buffer_size",
                value=-10.0,
                labels={"buffer": "write"},
            )

        # Should have logged a warning about negative value
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_logs) >= 1, "Should log warning for negative buffer gauge"
        assert "negative" in warning_logs[0].message.lower()

    def test_regular_gauge_allows_negative(
        self,
        mock_metrics_sink: MagicMock,
    ) -> None:
        """Verify regular set_gauge allows negative values (e.g., temperature)."""
        hook = HookObservability(metrics_sink=mock_metrics_sink)

        hook.set_gauge(
            "temperature_delta",
            value=-5.0,  # Negative is valid for non-buffer gauges
            labels={"sensor": "cpu"},
        )

        mock_metrics_sink.set_gauge.assert_called_once_with(
            name="temperature_delta",
            labels={"sensor": "cpu"},
            value=-5.0,  # Should NOT be clamped
        )


# =============================================================================
# HIGH-CARDINALITY LABEL FILTERING LOGGING TESTS
# =============================================================================


class TestHighCardinalityFilteringLogging:
    """Test that high-cardinality label filtering is logged for debugging."""

    def test_filtering_logs_debug_message(
        self,
        mock_metrics_sink: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify filtering high-cardinality labels produces debug log."""
        import logging

        hook = HookObservability(metrics_sink=mock_metrics_sink)

        with caplog.at_level(logging.DEBUG):
            hook.before_operation(
                "test_op",
                labels={"correlation_id": "debug-test-123", "handler": "test"},
            )
            hook.record_success()

        # Should have logged debug message about filtering
        debug_logs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        filtering_logs = [r for r in debug_logs if "filtered" in r.message.lower()]
        assert len(filtering_logs) >= 1, (
            "Should log debug message when filtering labels"
        )

        hook.after_operation()

    def test_no_filtering_log_when_no_high_cardinality_labels(
        self,
        mock_metrics_sink: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify no filtering log when no high-cardinality labels present."""
        import logging

        hook = HookObservability(metrics_sink=mock_metrics_sink)

        with caplog.at_level(logging.DEBUG):
            hook.before_operation(
                "test_op",
                labels={"handler": "test", "method": "GET"},  # No high-cardinality
            )
            hook.record_success()

        # Should NOT have logged about filtering (no high-cardinality labels)
        debug_logs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        filtering_logs = [
            r
            for r in debug_logs
            if "filtered" in r.message.lower()
            and "high-cardinality" in r.message.lower()
        ]
        assert len(filtering_logs) == 0, (
            "Should not log filtering when no high-cardinality labels"
        )

        hook.after_operation()
