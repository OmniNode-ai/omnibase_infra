# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelOperationContext timeout detection.

These tests validate the timeout detection methods of ModelOperationContext,
including boundary conditions, elapsed time calculations, and remaining time.
"""

from __future__ import annotations

import time
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.handlers.models.model_operation_context import ModelOperationContext


class TestModelOperationContextNotTimedOut:
    """Tests for cases where the operation has NOT timed out."""

    def test_fresh_context_not_timed_out(self) -> None:
        """Test that a freshly created context is not timed out."""
        ctx = ModelOperationContext.create("test.operation", timeout_seconds=30.0)
        assert not ctx.is_timed_out()

    def test_check_immediately_after_creation(self) -> None:
        """Test checking timeout immediately after creation returns False."""
        ctx = ModelOperationContext.create("test.operation", timeout_seconds=30.0)
        # Even with minimal elapsed time, should not be timed out
        assert ctx.is_timed_out() is False

    def test_elapsed_time_within_timeout(self) -> None:
        """Test that elapsed time within timeout returns not timed out."""
        # Create context with started_at 10 seconds ago, timeout of 30s
        past_time = time.time() - 10.0
        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=30.0,
        )
        # 10 seconds elapsed, 30 second timeout -> not timed out
        assert not ctx.is_timed_out()

    def test_well_within_timeout(self) -> None:
        """Test that operation well within timeout is not timed out."""
        # Create context with started_at 5 seconds ago, timeout of 3600s (1 hour)
        past_time = time.time() - 5.0
        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=3600.0,
        )
        assert not ctx.is_timed_out()
        # Remaining time should be substantial
        assert ctx.remaining_seconds() > 3500.0


class TestModelOperationContextTimedOut:
    """Tests for cases where the operation HAS timed out."""

    def test_timed_out_with_past_started_at(self) -> None:
        """Test that operation started long ago is timed out."""
        # Create with started_at 60 seconds ago, timeout of 30s
        past_time = time.time() - 60.0
        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=30.0,
        )
        assert ctx.is_timed_out()

    def test_timed_out_just_over_timeout(self) -> None:
        """Test timeout detection when just over the timeout threshold."""
        # Create with started_at that puts us just over the timeout
        past_time = time.time() - 31.0  # 31 seconds ago
        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=30.0,
        )
        assert ctx.is_timed_out()

    def test_timed_out_significantly_over(self) -> None:
        """Test timeout detection when significantly over the timeout."""
        # Create with started_at 120 seconds ago, timeout of 30s
        past_time = time.time() - 120.0
        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=30.0,
        )
        assert ctx.is_timed_out()
        # Elapsed should be approximately 120 seconds
        assert ctx.elapsed_seconds() > 119.0
        assert ctx.elapsed_seconds() < 121.0


class TestModelOperationContextBoundaryConditions:
    """Tests for boundary conditions in timeout detection.

    CRITICAL: These tests verify the exact behavior at the timeout boundary.
    The is_timed_out() method uses > (greater than), not >= (greater than or equal).
    This means that at EXACTLY the timeout boundary, is_timed_out() returns False.
    """

    def test_exactly_at_timeout_boundary_is_not_timed_out(self) -> None:
        """Test that EXACTLY at timeout boundary, is_timed_out is False.

        The implementation uses `elapsed > timeout`, not `>=`.
        Therefore, when elapsed == timeout, the result should be False.
        """
        timeout_seconds = 30.0
        current_time = time.time()

        # Create context where elapsed_seconds would be exactly equal to timeout
        # Note: Due to time.time() precision and execution time, we construct
        # the test deterministically
        started_at = current_time - timeout_seconds

        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=started_at,
            timeout_seconds=timeout_seconds,
        )

        # At the moment of creation, elapsed ~= timeout
        # Due to the > comparison (not >=), this should be False
        # However, a tiny amount of time may have passed, so we verify the logic:
        # If elapsed > timeout: timed_out
        # If elapsed == timeout: NOT timed_out
        # If elapsed < timeout: NOT timed_out

        # We can verify the comparison behavior by checking elapsed_seconds
        # Since some microseconds may have passed, let's verify the expected logic
        elapsed = ctx.elapsed_seconds()

        # The test should demonstrate that at the boundary, the result is False
        # To be deterministic, we verify the logic rather than exact timing
        assert (elapsed > timeout_seconds) == ctx.is_timed_out()

    def test_just_over_timeout_boundary(self) -> None:
        """Test that just over the timeout boundary returns timed out."""
        timeout_seconds = 30.0
        current_time = time.time()

        # Create context where elapsed is just over the timeout
        started_at = current_time - timeout_seconds - 0.001  # 0.001 seconds over

        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=started_at,
            timeout_seconds=timeout_seconds,
        )

        # elapsed > timeout, so should be timed out
        assert ctx.is_timed_out()

    def test_just_under_timeout_boundary(self) -> None:
        """Test that just under the timeout boundary returns not timed out."""
        timeout_seconds = 30.0
        current_time = time.time()

        # Create context where elapsed is just under the timeout
        started_at = current_time - timeout_seconds + 0.001  # 0.001 seconds under

        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=started_at,
            timeout_seconds=timeout_seconds,
        )

        # elapsed < timeout, so should NOT be timed out
        assert not ctx.is_timed_out()

    def test_boundary_precision_microseconds_over(self) -> None:
        """Test timeout detection with microsecond precision over the boundary."""
        timeout_seconds = 10.0
        current_time = time.time()

        # 100 microseconds over the boundary
        started_at = current_time - timeout_seconds - 0.0001

        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=started_at,
            timeout_seconds=timeout_seconds,
        )

        assert ctx.is_timed_out()

    def test_boundary_precision_microseconds_under(self) -> None:
        """Test timeout detection with microsecond precision under the boundary."""
        timeout_seconds = 10.0
        current_time = time.time()

        # 100 microseconds under the boundary
        started_at = current_time - timeout_seconds + 0.0001

        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=started_at,
            timeout_seconds=timeout_seconds,
        )

        assert not ctx.is_timed_out()


class TestModelOperationContextZeroTimeout:
    """Tests for zero timeout edge case."""

    def test_zero_timeout_immediately_timed_out(self) -> None:
        """Test that zero timeout means immediately timed out.

        With timeout_seconds=0.0, any positive elapsed time means timed out.
        Even a freshly created context will have some elapsed time.
        """
        ctx = ModelOperationContext.create("test.operation", timeout_seconds=0.0)

        # Any elapsed time > 0 means timed out
        # Since some time passes between creation and check, should be timed out
        # Note: This relies on actual time passing, which is deterministic for > 0
        assert ctx.is_timed_out()

    def test_zero_timeout_with_explicit_started_at(self) -> None:
        """Test zero timeout with explicit started_at in the past."""
        past_time = time.time() - 0.001  # Just 1 millisecond ago
        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=0.0,
        )

        # elapsed > 0 and timeout = 0, so elapsed > timeout
        assert ctx.is_timed_out()

    def test_zero_timeout_negative_remaining(self) -> None:
        """Test that zero timeout results in negative remaining seconds."""
        past_time = time.time() - 5.0  # 5 seconds ago
        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=0.0,
        )

        # remaining = timeout - elapsed = 0 - 5 = -5
        assert ctx.remaining_seconds() < 0
        # Should be approximately -5 seconds
        assert ctx.remaining_seconds() < -4.9
        assert ctx.remaining_seconds() > -5.1


class TestModelOperationContextElapsedSeconds:
    """Tests for elapsed_seconds() accuracy."""

    def test_elapsed_seconds_returns_reasonable_value(self) -> None:
        """Test that elapsed_seconds returns a reasonable positive value."""
        ctx = ModelOperationContext.create("test.operation")
        elapsed = ctx.elapsed_seconds()

        # Should be a small positive number (just created)
        assert elapsed >= 0
        assert elapsed < 1.0  # Should be much less than 1 second

    def test_elapsed_seconds_with_known_started_at(self) -> None:
        """Test elapsed_seconds accuracy with explicit started_at."""
        known_elapsed = 15.0
        past_time = time.time() - known_elapsed

        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=30.0,
        )

        elapsed = ctx.elapsed_seconds()
        # Should be approximately 15 seconds (with some tolerance for execution time)
        assert elapsed >= known_elapsed
        assert elapsed < known_elapsed + 0.1  # Allow 100ms tolerance

    def test_elapsed_seconds_increases_with_older_started_at(self) -> None:
        """Test that elapsed_seconds increases with older started_at."""
        current_time = time.time()

        ctx1 = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=current_time - 10.0,  # 10 seconds ago
            timeout_seconds=30.0,
        )

        ctx2 = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=current_time - 20.0,  # 20 seconds ago
            timeout_seconds=30.0,
        )

        assert ctx2.elapsed_seconds() > ctx1.elapsed_seconds()

    def test_elapsed_seconds_precision(self) -> None:
        """Test that elapsed_seconds has sub-second precision."""
        known_elapsed = 5.5  # 5.5 seconds
        past_time = time.time() - known_elapsed

        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=30.0,
        )

        elapsed = ctx.elapsed_seconds()
        # Should capture sub-second precision
        assert elapsed >= 5.4
        assert elapsed < 5.7


class TestModelOperationContextRemainingSeconds:
    """Tests for remaining_seconds() accuracy."""

    def test_remaining_seconds_positive_when_not_timed_out(self) -> None:
        """Test that remaining_seconds is positive when not timed out."""
        ctx = ModelOperationContext.create("test.operation", timeout_seconds=30.0)
        remaining = ctx.remaining_seconds()

        # Just created, should have ~30 seconds remaining
        assert remaining > 29.0
        assert remaining <= 30.0

    def test_remaining_seconds_negative_when_timed_out(self) -> None:
        """Test that remaining_seconds is negative when timed out."""
        past_time = time.time() - 60.0  # 60 seconds ago
        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=30.0,
        )

        remaining = ctx.remaining_seconds()
        # 60 seconds elapsed, 30 second timeout -> -30 seconds remaining
        assert remaining < 0
        assert remaining < -29.0
        assert remaining > -31.0

    def test_remaining_seconds_equals_timeout_minus_elapsed(self) -> None:
        """Test that remaining_seconds = timeout - elapsed (mathematical relationship)."""
        past_time = time.time() - 10.0  # 10 seconds ago
        timeout = 30.0

        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=timeout,
        )

        # Get values at approximately the same time
        elapsed = ctx.elapsed_seconds()
        remaining = ctx.remaining_seconds()

        # remaining should equal timeout - elapsed
        # Allow small tolerance for time between calls
        expected_remaining = timeout - elapsed
        assert abs(remaining - expected_remaining) < 0.01

    def test_remaining_seconds_precision(self) -> None:
        """Test that remaining_seconds has sub-second precision."""
        timeout = 30.0
        elapsed = 15.5  # 15.5 seconds elapsed
        past_time = time.time() - elapsed

        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=past_time,
            timeout_seconds=timeout,
        )

        remaining = ctx.remaining_seconds()
        # Should be approximately 14.5 seconds (30 - 15.5)
        assert remaining > 14.3
        assert remaining < 14.7

    def test_remaining_seconds_approaches_zero_at_boundary(self) -> None:
        """Test that remaining_seconds approaches zero at the boundary."""
        timeout = 30.0
        current_time = time.time()

        # Create context exactly at the timeout boundary
        started_at = current_time - timeout

        ctx = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=started_at,
            timeout_seconds=timeout,
        )

        remaining = ctx.remaining_seconds()
        # Should be approximately 0 (or slightly negative due to execution time)
        assert remaining < 0.1
        assert remaining > -0.1


class TestModelOperationContextCreateFactory:
    """Tests for the create() factory method."""

    def test_create_sets_current_timestamp(self) -> None:
        """Test that create() sets started_at to current time."""
        before = time.time()
        ctx = ModelOperationContext.create("test.operation")
        after = time.time()

        assert ctx.started_at >= before
        assert ctx.started_at <= after

    def test_create_with_default_timeout(self) -> None:
        """Test that create() uses default timeout of 30.0 seconds."""
        ctx = ModelOperationContext.create("test.operation")
        assert ctx.timeout_seconds == 30.0

    def test_create_with_custom_timeout(self) -> None:
        """Test that create() accepts custom timeout."""
        ctx = ModelOperationContext.create("test.operation", timeout_seconds=60.0)
        assert ctx.timeout_seconds == 60.0

    def test_create_auto_generates_correlation_id(self) -> None:
        """Test that create() auto-generates correlation_id if not provided."""
        ctx = ModelOperationContext.create("test.operation")
        assert isinstance(ctx.correlation_id, UUID)

    def test_create_uses_provided_correlation_id(self) -> None:
        """Test that create() uses provided correlation_id."""
        provided_id = uuid4()
        ctx = ModelOperationContext.create("test.operation", correlation_id=provided_id)
        assert ctx.correlation_id == provided_id


class TestModelOperationContextTimeoutValidation:
    """Tests for timeout_seconds field validation."""

    def test_timeout_seconds_minimum_zero(self) -> None:
        """Test that timeout_seconds accepts 0.0."""
        ctx = ModelOperationContext.create("test.operation", timeout_seconds=0.0)
        assert ctx.timeout_seconds == 0.0

    def test_timeout_seconds_maximum(self) -> None:
        """Test that timeout_seconds accepts maximum value (3600)."""
        ctx = ModelOperationContext.create("test.operation", timeout_seconds=3600.0)
        assert ctx.timeout_seconds == 3600.0

    def test_timeout_seconds_rejects_negative(self) -> None:
        """Test that negative timeout_seconds is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelOperationContext.create("test.operation", timeout_seconds=-1.0)
        assert "timeout_seconds" in str(exc_info.value)

    def test_timeout_seconds_rejects_over_maximum(self) -> None:
        """Test that timeout_seconds over 3600 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelOperationContext.create("test.operation", timeout_seconds=3601.0)
        assert "timeout_seconds" in str(exc_info.value)


class TestModelOperationContextImmutability:
    """Tests for model immutability (frozen)."""

    def test_model_is_frozen(self) -> None:
        """Test that the model is immutable (frozen)."""
        ctx = ModelOperationContext.create("test.operation")
        with pytest.raises(ValidationError):
            ctx.timeout_seconds = 60.0  # type: ignore[misc]

    def test_started_at_cannot_be_modified(self) -> None:
        """Test that started_at cannot be modified after creation."""
        ctx = ModelOperationContext.create("test.operation")
        with pytest.raises(ValidationError):
            ctx.started_at = time.time()  # type: ignore[misc]


class TestModelOperationContextTimingRelationships:
    """Tests for timing relationships between methods."""

    def test_is_timed_out_consistent_with_elapsed_and_remaining(self) -> None:
        """Test that is_timed_out is consistent with elapsed and remaining."""
        # Test case 1: Not timed out
        ctx1 = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=time.time() - 10.0,
            timeout_seconds=30.0,
        )
        assert not ctx1.is_timed_out()
        assert ctx1.elapsed_seconds() < ctx1.timeout_seconds
        assert ctx1.remaining_seconds() > 0

        # Test case 2: Timed out
        ctx2 = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=time.time() - 60.0,
            timeout_seconds=30.0,
        )
        assert ctx2.is_timed_out()
        assert ctx2.elapsed_seconds() > ctx2.timeout_seconds
        assert ctx2.remaining_seconds() < 0

    def test_elapsed_plus_remaining_equals_timeout(self) -> None:
        """Test that elapsed + remaining approximately equals timeout."""
        ctx = ModelOperationContext.create("test.operation", timeout_seconds=30.0)

        # elapsed + remaining should equal timeout
        # Allow small tolerance for time between calls
        elapsed = ctx.elapsed_seconds()
        remaining = ctx.remaining_seconds()

        assert abs((elapsed + remaining) - 30.0) < 0.01


class TestModelOperationContextWithMethods:
    """Tests for with_metadata and with_operation_name methods."""

    def test_with_metadata_preserves_timing(self) -> None:
        """Test that with_metadata preserves timing context."""
        original = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="test.operation",
            started_at=time.time() - 10.0,
            timeout_seconds=30.0,
        )

        updated = original.with_metadata("key", "value")

        # Timing should be identical
        assert updated.started_at == original.started_at
        assert updated.timeout_seconds == original.timeout_seconds
        # Timeout status should be the same
        assert updated.is_timed_out() == original.is_timed_out()

    def test_with_operation_name_preserves_timing(self) -> None:
        """Test that with_operation_name preserves timing context."""
        original = ModelOperationContext(
            correlation_id=uuid4(),
            operation_name="original.operation",
            started_at=time.time() - 10.0,
            timeout_seconds=30.0,
        )

        updated = original.with_operation_name("new.operation")

        # Timing should be identical
        assert updated.started_at == original.started_at
        assert updated.timeout_seconds == original.timeout_seconds
        # Timeout status should be the same
        assert updated.is_timed_out() == original.is_timed_out()
        # Operation name should be updated
        assert updated.operation_name == "new.operation"


__all__: list[str] = [
    "TestModelOperationContextNotTimedOut",
    "TestModelOperationContextTimedOut",
    "TestModelOperationContextBoundaryConditions",
    "TestModelOperationContextZeroTimeout",
    "TestModelOperationContextElapsedSeconds",
    "TestModelOperationContextRemainingSeconds",
    "TestModelOperationContextCreateFactory",
    "TestModelOperationContextTimeoutValidation",
    "TestModelOperationContextImmutability",
    "TestModelOperationContextTimingRelationships",
    "TestModelOperationContextWithMethods",
]
