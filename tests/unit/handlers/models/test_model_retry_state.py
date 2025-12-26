# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelRetryState.next_attempt() method.

This module provides comprehensive tests for the retry state model's
backoff calculation, max delay capping, timestamp handling, and
error message propagation.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from omnibase_infra.handlers.models.model_retry_state import ModelRetryState


class TestModelRetryStateNextAttempt:
    """Test suite for ModelRetryState.next_attempt() method."""

    # =========================================================================
    # Basic Backoff Calculation Tests
    # =========================================================================

    def test_basic_backoff_calculation_doubles_delay(self) -> None:
        """Test that delay doubles with backoff_multiplier=2.0.

        Given initial delay=1.0 and multiplier=2.0, the next delay
        should be 1.0 * 2.0 = 2.0.
        """
        state = ModelRetryState(
            delay_seconds=1.0,
            backoff_multiplier=2.0,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        assert next_state.delay_seconds == 2.0

    def test_basic_backoff_calculation_triples_delay(self) -> None:
        """Test that delay triples with backoff_multiplier=3.0.

        Given initial delay=1.0 and multiplier=3.0, the next delay
        should be 1.0 * 3.0 = 3.0.
        """
        state = ModelRetryState(
            delay_seconds=1.0,
            backoff_multiplier=3.0,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        assert next_state.delay_seconds == 3.0

    def test_exponential_growth_over_multiple_attempts(self) -> None:
        """Test exponential backoff growth over successive attempts.

        With delay=1.0 and multiplier=2.0, delays should be:
        - Attempt 0: 1.0 (initial)
        - Attempt 1: 2.0 (1.0 * 2.0)
        - Attempt 2: 4.0 (2.0 * 2.0)
        - Attempt 3: 8.0 (4.0 * 2.0)
        """
        state = ModelRetryState(
            delay_seconds=1.0,
            backoff_multiplier=2.0,
            max_attempts=10,
        )

        expected_delays = [2.0, 4.0, 8.0, 16.0, 32.0]

        for i, expected_delay in enumerate(expected_delays):
            state = state.next_attempt(timestamp=1000.0 + i)
            assert state.delay_seconds == expected_delay, (
                f"Attempt {i + 1}: expected delay={expected_delay}, "
                f"got delay={state.delay_seconds}"
            )

    def test_backoff_with_fractional_multiplier(self) -> None:
        """Test backoff calculation with fractional multiplier (1.5).

        Given delay=2.0 and multiplier=1.5, the next delay should be
        2.0 * 1.5 = 3.0.
        """
        state = ModelRetryState(
            delay_seconds=2.0,
            backoff_multiplier=1.5,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        assert next_state.delay_seconds == 3.0

    def test_backoff_with_minimum_multiplier(self) -> None:
        """Test backoff with minimum multiplier (1.0) keeps delay constant.

        With multiplier=1.0, the delay should remain unchanged.
        """
        state = ModelRetryState(
            delay_seconds=5.0,
            backoff_multiplier=1.0,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        assert next_state.delay_seconds == 5.0

    # =========================================================================
    # Max Delay Capping Tests (Critical Edge Cases)
    # =========================================================================

    def test_max_delay_capping_at_default_limit(self) -> None:
        """Test that delay caps at default max_delay_seconds (300.0).

        Given delay=200.0 and multiplier=2.0, calculated delay would be
        400.0, which should cap at the default 300.0.
        """
        state = ModelRetryState(
            delay_seconds=200.0,
            backoff_multiplier=2.0,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        # 200.0 * 2.0 = 400.0, should cap at 300.0
        assert next_state.delay_seconds == 300.0

    def test_max_delay_capping_larger_calculated_value(self) -> None:
        """Test capping when calculated delay significantly exceeds max.

        Given delay=150.0 and multiplier=3.0, calculated delay would be
        450.0, which should cap at the default 300.0.
        """
        state = ModelRetryState(
            delay_seconds=150.0,
            backoff_multiplier=3.0,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        # 150.0 * 3.0 = 450.0, should cap at 300.0
        assert next_state.delay_seconds == 300.0

    def test_max_delay_capping_at_custom_limit(self) -> None:
        """Test capping at custom max_delay_seconds value.

        Given delay=60.0, multiplier=2.0, and custom max=100.0,
        calculated delay would be 120.0, which should cap at 100.0.
        """
        state = ModelRetryState(
            delay_seconds=60.0,
            backoff_multiplier=2.0,
        )

        next_state = state.next_attempt(timestamp=1000.0, max_delay_seconds=100.0)

        # 60.0 * 2.0 = 120.0, should cap at 100.0
        assert next_state.delay_seconds == 100.0

    def test_max_delay_capping_with_very_low_custom_limit(self) -> None:
        """Test capping at very low custom max_delay_seconds.

        Even with low initial delay, custom max should cap the result.
        """
        state = ModelRetryState(
            delay_seconds=5.0,
            backoff_multiplier=2.0,
        )

        next_state = state.next_attempt(timestamp=1000.0, max_delay_seconds=8.0)

        # 5.0 * 2.0 = 10.0, should cap at 8.0
        assert next_state.delay_seconds == 8.0

    def test_delay_below_max_not_capped(self) -> None:
        """Test that delay below max is not affected by capping.

        When calculated delay is less than max_delay_seconds, it should
        be used as-is without modification.
        """
        state = ModelRetryState(
            delay_seconds=10.0,
            backoff_multiplier=2.0,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        # 10.0 * 2.0 = 20.0, below 300.0 so no capping
        assert next_state.delay_seconds == 20.0

    # =========================================================================
    # Boundary Condition Tests
    # =========================================================================

    def test_delay_exactly_at_cap_stays_at_cap(self) -> None:
        """Test that delay exactly at cap stays at cap after next_attempt.

        When delay=300.0 (exactly at default cap) and multiplier=2.0,
        calculated delay would be 600.0, which caps at 300.0.
        """
        state = ModelRetryState(
            delay_seconds=300.0,
            backoff_multiplier=2.0,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        # 300.0 * 2.0 = 600.0, should cap at 300.0
        assert next_state.delay_seconds == 300.0

    def test_delay_just_below_cap_after_calculation(self) -> None:
        """Test delay that is just below cap after backoff calculation.

        Verify that 299.9 * 2.0 = 599.8 correctly caps at 300.0.
        """
        state = ModelRetryState(
            delay_seconds=299.9,
            backoff_multiplier=2.0,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        # 299.9 * 2.0 = 599.8, should cap at 300.0
        assert next_state.delay_seconds == 300.0

    def test_delay_exactly_equals_calculated_when_no_cap_needed(self) -> None:
        """Test that calculated delay is exact when below cap.

        When delay * multiplier < max_delay_seconds, result should be
        exactly delay * multiplier with no rounding.
        """
        state = ModelRetryState(
            delay_seconds=7.5,
            backoff_multiplier=2.0,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        # 7.5 * 2.0 = 15.0 exactly
        assert next_state.delay_seconds == 15.0

    def test_small_delay_with_large_multiplier_caps_correctly(self) -> None:
        """Test small delay with maximum multiplier (10.0) caps correctly.

        Even with small initial delay, large multiplier over many attempts
        will eventually hit the cap.
        """
        state = ModelRetryState(
            delay_seconds=50.0,
            backoff_multiplier=10.0,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        # 50.0 * 10.0 = 500.0, should cap at 300.0
        assert next_state.delay_seconds == 300.0

    def test_zero_delay_stays_zero(self) -> None:
        """Test that zero delay stays at zero (edge case).

        While delay_seconds has ge=0.0 constraint, if 0.0 is used,
        backoff should keep it at 0.0 (0.0 * anything = 0.0).
        """
        state = ModelRetryState(
            delay_seconds=0.0,
            backoff_multiplier=2.0,
        )

        next_state = state.next_attempt(timestamp=1000.0)

        # 0.0 * 2.0 = 0.0
        assert next_state.delay_seconds == 0.0

    # =========================================================================
    # Timestamp Handling Tests
    # =========================================================================

    def test_explicit_timestamp_used_when_provided(self) -> None:
        """Test that explicit timestamp is used when provided.

        When timestamp parameter is passed, it should be stored
        in last_attempt_at without modification.
        """
        state = ModelRetryState()
        explicit_timestamp = 1234567890.123

        next_state = state.next_attempt(timestamp=explicit_timestamp)

        assert next_state.last_attempt_at == explicit_timestamp

    def test_current_time_used_when_no_timestamp_provided(self) -> None:
        """Test that current time is used when no timestamp provided.

        When timestamp is not passed (None), time.time() should be used.
        We verify this by checking the timestamp is reasonable (within
        a small window around current time).
        """
        state = ModelRetryState()

        before = time.time()
        next_state = state.next_attempt()
        after = time.time()

        assert next_state.last_attempt_at is not None
        assert before <= next_state.last_attempt_at <= after

    def test_none_timestamp_triggers_time_call(self) -> None:
        """Test that None timestamp explicitly triggers time.time().

        Verify by patching time.time() that it's called when timestamp
        is not provided.
        """
        state = ModelRetryState()
        mock_time = 9999999999.999

        with patch("time.time", return_value=mock_time):
            next_state = state.next_attempt(timestamp=None)

        assert next_state.last_attempt_at == mock_time

    def test_zero_timestamp_is_valid(self) -> None:
        """Test that zero is a valid explicit timestamp.

        Timestamp of 0.0 (Unix epoch) should be accepted as valid
        and stored without triggering time.time() fallback.
        """
        state = ModelRetryState()

        next_state = state.next_attempt(timestamp=0.0)

        assert next_state.last_attempt_at == 0.0

    def test_negative_timestamp_accepted(self) -> None:
        """Test that negative timestamp is accepted (pre-epoch).

        While unusual, negative timestamps (dates before Unix epoch)
        should be accepted without validation errors.
        """
        state = ModelRetryState()

        next_state = state.next_attempt(timestamp=-86400.0)

        assert next_state.last_attempt_at == -86400.0

    # =========================================================================
    # Error Message Propagation Tests
    # =========================================================================

    def test_error_message_stored_when_provided(self) -> None:
        """Test that error message is stored in last_error when provided."""
        state = ModelRetryState()
        error_msg = "Connection timeout after 30 seconds"

        next_state = state.next_attempt(error_message=error_msg, timestamp=1000.0)

        assert next_state.last_error == error_msg

    def test_last_error_is_none_when_no_message_provided(self) -> None:
        """Test that last_error is None when no error message provided."""
        state = ModelRetryState()

        next_state = state.next_attempt(timestamp=1000.0)

        assert next_state.last_error is None

    def test_empty_string_error_message_is_valid(self) -> None:
        """Test that empty string is accepted as error message."""
        state = ModelRetryState()

        next_state = state.next_attempt(error_message="", timestamp=1000.0)

        assert next_state.last_error == ""

    def test_error_message_replaces_previous_error(self) -> None:
        """Test that new error message replaces previous one.

        Each call to next_attempt() creates a new state with the new
        error message, regardless of what the previous error was.
        """
        state = ModelRetryState()
        first_error = "First error: connection refused"
        second_error = "Second error: authentication failed"

        state1 = state.next_attempt(error_message=first_error, timestamp=1000.0)
        state2 = state1.next_attempt(error_message=second_error, timestamp=1001.0)

        assert state1.last_error == first_error
        assert state2.last_error == second_error

    def test_none_error_clears_previous_error(self) -> None:
        """Test that None error message clears previous error.

        If a subsequent attempt succeeds (no error message), the
        last_error should be None even if previous state had an error.
        """
        state = ModelRetryState()

        state_with_error = state.next_attempt(
            error_message="Something went wrong", timestamp=1000.0
        )
        state_without_error = state_with_error.next_attempt(timestamp=1001.0)

        assert state_with_error.last_error == "Something went wrong"
        assert state_without_error.last_error is None

    def test_multiline_error_message_preserved(self) -> None:
        """Test that multiline error messages are preserved."""
        state = ModelRetryState()
        multiline_error = "Line 1: Connection failed\nLine 2: Host unreachable"

        next_state = state.next_attempt(error_message=multiline_error, timestamp=1000.0)

        assert next_state.last_error == multiline_error

    def test_unicode_error_message_preserved(self) -> None:
        """Test that unicode characters in error messages are preserved."""
        state = ModelRetryState()
        unicode_error = "Connection failed: host=\u65e5\u672c, status=\u2716"

        next_state = state.next_attempt(error_message=unicode_error, timestamp=1000.0)

        assert next_state.last_error == unicode_error

    # =========================================================================
    # Attempt Increment and Field Preservation Tests
    # =========================================================================

    def test_attempt_increments_by_one(self) -> None:
        """Test that attempt field increments by exactly 1."""
        state = ModelRetryState(attempt=0)

        next_state = state.next_attempt(timestamp=1000.0)

        assert next_state.attempt == 1

    def test_attempt_increments_from_nonzero(self) -> None:
        """Test that attempt increments correctly from non-zero starting point."""
        state = ModelRetryState(attempt=5)

        next_state = state.next_attempt(timestamp=1000.0)

        assert next_state.attempt == 6

    def test_max_attempts_preserved_across_next_attempt(self) -> None:
        """Test that max_attempts is preserved unchanged."""
        state = ModelRetryState(max_attempts=10)

        next_state = state.next_attempt(timestamp=1000.0)

        assert next_state.max_attempts == 10

    def test_backoff_multiplier_preserved_across_next_attempt(self) -> None:
        """Test that backoff_multiplier is preserved unchanged."""
        state = ModelRetryState(backoff_multiplier=3.5)

        next_state = state.next_attempt(timestamp=1000.0)

        assert next_state.backoff_multiplier == 3.5

    def test_all_immutable_fields_preserved(self) -> None:
        """Test that all immutable configuration fields are preserved.

        Fields that should be preserved:
        - max_attempts
        - backoff_multiplier

        Fields that should change:
        - attempt (incremented)
        - delay_seconds (calculated)
        - last_error (from parameter)
        - last_attempt_at (from parameter or time.time())
        """
        state = ModelRetryState(
            attempt=2,
            max_attempts=5,
            delay_seconds=4.0,
            backoff_multiplier=2.5,
            last_error="previous error",
            last_attempt_at=500.0,
        )

        next_state = state.next_attempt(error_message="new error", timestamp=1000.0)

        # Preserved fields
        assert next_state.max_attempts == 5
        assert next_state.backoff_multiplier == 2.5

        # Changed fields
        assert next_state.attempt == 3
        assert next_state.delay_seconds == 10.0  # 4.0 * 2.5 = 10.0
        assert next_state.last_error == "new error"
        assert next_state.last_attempt_at == 1000.0

    def test_multiple_consecutive_increments(self) -> None:
        """Test multiple consecutive next_attempt calls.

        Verify attempt correctly increments over many calls.
        """
        state = ModelRetryState(attempt=0, max_attempts=100)

        for expected_attempt in range(1, 11):
            state = state.next_attempt(timestamp=float(expected_attempt))
            assert state.attempt == expected_attempt

    # =========================================================================
    # Integration / Combined Behavior Tests
    # =========================================================================

    def test_full_retry_sequence_with_capping(self) -> None:
        """Test complete retry sequence that eventually hits delay cap.

        Starting with delay=1.0 and multiplier=2.0:
        - Attempt 1: delay=2.0
        - Attempt 2: delay=4.0
        - Attempt 3: delay=8.0
        - Attempt 4: delay=16.0
        - Attempt 5: delay=32.0
        - Attempt 6: delay=64.0
        - Attempt 7: delay=128.0
        - Attempt 8: delay=256.0
        - Attempt 9: delay=300.0 (capped, 512.0 > 300.0)
        - Attempt 10: delay=300.0 (stays capped)
        """
        state = ModelRetryState(
            delay_seconds=1.0,
            backoff_multiplier=2.0,
            max_attempts=15,
        )

        expected_delays = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 300.0, 300.0]

        for i, expected_delay in enumerate(expected_delays):
            state = state.next_attempt(
                error_message=f"Attempt {i + 1} failed", timestamp=1000.0 + i
            )
            assert state.delay_seconds == expected_delay, (
                f"Attempt {i + 1}: expected delay={expected_delay}, "
                f"got delay={state.delay_seconds}"
            )
            assert state.attempt == i + 1
            assert state.last_error == f"Attempt {i + 1} failed"

    def test_custom_max_delay_changes_capping_behavior(self) -> None:
        """Test that custom max_delay changes when capping occurs.

        With max_delay_seconds=50.0 instead of 300.0:
        - Attempt 1: delay=2.0
        - Attempt 2: delay=4.0
        - Attempt 3: delay=8.0
        - Attempt 4: delay=16.0
        - Attempt 5: delay=32.0
        - Attempt 6: delay=50.0 (capped, 64.0 > 50.0)
        """
        state = ModelRetryState(
            delay_seconds=1.0,
            backoff_multiplier=2.0,
            max_attempts=10,
        )

        expected_delays = [2.0, 4.0, 8.0, 16.0, 32.0, 50.0, 50.0]

        for i, expected_delay in enumerate(expected_delays):
            state = state.next_attempt(timestamp=1000.0 + i, max_delay_seconds=50.0)
            assert state.delay_seconds == expected_delay

    def test_immutability_original_state_unchanged(self) -> None:
        """Test that calling next_attempt does not mutate original state.

        ModelRetryState is frozen=True, so original should be unchanged.
        """
        original_state = ModelRetryState(
            attempt=0,
            max_attempts=3,
            delay_seconds=1.0,
            backoff_multiplier=2.0,
            last_error=None,
            last_attempt_at=None,
        )

        _ = original_state.next_attempt(error_message="error", timestamp=1000.0)

        # Original state should be completely unchanged
        assert original_state.attempt == 0
        assert original_state.delay_seconds == 1.0
        assert original_state.last_error is None
        assert original_state.last_attempt_at is None


class TestModelRetryStateCreation:
    """Test ModelRetryState creation and default values."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        state = ModelRetryState()

        assert state.attempt == 0
        assert state.max_attempts == 3
        assert state.delay_seconds == 1.0
        assert state.backoff_multiplier == 2.0
        assert state.last_error is None
        assert state.last_attempt_at is None

    def test_custom_values_accepted(self) -> None:
        """Test that custom values are accepted."""
        state = ModelRetryState(
            attempt=5,
            max_attempts=10,
            delay_seconds=5.5,
            backoff_multiplier=3.0,
            last_error="test error",
            last_attempt_at=12345.0,
        )

        assert state.attempt == 5
        assert state.max_attempts == 10
        assert state.delay_seconds == 5.5
        assert state.backoff_multiplier == 3.0
        assert state.last_error == "test error"
        assert state.last_attempt_at == 12345.0


class TestModelRetryStateHelperMethods:
    """Test ModelRetryState helper methods (is_retriable, is_final_attempt)."""

    def test_is_retriable_true_when_attempts_remaining(self) -> None:
        """Test is_retriable returns True when attempts remain."""
        state = ModelRetryState(attempt=0, max_attempts=3)
        assert state.is_retriable() is True

        state = ModelRetryState(attempt=2, max_attempts=3)
        assert state.is_retriable() is True

    def test_is_retriable_false_when_attempts_exhausted(self) -> None:
        """Test is_retriable returns False when attempts exhausted."""
        state = ModelRetryState(attempt=3, max_attempts=3)
        assert state.is_retriable() is False

        state = ModelRetryState(attempt=5, max_attempts=3)
        assert state.is_retriable() is False

    def test_is_final_attempt_true_at_last_attempt(self) -> None:
        """Test is_final_attempt returns True at last attempt."""
        state = ModelRetryState(attempt=2, max_attempts=3)
        assert state.is_final_attempt() is True

    def test_is_final_attempt_false_before_last(self) -> None:
        """Test is_final_attempt returns False before last attempt."""
        state = ModelRetryState(attempt=0, max_attempts=3)
        assert state.is_final_attempt() is False

        state = ModelRetryState(attempt=1, max_attempts=3)
        assert state.is_final_attempt() is False
