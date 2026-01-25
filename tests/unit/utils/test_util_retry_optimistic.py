# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for optimistic locking retry helper.

This test suite provides comprehensive coverage of the optimistic locking
retry utilities in omnibase_infra.utils.util_retry_optimistic:
    - OptimisticConflictError: Exception for exhausted retries
    - retry_on_optimistic_conflict: Async retry helper with exponential backoff

Test Organization:
    - TestOptimisticConflictErrorBasic: Exception attributes and message
    - TestRetryOnOptimisticConflictSuccess: Success scenarios (no conflict, retry success)
    - TestRetryOnOptimisticConflictExhausted: Retry exhaustion raises error
    - TestRetryOnOptimisticConflictBackoff: Backoff timing and calculation
    - TestRetryOnOptimisticConflictJitter: Jitter randomization behavior
    - TestRetryOnOptimisticConflictLogging: Correlation ID logging

Coverage Goals:
    - Success on first attempt (no conflict)
    - Success after N retries
    - Exhausted retries raises OptimisticConflictError
    - Backoff timing calculation (mock asyncio.sleep)
    - Jitter randomization behavior
    - Correlation ID logging
    - Edge cases (max_retries=0, immediate success after conflict)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.utils.util_retry_optimistic import (
    OptimisticConflictError,
    retry_on_optimistic_conflict,
)

# =============================================================================
# OptimisticConflictError Tests
# =============================================================================


@pytest.mark.unit
class TestOptimisticConflictErrorBasic:
    """Test suite for OptimisticConflictError basic functionality.

    Tests verify:
    - Exception attributes (attempts, last_result) are set correctly
    - Error message contains attempt count
    - Error message contains last result
    - Exception is a standard Exception subclass
    """

    def test_exception_has_attempts_attribute(self) -> None:
        """Test exception stores attempts count."""
        error = OptimisticConflictError(attempts=5, last_result={"row_count": 0})

        assert error.attempts == 5

    def test_exception_has_last_result_attribute(self) -> None:
        """Test exception stores last result."""
        last_result = {"row_count": 0, "version": 3}
        error = OptimisticConflictError(attempts=3, last_result=last_result)

        assert error.last_result == last_result

    def test_error_message_contains_attempts(self) -> None:
        """Test error message includes attempt count."""
        error = OptimisticConflictError(attempts=4, last_result={})

        assert "4 attempts" in str(error)

    def test_error_message_contains_last_result(self) -> None:
        """Test error message includes last result representation."""
        last_result = {"status": "conflict"}
        error = OptimisticConflictError(attempts=2, last_result=last_result)

        assert "conflict" in str(error)

    def test_exception_is_exception_subclass(self) -> None:
        """Test OptimisticConflictError is a standard Exception."""
        error = OptimisticConflictError(attempts=1, last_result=None)

        assert isinstance(error, Exception)

    def test_exception_can_be_raised_and_caught(self) -> None:
        """Test exception can be raised and caught normally."""
        with pytest.raises(OptimisticConflictError) as exc_info:
            raise OptimisticConflictError(attempts=3, last_result={"data": "test"})

        assert exc_info.value.attempts == 3
        assert exc_info.value.last_result == {"data": "test"}

    def test_exception_with_none_last_result(self) -> None:
        """Test exception accepts None as last_result."""
        error = OptimisticConflictError(attempts=1, last_result=None)

        assert error.last_result is None
        assert "None" in str(error)

    def test_exception_with_complex_last_result(self) -> None:
        """Test exception accepts complex objects as last_result."""
        complex_result = {
            "row_count": 0,
            "metadata": {"version": 5, "tags": ["a", "b"]},
        }
        error = OptimisticConflictError(attempts=2, last_result=complex_result)

        assert error.last_result == complex_result


# =============================================================================
# retry_on_optimistic_conflict Success Tests
# =============================================================================


@pytest.mark.unit
class TestRetryOnOptimisticConflictSuccess:
    """Test suite for successful retry scenarios.

    Tests verify:
    - Success on first attempt (no conflict detected)
    - Success after one retry
    - Success after multiple retries
    - Return value is passed through correctly
    """

    @pytest.mark.asyncio
    async def test_success_on_first_attempt_no_conflict(self) -> None:
        """Test function returns immediately when no conflict on first try."""

        async def successful_fn() -> dict[str, int]:
            return {"row_count": 1}

        result = await retry_on_optimistic_conflict(
            successful_fn,
            check_conflict=lambda r: r["row_count"] == 0,
        )

        assert result == {"row_count": 1}

    @pytest.mark.asyncio
    async def test_success_after_one_retry(self) -> None:
        """Test function succeeds after one retry."""
        attempt = 0

        async def fn_succeeds_second_try() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                return {"row_count": 0}  # Conflict
            return {"row_count": 1}  # Success

        result = await retry_on_optimistic_conflict(
            fn_succeeds_second_try,
            check_conflict=lambda r: r["row_count"] == 0,
            max_retries=3,
            initial_backoff=0.001,  # Fast for testing
        )

        assert result == {"row_count": 1}
        assert attempt == 2

    @pytest.mark.asyncio
    async def test_success_after_multiple_retries(self) -> None:
        """Test function succeeds after multiple retries."""
        attempt = 0

        async def fn_succeeds_on_fourth() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            if attempt < 4:
                return {"row_count": 0}  # Conflict
            return {"row_count": 1}  # Success

        result = await retry_on_optimistic_conflict(
            fn_succeeds_on_fourth,
            check_conflict=lambda r: r["row_count"] == 0,
            max_retries=5,
            initial_backoff=0.001,
        )

        assert result == {"row_count": 1}
        assert attempt == 4

    @pytest.mark.asyncio
    async def test_return_value_passed_through(self) -> None:
        """Test the actual return value is passed through correctly."""
        expected_result = {
            "row_count": 1,
            "data": {"name": "test", "version": 42},
            "metadata": ["tag1", "tag2"],
        }

        async def fn_with_complex_return() -> dict[str, object]:
            return expected_result

        result = await retry_on_optimistic_conflict(
            fn_with_complex_return,
            check_conflict=lambda r: False,  # Never conflict
        )

        assert result == expected_result

    @pytest.mark.asyncio
    async def test_custom_conflict_check_function(self) -> None:
        """Test custom conflict check function works correctly."""

        async def fn_returns_success_flag() -> dict[str, bool]:
            return {"success": True, "data": "value"}

        result = await retry_on_optimistic_conflict(
            fn_returns_success_flag,
            check_conflict=lambda r: r["success"] is False,  # Custom check
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_fn_called_expected_times_on_success_first_try(self) -> None:
        """Test fn is called exactly once on first-try success."""
        mock_fn = AsyncMock(return_value={"row_count": 1})

        await retry_on_optimistic_conflict(
            mock_fn,
            check_conflict=lambda r: r["row_count"] == 0,
        )

        assert mock_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_fn_called_expected_times_on_retry_success(self) -> None:
        """Test fn is called expected number of times on retry success."""
        call_count = 0

        async def counting_fn() -> dict[str, int]:
            nonlocal call_count
            call_count += 1
            # Succeed on 3rd attempt
            return {"row_count": 1 if call_count >= 3 else 0}

        await retry_on_optimistic_conflict(
            counting_fn,
            check_conflict=lambda r: r["row_count"] == 0,
            max_retries=5,
            initial_backoff=0.001,
        )

        assert call_count == 3


# =============================================================================
# retry_on_optimistic_conflict Exhausted Retries Tests
# =============================================================================


@pytest.mark.unit
class TestRetryOnOptimisticConflictExhausted:
    """Test suite for retry exhaustion scenarios.

    Tests verify:
    - OptimisticConflictError raised when all retries exhausted
    - Error contains correct attempt count
    - Error contains last result
    - Works with max_retries=0 (only initial attempt)
    """

    @pytest.mark.asyncio
    async def test_raises_error_when_retries_exhausted(self) -> None:
        """Test OptimisticConflictError raised when all retries fail."""

        async def always_conflict() -> dict[str, int]:
            return {"row_count": 0}

        with pytest.raises(OptimisticConflictError):
            await retry_on_optimistic_conflict(
                always_conflict,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=3,
                initial_backoff=0.001,
            )

    @pytest.mark.asyncio
    async def test_error_contains_correct_attempt_count(self) -> None:
        """Test error has correct total attempt count."""

        async def always_conflict() -> dict[str, int]:
            return {"row_count": 0}

        with pytest.raises(OptimisticConflictError) as exc_info:
            await retry_on_optimistic_conflict(
                always_conflict,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=4,  # 5 total attempts (1 initial + 4 retries)
                initial_backoff=0.001,
            )

        assert exc_info.value.attempts == 5

    @pytest.mark.asyncio
    async def test_error_contains_last_result(self) -> None:
        """Test error contains the result from the last failed attempt."""
        final_result = {"row_count": 0, "version": 99}

        async def always_conflict() -> dict[str, int]:
            return final_result

        with pytest.raises(OptimisticConflictError) as exc_info:
            await retry_on_optimistic_conflict(
                always_conflict,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=2,
                initial_backoff=0.001,
            )

        assert exc_info.value.last_result == final_result

    @pytest.mark.asyncio
    async def test_max_retries_zero_only_initial_attempt(self) -> None:
        """Test max_retries=0 means only initial attempt, no retries."""
        call_count = 0

        async def counting_conflict() -> dict[str, int]:
            nonlocal call_count
            call_count += 1
            return {"row_count": 0}

        with pytest.raises(OptimisticConflictError) as exc_info:
            await retry_on_optimistic_conflict(
                counting_conflict,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=0,
            )

        assert call_count == 1  # Only initial attempt
        assert exc_info.value.attempts == 1

    @pytest.mark.asyncio
    async def test_fn_called_max_retries_plus_one_times(self) -> None:
        """Test fn is called exactly max_retries + 1 times when exhausted."""
        mock_fn = AsyncMock(return_value={"row_count": 0})

        with pytest.raises(OptimisticConflictError):
            await retry_on_optimistic_conflict(
                mock_fn,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=5,
                initial_backoff=0.001,
            )

        assert mock_fn.call_count == 6  # 1 initial + 5 retries


# =============================================================================
# retry_on_optimistic_conflict Backoff Tests
# =============================================================================


@pytest.mark.unit
class TestRetryOnOptimisticConflictBackoff:
    """Test suite for backoff timing and calculation.

    Tests verify:
    - Exponential backoff with multiplier
    - Max backoff cap applied
    - Initial backoff used on first retry
    - asyncio.sleep called with correct delays
    """

    @pytest.mark.asyncio
    async def test_exponential_backoff_progression(self) -> None:
        """Test backoff increases exponentially with multiplier."""
        sleep_calls: list[float] = []

        async def track_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        attempt = 0

        async def fn_fails_until_last() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            # Succeed on 4th attempt (after 3 sleeps)
            return {"row_count": 1 if attempt >= 4 else 0}

        with patch(
            "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep", track_sleep
        ):
            await retry_on_optimistic_conflict(
                fn_fails_until_last,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=5,
                initial_backoff=0.1,
                backoff_multiplier=2.0,
                jitter=False,  # Disable jitter for predictable testing
            )

        # Should have 3 sleep calls: 0.1, 0.2, 0.4
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == pytest.approx(0.1)
        assert sleep_calls[1] == pytest.approx(0.2)
        assert sleep_calls[2] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_max_backoff_cap_applied(self) -> None:
        """Test backoff is capped at max_backoff."""
        sleep_calls: list[float] = []

        async def track_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        async def always_conflict() -> dict[str, int]:
            return {"row_count": 0}

        with (
            patch(
                "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep", track_sleep
            ),
            pytest.raises(OptimisticConflictError),
        ):
            await retry_on_optimistic_conflict(
                always_conflict,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=5,
                initial_backoff=1.0,
                max_backoff=2.0,  # Cap at 2.0
                backoff_multiplier=3.0,  # Would be 1, 3, 9, 27, 81 without cap
                jitter=False,
            )

        # All backoffs should be capped at 2.0 after the second
        assert sleep_calls[0] == pytest.approx(1.0)
        assert sleep_calls[1] == pytest.approx(2.0)  # 3.0 capped to 2.0
        assert sleep_calls[2] == pytest.approx(2.0)  # 9.0 capped to 2.0
        assert sleep_calls[3] == pytest.approx(2.0)  # 27.0 capped to 2.0
        assert sleep_calls[4] == pytest.approx(2.0)  # 81.0 capped to 2.0

    @pytest.mark.asyncio
    async def test_initial_backoff_used_on_first_retry(self) -> None:
        """Test initial_backoff is the delay for first retry."""
        sleep_calls: list[float] = []

        async def track_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        attempt = 0

        async def fn_succeeds_second() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            return {"row_count": 1 if attempt >= 2 else 0}

        with patch(
            "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep", track_sleep
        ):
            await retry_on_optimistic_conflict(
                fn_succeeds_second,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=3,
                initial_backoff=0.5,
                jitter=False,
            )

        assert len(sleep_calls) == 1
        assert sleep_calls[0] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_no_sleep_on_first_try_success(self) -> None:
        """Test no sleep occurs when first attempt succeeds."""
        sleep_called = False

        async def track_sleep(delay: float) -> None:
            nonlocal sleep_called
            sleep_called = True

        async def immediate_success() -> dict[str, int]:
            return {"row_count": 1}

        with patch(
            "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep", track_sleep
        ):
            await retry_on_optimistic_conflict(
                immediate_success,
                check_conflict=lambda r: r["row_count"] == 0,
            )

        assert sleep_called is False

    @pytest.mark.asyncio
    async def test_custom_backoff_multiplier(self) -> None:
        """Test custom backoff multiplier works correctly."""
        sleep_calls: list[float] = []

        async def track_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        attempt = 0

        async def fn_fails_thrice() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            return {"row_count": 1 if attempt >= 4 else 0}

        with patch(
            "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep", track_sleep
        ):
            await retry_on_optimistic_conflict(
                fn_fails_thrice,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=5,
                initial_backoff=0.1,
                backoff_multiplier=3.0,  # Triple each time
                jitter=False,
            )

        # 0.1, 0.3, 0.9
        assert sleep_calls[0] == pytest.approx(0.1)
        assert sleep_calls[1] == pytest.approx(0.3)
        assert sleep_calls[2] == pytest.approx(0.9)


# =============================================================================
# retry_on_optimistic_conflict Jitter Tests
# =============================================================================


@pytest.mark.unit
class TestRetryOnOptimisticConflictJitter:
    """Test suite for jitter randomization behavior.

    Tests verify:
    - Jitter adds randomization to delays
    - Jitter range is 50-150% of base delay
    - Jitter disabled produces consistent delays
    - Multiple retries have different jittered values
    """

    @pytest.mark.asyncio
    async def test_jitter_enabled_adds_randomization(self) -> None:
        """Test jitter adds randomization to sleep delays."""
        sleep_calls: list[float] = []

        async def track_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        async def always_conflict() -> dict[str, int]:
            return {"row_count": 0}

        # Run multiple times to verify randomization
        with patch(
            "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep", track_sleep
        ):
            with pytest.raises(OptimisticConflictError):
                await retry_on_optimistic_conflict(
                    always_conflict,
                    check_conflict=lambda r: r["row_count"] == 0,
                    max_retries=3,
                    initial_backoff=1.0,
                    jitter=True,
                )

        # With jitter, delays should be between 0.5*base and 1.5*base
        # Base delays: 1.0, 2.0, 4.0
        assert 0.5 <= sleep_calls[0] <= 1.5
        assert 1.0 <= sleep_calls[1] <= 3.0
        assert 2.0 <= sleep_calls[2] <= 6.0

    @pytest.mark.asyncio
    async def test_jitter_range_50_to_150_percent(self) -> None:
        """Test jitter is within 50-150% of base delay."""
        sleep_calls: list[float] = []

        async def track_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        async def always_conflict() -> dict[str, int]:
            return {"row_count": 0}

        # Fixed random value for testing (returns 0.5, making multiplier 1.0)
        with (
            patch(
                "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep", track_sleep
            ),
            patch(
                "omnibase_infra.utils.util_retry_optimistic.random.random",
                return_value=0.5,
            ),
            pytest.raises(OptimisticConflictError),
        ):
            await retry_on_optimistic_conflict(
                always_conflict,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=2,
                initial_backoff=1.0,
                backoff_multiplier=2.0,
                jitter=True,
            )

        # With random.random() = 0.5, multiplier is 0.5 + 0.5 = 1.0
        # So delays should equal base delays
        assert sleep_calls[0] == pytest.approx(1.0)
        assert sleep_calls[1] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_jitter_disabled_consistent_delays(self) -> None:
        """Test jitter=False produces consistent (non-random) delays."""

        async def run_with_tracking() -> list[float]:
            """Run a single test iteration and return recorded sleep calls."""
            sleep_calls: list[float] = []

            async def track_sleep(delay: float) -> None:
                sleep_calls.append(delay)

            async def always_conflict() -> dict[str, int]:
                return {"row_count": 0}

            with (
                patch(
                    "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep",
                    track_sleep,
                ),
                pytest.raises(OptimisticConflictError),
            ):
                await retry_on_optimistic_conflict(
                    always_conflict,
                    check_conflict=lambda r: r["row_count"] == 0,
                    max_retries=2,
                    initial_backoff=0.1,
                    jitter=False,
                )

            return sleep_calls.copy()

        # Run multiple times
        all_sleep_calls = [await run_with_tracking() for _ in range(3)]

        # All runs should have identical delays
        assert all_sleep_calls[0] == all_sleep_calls[1] == all_sleep_calls[2]

    @pytest.mark.asyncio
    async def test_jitter_with_min_random_value(self) -> None:
        """Test jitter with minimum random value (0.0) gives 50% of base."""
        sleep_calls: list[float] = []

        async def track_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        attempt = 0

        async def fn_succeeds_second() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            return {"row_count": 1 if attempt >= 2 else 0}

        with (
            patch(
                "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep", track_sleep
            ),
            patch(
                "omnibase_infra.utils.util_retry_optimistic.random.random",
                return_value=0.0,
            ),
        ):
            await retry_on_optimistic_conflict(
                fn_succeeds_second,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=3,
                initial_backoff=1.0,
                jitter=True,
            )

        # random.random() = 0.0, multiplier = 0.5 + 0.0 = 0.5
        assert sleep_calls[0] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_jitter_with_max_random_value(self) -> None:
        """Test jitter with maximum random value (1.0) gives 150% of base."""
        sleep_calls: list[float] = []

        async def track_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        attempt = 0

        async def fn_succeeds_second() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            return {"row_count": 1 if attempt >= 2 else 0}

        with (
            patch(
                "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep", track_sleep
            ),
            # Use 0.9999 instead of 1.0 since random.random() returns [0.0, 1.0)
            patch(
                "omnibase_infra.utils.util_retry_optimistic.random.random",
                return_value=0.9999,
            ),
        ):
            await retry_on_optimistic_conflict(
                fn_succeeds_second,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=3,
                initial_backoff=1.0,
                jitter=True,
            )

        # random.random() ~= 1.0, multiplier ~= 0.5 + 1.0 = 1.5
        assert sleep_calls[0] == pytest.approx(1.4999, rel=0.01)


# =============================================================================
# retry_on_optimistic_conflict Logging Tests
# =============================================================================


@pytest.mark.unit
class TestRetryOnOptimisticConflictLogging:
    """Test suite for correlation ID logging behavior.

    Tests verify:
    - Debug log on retry when correlation_id provided
    - Info log on resolution when correlation_id provided
    - No logging when correlation_id is None
    - Log extra contains correlation_id and attempt info
    """

    @pytest.mark.asyncio
    async def test_debug_log_on_retry_with_correlation_id(self) -> None:
        """Test debug log emitted on retry when correlation_id provided."""
        corr_id = uuid4()
        attempt = 0

        async def fn_succeeds_third() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            return {"row_count": 1 if attempt >= 3 else 0}

        with patch("omnibase_infra.utils.util_retry_optimistic.logger") as mock_logger:
            await retry_on_optimistic_conflict(
                fn_succeeds_third,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=5,
                initial_backoff=0.001,
                correlation_id=corr_id,
            )

            # Should have debug calls for retries
            assert mock_logger.debug.call_count == 2  # 2 retries before success

    @pytest.mark.asyncio
    async def test_info_log_on_resolution_with_correlation_id(self) -> None:
        """Test info log emitted when conflict resolved after retries."""
        corr_id = uuid4()
        attempt = 0

        async def fn_succeeds_second() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            return {"row_count": 1 if attempt >= 2 else 0}

        with patch("omnibase_infra.utils.util_retry_optimistic.logger") as mock_logger:
            await retry_on_optimistic_conflict(
                fn_succeeds_second,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=5,
                initial_backoff=0.001,
                correlation_id=corr_id,
            )

            # Should have info call for resolution
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "resolved" in call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_no_logging_without_correlation_id(self) -> None:
        """Test no logging when correlation_id is None."""
        attempt = 0

        async def fn_succeeds_second() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            return {"row_count": 1 if attempt >= 2 else 0}

        with patch("omnibase_infra.utils.util_retry_optimistic.logger") as mock_logger:
            await retry_on_optimistic_conflict(
                fn_succeeds_second,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=5,
                initial_backoff=0.001,
                correlation_id=None,
            )

            # Should have no logging calls
            mock_logger.debug.assert_not_called()
            mock_logger.info.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_extra_contains_correlation_id(self) -> None:
        """Test log extra dict contains correlation_id string."""
        corr_id = uuid4()
        attempt = 0

        async def fn_succeeds_second() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            return {"row_count": 1 if attempt >= 2 else 0}

        with patch("omnibase_infra.utils.util_retry_optimistic.logger") as mock_logger:
            await retry_on_optimistic_conflict(
                fn_succeeds_second,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=5,
                initial_backoff=0.001,
                correlation_id=corr_id,
            )

            # Check debug call extra
            debug_kwargs = mock_logger.debug.call_args[1]
            assert "extra" in debug_kwargs
            assert debug_kwargs["extra"]["correlation_id"] == str(corr_id)

            # Check info call extra
            info_kwargs = mock_logger.info.call_args[1]
            assert "extra" in info_kwargs
            assert info_kwargs["extra"]["correlation_id"] == str(corr_id)

    @pytest.mark.asyncio
    async def test_log_extra_contains_attempt_info(self) -> None:
        """Test log extra dict contains attempt information."""
        corr_id = uuid4()
        attempt = 0

        async def fn_succeeds_third() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            return {"row_count": 1 if attempt >= 3 else 0}

        with patch("omnibase_infra.utils.util_retry_optimistic.logger") as mock_logger:
            await retry_on_optimistic_conflict(
                fn_succeeds_third,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=5,
                initial_backoff=0.001,
                correlation_id=corr_id,
            )

            # Check first debug call has attempt info
            first_debug_kwargs = mock_logger.debug.call_args_list[0][1]
            assert first_debug_kwargs["extra"]["attempt"] == 1
            assert first_debug_kwargs["extra"]["max_attempts"] == 6

    @pytest.mark.asyncio
    async def test_no_info_log_on_first_try_success(self) -> None:
        """Test no info log when first attempt succeeds (no resolution needed)."""
        corr_id = uuid4()

        async def immediate_success() -> dict[str, int]:
            return {"row_count": 1}

        with patch("omnibase_infra.utils.util_retry_optimistic.logger") as mock_logger:
            await retry_on_optimistic_conflict(
                immediate_success,
                check_conflict=lambda r: r["row_count"] == 0,
                correlation_id=corr_id,
            )

            # No retries = no resolution log
            mock_logger.info.assert_not_called()
            mock_logger.debug.assert_not_called()


# =============================================================================
# retry_on_optimistic_conflict Edge Cases Tests
# =============================================================================


@pytest.mark.unit
class TestRetryOnOptimisticConflictEdgeCases:
    """Test suite for edge cases and boundary conditions.

    Tests verify:
    - Works with various check_conflict return types
    - Handles None return from fn
    - Default parameter values work correctly
    - Works with lambda closures
    """

    @pytest.mark.asyncio
    async def test_check_conflict_with_boolean_result(self) -> None:
        """Test check_conflict works with boolean return value."""

        async def fn_returns_bool() -> bool:
            return True

        result = await retry_on_optimistic_conflict(
            fn_returns_bool,
            check_conflict=lambda r: r is False,  # False indicates conflict
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_check_conflict_with_none_result(self) -> None:
        """Test check_conflict works when fn returns None."""

        async def fn_returns_none() -> None:
            return None

        result = await retry_on_optimistic_conflict(
            fn_returns_none,
            check_conflict=lambda r: r is not None,  # None means success
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_default_parameters_work(self) -> None:
        """Test function works with all default parameters."""

        async def simple_fn() -> dict[str, int]:
            return {"row_count": 1}

        # Only required parameters
        result = await retry_on_optimistic_conflict(
            simple_fn,
            check_conflict=lambda r: r["row_count"] == 0,
        )

        assert result == {"row_count": 1}

    @pytest.mark.asyncio
    async def test_works_with_closure(self) -> None:
        """Test fn can be a closure capturing external state."""
        external_data = {"counter": 0}

        async def closure_fn() -> dict[str, int]:
            external_data["counter"] += 1
            return {"row_count": external_data["counter"]}

        result = await retry_on_optimistic_conflict(
            closure_fn,
            check_conflict=lambda r: r["row_count"] < 3,  # Conflict until counter >= 3
            max_retries=5,
            initial_backoff=0.001,
        )

        assert result == {"row_count": 3}
        assert external_data["counter"] == 3

    @pytest.mark.asyncio
    async def test_works_with_partial_function(self) -> None:
        """Test fn can be created with functools.partial."""
        from functools import partial

        async def parameterized_fn(multiplier: int, base: int) -> dict[str, int]:
            return {"value": multiplier * base}

        bound_fn = partial(parameterized_fn, multiplier=5, base=10)

        result = await retry_on_optimistic_conflict(
            bound_fn,
            check_conflict=lambda r: r["value"] != 50,
        )

        assert result == {"value": 50}

    @pytest.mark.asyncio
    async def test_exception_from_fn_propagates(self) -> None:
        """Test exceptions from fn propagate up (not swallowed)."""

        async def fn_raises() -> dict[str, int]:
            raise ValueError("Intentional error")

        with pytest.raises(ValueError, match="Intentional error"):
            await retry_on_optimistic_conflict(
                fn_raises,
                check_conflict=lambda r: r["row_count"] == 0,
            )

    @pytest.mark.asyncio
    async def test_very_small_backoff_values(self) -> None:
        """Test works with very small backoff values."""
        attempt = 0

        async def fn_succeeds_second() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            return {"row_count": 1 if attempt >= 2 else 0}

        result = await retry_on_optimistic_conflict(
            fn_succeeds_second,
            check_conflict=lambda r: r["row_count"] == 0,
            initial_backoff=0.0001,  # 0.1ms
            max_backoff=0.001,  # 1ms
        )

        assert result == {"row_count": 1}

    @pytest.mark.asyncio
    async def test_backoff_multiplier_one(self) -> None:
        """Test backoff_multiplier=1.0 keeps constant delay."""
        sleep_calls: list[float] = []

        async def track_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        async def always_conflict() -> dict[str, int]:
            return {"row_count": 0}

        with (
            patch(
                "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep", track_sleep
            ),
            pytest.raises(OptimisticConflictError),
        ):
            await retry_on_optimistic_conflict(
                always_conflict,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=3,
                initial_backoff=0.5,
                backoff_multiplier=1.0,  # No increase
                jitter=False,
            )

        # All delays should be 0.5
        assert all(d == pytest.approx(0.5) for d in sleep_calls)

    @pytest.mark.asyncio
    async def test_large_max_retries(self) -> None:
        """Test works with large max_retries value."""
        attempt = 0

        async def fn_succeeds_eventually() -> dict[str, int]:
            nonlocal attempt
            attempt += 1
            # Succeed on 50th attempt
            return {"row_count": 1 if attempt >= 50 else 0}

        async def no_op_sleep(delay: float) -> None:
            pass  # Skip actual sleep for fast testing

        with patch(
            "omnibase_infra.utils.util_retry_optimistic.asyncio.sleep", no_op_sleep
        ):
            result = await retry_on_optimistic_conflict(
                fn_succeeds_eventually,
                check_conflict=lambda r: r["row_count"] == 0,
                max_retries=100,
                initial_backoff=0.0001,
                jitter=False,
            )

        assert result == {"row_count": 1}
        assert attempt == 50
