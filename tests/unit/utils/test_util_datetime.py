# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dedicated unit tests for util_datetime module.

This test suite provides comprehensive coverage of the datetime utilities
in omnibase_infra.utils.util_datetime:
    - is_timezone_aware: Check if a datetime is timezone-aware
    - ensure_timezone_aware: Convert naive datetimes to UTC with validation

Test Organization:
    - TestIsTimezoneAwareBasic: Core functionality for is_timezone_aware
    - TestIsTimezoneAwareEdgeCases: Edge cases including broken tzinfo
    - TestEnsureTimezoneAwareConversion: Naive to UTC conversion
    - TestEnsureTimezoneAwarePassthrough: Aware datetime passthrough
    - TestEnsureTimezoneAwareStrictMode: ValueError on naive with assume_utc=False
    - TestEnsureTimezoneAwareWarning: Warning logging behavior
    - TestEnsureTimezoneAwareContext: Context parameter in warnings

Coverage Goals:
    - Full coverage of is_timezone_aware function
    - Full coverage of ensure_timezone_aware function
    - Edge case handling (broken tzinfo, various timezones)
    - Logger mock verification for warning behavior
    - Context parameter propagation

Note: These tests focus on the utility functions themselves, independent of any
specific handler or node context.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.utils.util_datetime import (
    ensure_timezone_aware,
    is_timezone_aware,
    warn_if_naive_datetime,
)


class BrokenTzInfo(tzinfo):
    """A timezone implementation that returns None from utcoffset().

    This simulates a misconfigured tzinfo object where tzinfo is not None
    but utcoffset() returns None, which should be treated as naive.
    """

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        """Return None to simulate broken tzinfo."""
        return None

    def tzname(self, dt: datetime | None) -> str | None:
        """Return timezone name."""
        return "Broken"

    def dst(self, dt: datetime | None) -> timedelta | None:
        """Return DST offset."""
        return None


@pytest.mark.unit
class TestIsTimezoneAwareBasic:
    """Test suite for basic is_timezone_aware functionality.

    Tests verify core behavior:
    - Naive datetime returns False
    - UTC aware datetime returns True
    - Other timezone aware datetime returns True
    """

    def test_naive_datetime_returns_false(self) -> None:
        """Test naive datetime (no tzinfo) returns False."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        result = is_timezone_aware(naive_dt)

        assert result is False

    def test_utc_aware_datetime_returns_true(self) -> None:
        """Test datetime with UTC timezone returns True."""
        aware_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

        result = is_timezone_aware(aware_dt)

        assert result is True

    def test_datetime_now_utc_returns_true(self) -> None:
        """Test datetime.now(UTC) returns True for timezone awareness."""
        aware_dt = datetime.now(UTC)

        result = is_timezone_aware(aware_dt)

        assert result is True

    def test_positive_offset_timezone_returns_true(self) -> None:
        """Test datetime with positive UTC offset timezone returns True."""
        # UTC+5:30 (India Standard Time)
        ist = timezone(timedelta(hours=5, minutes=30))
        aware_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=ist)

        result = is_timezone_aware(aware_dt)

        assert result is True

    def test_negative_offset_timezone_returns_true(self) -> None:
        """Test datetime with negative UTC offset timezone returns True."""
        # UTC-8 (Pacific Standard Time)
        pst = timezone(timedelta(hours=-8))
        aware_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=pst)

        result = is_timezone_aware(aware_dt)

        assert result is True

    def test_zero_offset_timezone_returns_true(self) -> None:
        """Test datetime with zero UTC offset timezone (not UTC) returns True."""
        # timezone(timedelta(0)) is equivalent to UTC but a separate object
        zero_offset = timezone(timedelta(0))
        aware_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=zero_offset)

        result = is_timezone_aware(aware_dt)

        assert result is True


@pytest.mark.unit
class TestIsTimezoneAwareEdgeCases:
    """Test suite for edge cases in is_timezone_aware.

    Tests verify:
    - Broken tzinfo (returns None from utcoffset) treated as naive
    - Min/max datetime values
    """

    def test_broken_tzinfo_returns_false(self) -> None:
        """Test datetime with tzinfo set but utcoffset returning None returns False.

        This tests a malformed timezone implementation where tzinfo is not None
        but utcoffset() returns None, which should be treated as naive.
        """
        broken_tz = BrokenTzInfo()
        dt_with_broken_tz = datetime(2025, 1, 15, 12, 0, 0, tzinfo=broken_tz)

        # Verify the tzinfo is set but utcoffset returns None
        assert dt_with_broken_tz.tzinfo is not None
        assert dt_with_broken_tz.utcoffset() is None

        result = is_timezone_aware(dt_with_broken_tz)

        assert result is False

    def test_min_datetime_naive_returns_false(self) -> None:
        """Test datetime.min (naive) returns False."""
        result = is_timezone_aware(datetime.min)  # noqa: DTZ901

        assert result is False

    def test_max_datetime_naive_returns_false(self) -> None:
        """Test datetime.max (naive) returns False."""
        result = is_timezone_aware(datetime.max)  # noqa: DTZ901

        assert result is False

    def test_min_datetime_aware_returns_true(self) -> None:
        """Test datetime.min with UTC timezone returns True."""
        min_utc = datetime.min.replace(tzinfo=UTC)

        result = is_timezone_aware(min_utc)

        assert result is True


@pytest.mark.unit
class TestEnsureTimezoneAwareConversion:
    """Test suite for ensure_timezone_aware naive datetime conversion.

    Tests verify:
    - Naive datetime converted to UTC when assume_utc=True
    - Converted datetime has UTC tzinfo
    - Original datetime values preserved after conversion
    """

    def test_naive_datetime_converted_to_utc(self) -> None:
        """Test naive datetime is converted to UTC when assume_utc=True."""
        naive_dt = datetime(2025, 1, 15, 12, 30, 45)

        result = ensure_timezone_aware(naive_dt, assume_utc=True, warn_on_naive=False)

        assert result.tzinfo is UTC
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 12
        assert result.minute == 30
        assert result.second == 45

    def test_converted_datetime_has_utc_tzinfo(self) -> None:
        """Test converted datetime has exactly UTC tzinfo."""
        naive_dt = datetime(2025, 6, 15, 8, 0, 0)

        result = ensure_timezone_aware(naive_dt, assume_utc=True, warn_on_naive=False)

        assert result.tzinfo == UTC
        assert result.utcoffset() == timedelta(0)

    def test_conversion_preserves_microseconds(self) -> None:
        """Test conversion preserves microsecond precision."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0, 123456)

        result = ensure_timezone_aware(naive_dt, assume_utc=True, warn_on_naive=False)

        assert result.microsecond == 123456

    def test_default_assume_utc_is_true(self) -> None:
        """Test that assume_utc defaults to True."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        # Should not raise, meaning assume_utc defaults to True
        result = ensure_timezone_aware(naive_dt, warn_on_naive=False)

        assert result.tzinfo is UTC


@pytest.mark.unit
class TestEnsureTimezoneAwarePassthrough:
    """Test suite for ensure_timezone_aware aware datetime passthrough.

    Tests verify:
    - Aware datetime returned unchanged
    - Same object identity returned (not a copy)
    - Various timezone types pass through correctly
    """

    def test_utc_aware_datetime_returned_unchanged(self) -> None:
        """Test UTC aware datetime is returned without modification."""
        aware_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

        result = ensure_timezone_aware(aware_dt)

        assert result == aware_dt
        assert result.tzinfo == UTC

    def test_aware_datetime_returns_same_object(self) -> None:
        """Test aware datetime returns the exact same object (identity check)."""
        aware_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

        result = ensure_timezone_aware(aware_dt)

        assert result is aware_dt

    def test_non_utc_timezone_passed_through(self) -> None:
        """Test non-UTC timezone aware datetime is passed through unchanged."""
        # UTC+9 (Japan Standard Time)
        jst = timezone(timedelta(hours=9))
        aware_dt = datetime(2025, 1, 15, 21, 0, 0, tzinfo=jst)

        result = ensure_timezone_aware(aware_dt)

        assert result is aware_dt
        assert result.tzinfo == jst

    def test_negative_offset_timezone_passed_through(self) -> None:
        """Test negative offset timezone aware datetime is passed through."""
        # UTC-5 (Eastern Standard Time)
        est = timezone(timedelta(hours=-5))
        aware_dt = datetime(2025, 1, 15, 7, 0, 0, tzinfo=est)

        result = ensure_timezone_aware(aware_dt)

        assert result is aware_dt
        assert result.tzinfo == est


@pytest.mark.unit
class TestEnsureTimezoneAwareStrictMode:
    """Test suite for ensure_timezone_aware strict mode (assume_utc=False).

    Tests verify:
    - ValueError raised for naive datetime when assume_utc=False
    - Error message includes context when provided
    - Error message format is correct
    """

    def test_naive_datetime_raises_value_error_when_assume_utc_false(self) -> None:
        """Test naive datetime raises ValueError when assume_utc=False."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with pytest.raises(ValueError) as exc_info:
            ensure_timezone_aware(naive_dt, assume_utc=False)

        assert "Naive datetime not allowed" in str(exc_info.value)

    def test_value_error_includes_context_when_provided(self) -> None:
        """Test ValueError message includes context parameter."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with pytest.raises(ValueError) as exc_info:
            ensure_timezone_aware(naive_dt, assume_utc=False, context="updated_at")

        error_message = str(exc_info.value)
        assert "updated_at" in error_message
        assert "context:" in error_message

    def test_value_error_suggests_timezone_aware_usage(self) -> None:
        """Test ValueError message suggests using timezone-aware datetime."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with pytest.raises(ValueError) as exc_info:
            ensure_timezone_aware(naive_dt, assume_utc=False)

        error_message = str(exc_info.value)
        assert "datetime.now(UTC)" in error_message

    def test_aware_datetime_does_not_raise_in_strict_mode(self) -> None:
        """Test aware datetime does not raise even in strict mode."""
        aware_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

        # Should not raise
        result = ensure_timezone_aware(aware_dt, assume_utc=False)

        assert result is aware_dt


@pytest.mark.unit
class TestEnsureTimezoneAwareWarning:
    """Test suite for ensure_timezone_aware warning behavior.

    Tests verify:
    - Warning logged when warn_on_naive=True and datetime is naive
    - No warning logged when warn_on_naive=False
    - No warning logged for aware datetimes
    - Warning includes ISO format of datetime
    """

    def test_warning_logged_for_naive_datetime_when_warn_on_naive_true(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning is logged for naive datetime when warn_on_naive=True."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            ensure_timezone_aware(naive_dt, assume_utc=True, warn_on_naive=True)

        assert "Converting naive datetime to UTC" in caplog.text

    def test_warning_suggests_datetime_now_utc(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning message suggests using datetime.now(UTC)."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            ensure_timezone_aware(naive_dt, assume_utc=True, warn_on_naive=True)

        assert "datetime.now(UTC)" in caplog.text

    def test_no_warning_logged_when_warn_on_naive_false(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test no warning is logged when warn_on_naive=False."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            ensure_timezone_aware(naive_dt, assume_utc=True, warn_on_naive=False)

        assert "Converting naive datetime" not in caplog.text

    def test_no_warning_logged_for_aware_datetime(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test no warning is logged for aware datetime even with warn_on_naive=True."""
        aware_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

        with caplog.at_level(logging.WARNING):
            ensure_timezone_aware(aware_dt, assume_utc=True, warn_on_naive=True)

        assert "Converting naive datetime" not in caplog.text

    def test_default_warn_on_naive_is_true(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that warn_on_naive defaults to True."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            # Not specifying warn_on_naive, should default to True
            ensure_timezone_aware(naive_dt, assume_utc=True)

        assert "Converting naive datetime to UTC" in caplog.text

    def test_warning_logged_with_logger_mock(self) -> None:
        """Test warning is logged using mocked logger for precise verification."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with patch("omnibase_infra.utils.util_datetime.logger") as mock_logger:
            ensure_timezone_aware(naive_dt, assume_utc=True, warn_on_naive=True)

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "Converting naive datetime to UTC" in call_args[0][0]

    def test_no_warning_with_logger_mock_when_disabled(self) -> None:
        """Test no warning logged when warn_on_naive=False using mocked logger."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with patch("omnibase_infra.utils.util_datetime.logger") as mock_logger:
            ensure_timezone_aware(naive_dt, assume_utc=True, warn_on_naive=False)

            mock_logger.warning.assert_not_called()


@pytest.mark.unit
class TestEnsureTimezoneAwareContext:
    """Test suite for ensure_timezone_aware context parameter.

    Tests verify:
    - Context appears in warning message
    - Context appears in ValueError message
    - Context is included in logger extra data
    - None context is handled gracefully
    """

    def test_context_appears_in_warning_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test context parameter appears in warning message."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            ensure_timezone_aware(
                naive_dt, assume_utc=True, warn_on_naive=True, context="created_at"
            )

        assert "created_at" in caplog.text

    def test_context_format_in_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test context is formatted correctly in warning message."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            ensure_timezone_aware(
                naive_dt, assume_utc=True, warn_on_naive=True, context="updated_at"
            )

        # Context should appear with "for 'context'" format
        assert "for 'updated_at'" in caplog.text

    def test_context_in_logger_extra_data(self) -> None:
        """Test context is included in logger extra data."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with patch("omnibase_infra.utils.util_datetime.logger") as mock_logger:
            ensure_timezone_aware(
                naive_dt, assume_utc=True, warn_on_naive=True, context="my_column"
            )

            call_kwargs = mock_logger.warning.call_args[1]
            assert "extra" in call_kwargs
            assert call_kwargs["extra"]["context"] == "my_column"

    def test_none_context_handled_gracefully_in_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test None context does not cause errors in warning."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            # context=None is the default
            ensure_timezone_aware(naive_dt, assume_utc=True, warn_on_naive=True)

        # Should not have the "for 'context'" format when context is None
        assert "for '" not in caplog.text
        assert "Converting naive datetime to UTC" in caplog.text

    def test_none_context_handled_gracefully_in_error(self) -> None:
        """Test None context does not cause errors in ValueError."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with pytest.raises(ValueError) as exc_info:
            ensure_timezone_aware(naive_dt, assume_utc=False, context=None)

        # Should not have "(context: )" with empty context
        error_message = str(exc_info.value)
        assert "(context:" not in error_message

    def test_warning_extra_contains_naive_datetime_iso(self) -> None:
        """Test warning extra data contains naive datetime in ISO format."""
        naive_dt = datetime(2025, 1, 15, 12, 30, 45)

        with patch("omnibase_infra.utils.util_datetime.logger") as mock_logger:
            ensure_timezone_aware(naive_dt, assume_utc=True, warn_on_naive=True)

            call_kwargs = mock_logger.warning.call_args[1]
            assert "extra" in call_kwargs
            assert call_kwargs["extra"]["naive_datetime"] == "2025-01-15T12:30:45"

    def test_warning_extra_contains_action(self) -> None:
        """Test warning extra data contains action field."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with patch("omnibase_infra.utils.util_datetime.logger") as mock_logger:
            ensure_timezone_aware(naive_dt, assume_utc=True, warn_on_naive=True)

            call_kwargs = mock_logger.warning.call_args[1]
            assert "extra" in call_kwargs
            assert call_kwargs["extra"]["action"] == "converted_to_utc"


@pytest.mark.unit
class TestEnsureTimezoneAwareBrokenTzInfo:
    """Test suite for ensure_timezone_aware with broken tzinfo.

    Tests verify behavior when datetime has tzinfo set but utcoffset returns None.
    """

    def test_broken_tzinfo_treated_as_naive_with_assume_utc_true(self) -> None:
        """Test datetime with broken tzinfo is converted when assume_utc=True."""
        broken_tz = BrokenTzInfo()
        dt_with_broken_tz = datetime(2025, 1, 15, 12, 0, 0, tzinfo=broken_tz)

        result = ensure_timezone_aware(
            dt_with_broken_tz, assume_utc=True, warn_on_naive=False
        )

        # Should be converted to UTC since utcoffset() returns None
        assert result.tzinfo is UTC
        assert result.utcoffset() == timedelta(0)

    def test_broken_tzinfo_raises_with_assume_utc_false(self) -> None:
        """Test datetime with broken tzinfo raises ValueError when assume_utc=False."""
        broken_tz = BrokenTzInfo()
        dt_with_broken_tz = datetime(2025, 1, 15, 12, 0, 0, tzinfo=broken_tz)

        with pytest.raises(ValueError) as exc_info:
            ensure_timezone_aware(dt_with_broken_tz, assume_utc=False)

        assert "Naive datetime not allowed" in str(exc_info.value)

    def test_broken_tzinfo_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test datetime with broken tzinfo logs warning when warn_on_naive=True."""
        broken_tz = BrokenTzInfo()
        dt_with_broken_tz = datetime(2025, 1, 15, 12, 0, 0, tzinfo=broken_tz)

        with caplog.at_level(logging.WARNING):
            ensure_timezone_aware(
                dt_with_broken_tz, assume_utc=True, warn_on_naive=True
            )

        assert "Converting naive datetime to UTC" in caplog.text


# =============================================================================
# warn_if_naive_datetime Tests
# =============================================================================


@pytest.mark.unit
class TestWarnIfNaiveDatetimeBasic:
    """Test suite for basic warn_if_naive_datetime functionality.

    Tests verify core behavior:
    - Naive datetime logs warning
    - Aware datetime produces no warning (silent)
    - Function returns None (warning-only, no mutation)
    """

    def test_naive_datetime_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test naive datetime triggers a warning log."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(naive_dt)

        assert "Naive datetime detected" in caplog.text

    def test_aware_datetime_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test timezone-aware datetime produces no warning (silent)."""
        aware_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(aware_dt)

        assert "Naive datetime detected" not in caplog.text
        assert caplog.text == ""

    def test_function_returns_none(self) -> None:
        """Test function returns None (warning-only, does not mutate)."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        result = warn_if_naive_datetime(naive_dt)

        assert result is None

    def test_function_does_not_mutate_datetime(self) -> None:
        """Test function does not modify the input datetime."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)
        original_tzinfo = naive_dt.tzinfo

        warn_if_naive_datetime(naive_dt)

        # Datetime should be unchanged - still naive
        assert naive_dt.tzinfo is original_tzinfo
        assert naive_dt.tzinfo is None

    def test_warning_message_suggests_utc_usage(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning message suggests using datetime.now(UTC)."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(naive_dt)

        assert "datetime.now(UTC)" in caplog.text


@pytest.mark.unit
class TestWarnIfNaiveDatetimeParameters:
    """Test suite for warn_if_naive_datetime parameter handling.

    Tests verify:
    - field_name included in warning message
    - context included in warning message
    - correlation_id included in log extra
    - All parameters None handled gracefully
    - Both field_name AND context appear when provided
    """

    def test_field_name_appears_in_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test field_name parameter appears in warning message."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(naive_dt, field_name="created_at")

        assert "created_at" in caplog.text
        assert "field 'created_at'" in caplog.text

    def test_context_appears_in_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test context parameter appears in warning message."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(naive_dt, context="manifest_persistence")

        assert "manifest_persistence" in caplog.text
        assert "context 'manifest_persistence'" in caplog.text

    def test_correlation_id_in_log_extra(self) -> None:
        """Test correlation_id included in log extra when provided."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)
        corr_id = uuid4()

        with patch("omnibase_infra.utils.util_datetime.logger") as mock_logger:
            warn_if_naive_datetime(naive_dt, correlation_id=corr_id)

            mock_logger.warning.assert_called_once()
            call_kwargs = mock_logger.warning.call_args[1]
            assert "extra" in call_kwargs
            assert call_kwargs["extra"]["correlation_id"] == str(corr_id)

    def test_all_parameters_none_works(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test function works gracefully when all optional parameters are None."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            # All optional params default to None
            warn_if_naive_datetime(naive_dt)

        # Should still produce warning without errors
        assert "Naive datetime detected" in caplog.text
        # Should show generic location when no field_name/context
        assert "datetime value" in caplog.text

    def test_both_field_name_and_context_appear(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test both field_name AND context appear in message when provided."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(
                naive_dt,
                field_name="updated_at",
                context="handler_execution",
            )

        assert "updated_at" in caplog.text
        assert "handler_execution" in caplog.text
        assert "field 'updated_at'" in caplog.text
        assert "context 'handler_execution'" in caplog.text

    def test_field_name_in_extra_data(self) -> None:
        """Test field_name is included in logger extra data."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with patch("omnibase_infra.utils.util_datetime.logger") as mock_logger:
            warn_if_naive_datetime(naive_dt, field_name="my_field")

            call_kwargs = mock_logger.warning.call_args[1]
            assert "extra" in call_kwargs
            assert call_kwargs["extra"]["field_name"] == "my_field"

    def test_context_in_extra_data(self) -> None:
        """Test context is included in logger extra data."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with patch("omnibase_infra.utils.util_datetime.logger") as mock_logger:
            warn_if_naive_datetime(naive_dt, context="my_context")

            call_kwargs = mock_logger.warning.call_args[1]
            assert "extra" in call_kwargs
            assert call_kwargs["extra"]["context"] == "my_context"

    def test_datetime_value_in_extra_data(self) -> None:
        """Test datetime value in ISO format is in logger extra data."""
        naive_dt = datetime(2025, 1, 15, 12, 30, 45)

        with patch("omnibase_infra.utils.util_datetime.logger") as mock_logger:
            warn_if_naive_datetime(naive_dt)

            call_kwargs = mock_logger.warning.call_args[1]
            assert "extra" in call_kwargs
            assert call_kwargs["extra"]["datetime_value"] == "2025-01-15T12:30:45"

    def test_none_correlation_id_not_in_extra(self) -> None:
        """Test correlation_id not added to extra when None."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with patch("omnibase_infra.utils.util_datetime.logger") as mock_logger:
            warn_if_naive_datetime(naive_dt, correlation_id=None)

            call_kwargs = mock_logger.warning.call_args[1]
            assert "extra" in call_kwargs
            assert "correlation_id" not in call_kwargs["extra"]


@pytest.mark.unit
class TestWarnIfNaiveDatetimeLogger:
    """Test suite for warn_if_naive_datetime logger parameter.

    Tests verify:
    - Custom logger used when provided
    - Module logger used when logger=None
    """

    def test_custom_logger_used_when_provided(self) -> None:
        """Test custom logger is used when provided."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)
        custom_logger = logging.getLogger("custom.test.logger")

        with patch.object(custom_logger, "warning") as mock_warning:
            warn_if_naive_datetime(naive_dt, logger=custom_logger)

            mock_warning.assert_called_once()
            assert "Naive datetime detected" in mock_warning.call_args[0][0]

    def test_module_logger_used_when_none(self) -> None:
        """Test module logger is used when logger=None."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with patch("omnibase_infra.utils.util_datetime.logger") as mock_module_logger:
            warn_if_naive_datetime(naive_dt, logger=None)

            mock_module_logger.warning.assert_called_once()

    def test_custom_logger_not_module_logger(self) -> None:
        """Test custom logger means module logger is NOT used."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)
        custom_logger = logging.getLogger("custom.test.logger")

        with (
            patch.object(custom_logger, "warning") as mock_custom_warning,
            patch("omnibase_infra.utils.util_datetime.logger") as mock_module_logger,
        ):
            warn_if_naive_datetime(naive_dt, logger=custom_logger)

            # Custom logger should be called
            mock_custom_warning.assert_called_once()
            # Module logger should NOT be called
            mock_module_logger.warning.assert_not_called()

    def test_aware_datetime_custom_logger_not_called(self) -> None:
        """Test custom logger is not called for aware datetime."""
        aware_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        custom_logger = logging.getLogger("custom.test.logger")

        with patch.object(custom_logger, "warning") as mock_warning:
            warn_if_naive_datetime(aware_dt, logger=custom_logger)

            mock_warning.assert_not_called()


@pytest.mark.unit
class TestWarnIfNaiveDatetimeEdgeCases:
    """Test suite for warn_if_naive_datetime edge cases.

    Tests verify:
    - UTC aware datetime produces no warning
    - Non-UTC aware datetime produces no warning
    - Min/max naive datetime values warn correctly
    - Broken tzinfo (utcoffset returns None) triggers warning
    """

    def test_utc_aware_datetime_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test UTC-aware datetime produces no warning."""
        utc_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(utc_dt)

        assert "Naive datetime detected" not in caplog.text
        assert caplog.text == ""

    def test_non_utc_aware_datetime_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test non-UTC timezone-aware datetime produces no warning."""
        # UTC+9 (Japan Standard Time)
        jst = timezone(timedelta(hours=9))
        jst_dt = datetime(2025, 1, 15, 21, 0, 0, tzinfo=jst)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(jst_dt)

        assert "Naive datetime detected" not in caplog.text
        assert caplog.text == ""

    def test_negative_offset_aware_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test negative offset timezone-aware datetime produces no warning."""
        # UTC-8 (Pacific Standard Time)
        pst = timezone(timedelta(hours=-8))
        pst_dt = datetime(2025, 1, 15, 4, 0, 0, tzinfo=pst)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(pst_dt)

        assert "Naive datetime detected" not in caplog.text

    def test_min_datetime_naive_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test datetime.min (naive) triggers warning."""
        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(datetime.min)  # noqa: DTZ901

        assert "Naive datetime detected" in caplog.text

    def test_max_datetime_naive_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test datetime.max (naive) triggers warning."""
        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(datetime.max)  # noqa: DTZ901

        assert "Naive datetime detected" in caplog.text

    def test_min_datetime_aware_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test datetime.min with UTC timezone produces no warning."""
        min_utc = datetime.min.replace(tzinfo=UTC)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(min_utc)

        assert "Naive datetime detected" not in caplog.text

    def test_max_datetime_aware_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test datetime.max with UTC timezone produces no warning."""
        max_utc = datetime.max.replace(tzinfo=UTC)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(max_utc)

        assert "Naive datetime detected" not in caplog.text

    def test_broken_tzinfo_triggers_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test datetime with broken tzinfo (utcoffset returns None) triggers warning.

        This tests a malformed timezone implementation where tzinfo is not None
        but utcoffset() returns None, which should be treated as naive.
        """
        broken_tz = BrokenTzInfo()
        dt_with_broken_tz = datetime(2025, 1, 15, 12, 0, 0, tzinfo=broken_tz)

        # Verify the setup: tzinfo is set but utcoffset returns None
        assert dt_with_broken_tz.tzinfo is not None
        assert dt_with_broken_tz.utcoffset() is None

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(dt_with_broken_tz)

        assert "Naive datetime detected" in caplog.text

    def test_zero_offset_timezone_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test zero offset timezone (not UTC object) produces no warning."""
        # timezone(timedelta(0)) is equivalent to UTC but a separate object
        zero_offset = timezone(timedelta(0))
        aware_dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=zero_offset)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(aware_dt)

        assert "Naive datetime detected" not in caplog.text

    def test_datetime_with_microseconds_naive_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test naive datetime with microseconds triggers warning correctly."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0, 123456)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(naive_dt)

        assert "Naive datetime detected" in caplog.text


@pytest.mark.unit
class TestWarnIfNaiveDatetimeMessageFormat:
    """Test suite for warn_if_naive_datetime message format details.

    Tests verify the specific format of warning messages with various
    combinations of field_name and context parameters.
    """

    def test_field_name_only_format(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test message format when only field_name provided."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(naive_dt, field_name="created_at")

        # Should contain "field 'created_at'" but not "context"
        assert "field 'created_at'" in caplog.text
        # Should not mention context when not provided
        assert "context '" not in caplog.text

    def test_context_only_format(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test message format when only context provided."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(naive_dt, context="manifest_persistence")

        # Should contain "context 'manifest_persistence'" but not "field"
        assert "context 'manifest_persistence'" in caplog.text
        # Should not mention field when not provided
        assert "field '" not in caplog.text

    def test_neither_field_name_nor_context_format(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test message format when neither field_name nor context provided."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(naive_dt)

        # Should use generic "datetime value" location
        assert "datetime value" in caplog.text

    def test_both_field_and_context_joined_with_in(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test field_name and context are joined with 'in' when both provided."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)

        with caplog.at_level(logging.WARNING):
            warn_if_naive_datetime(
                naive_dt,
                field_name="timestamp",
                context="event_processing",
            )

        # The format should be "field 'X' in context 'Y'"
        assert "field 'timestamp'" in caplog.text
        assert "context 'event_processing'" in caplog.text
        assert " in " in caplog.text
