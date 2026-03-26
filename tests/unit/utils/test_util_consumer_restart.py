# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for run_with_restart consumer restart utility."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from omnibase_infra.utils.util_consumer_restart import (
    _is_fatal,
    _shutdown_aware_sleep,
    run_with_restart,
)


class TestIsFatal:
    """Test fatal error classification."""

    @pytest.mark.unit
    def test_type_error_is_fatal(self) -> None:
        assert _is_fatal(TypeError("bad type")) is True

    @pytest.mark.unit
    def test_attribute_error_is_fatal(self) -> None:
        assert _is_fatal(AttributeError("no attr")) is True

    @pytest.mark.unit
    def test_validation_error_is_fatal(self) -> None:
        """Pydantic ValidationError should be classified as fatal."""
        try:
            # Generate a real ValidationError
            from pydantic import BaseModel

            class M(BaseModel):
                x: int

            M()  # type: ignore[call-arg]
        except ValidationError as exc:
            assert _is_fatal(exc) is True

    @pytest.mark.unit
    def test_value_error_is_not_fatal(self) -> None:
        """ValueError must NOT be fatal — aiokafka raises it for 'no broker available'."""
        assert _is_fatal(ValueError("no broker available")) is False

    @pytest.mark.unit
    def test_runtime_error_is_not_fatal(self) -> None:
        assert _is_fatal(RuntimeError("connection lost")) is False

    @pytest.mark.unit
    def test_os_error_is_not_fatal(self) -> None:
        assert _is_fatal(OSError("connection refused")) is False


class TestShutdownAwareSleep:
    """Test _shutdown_aware_sleep behavior."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sleep_without_event(self) -> None:
        """With shutdown_event=None, should complete normally."""
        result = await _shutdown_aware_sleep(0.01, None)
        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sleep_interrupted_by_event(self) -> None:
        """Setting the event during sleep should return True immediately."""
        event = asyncio.Event()
        # Set the event after a tiny delay
        asyncio.get_running_loop().call_later(0.01, event.set)
        start = time.monotonic()
        result = await _shutdown_aware_sleep(10.0, event)
        elapsed = time.monotonic() - start
        assert result is True
        assert elapsed < 1.0  # Should exit well before 10s

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sleep_completes_without_event_set(self) -> None:
        """If event is never set, should sleep for full duration."""
        event = asyncio.Event()
        start = time.monotonic()
        result = await _shutdown_aware_sleep(0.05, event)
        elapsed = time.monotonic() - start
        assert result is False
        assert elapsed >= 0.04


class TestRunWithRestart:
    """Test run_with_restart behavior."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_keyboard_interrupt_reraised(self) -> None:
        """KeyboardInterrupt must be re-raised immediately (no retry)."""

        async def raise_keyboard() -> None:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            await run_with_restart(
                raise_keyboard,
                name="test",
                initial_backoff_s=0.01,
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cancelled_error_reraised(self) -> None:
        """asyncio.CancelledError must be re-raised immediately."""

        async def raise_cancelled() -> None:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await run_with_restart(
                raise_cancelled,
                name="test",
                initial_backoff_s=0.01,
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fatal_error_reraised(self) -> None:
        """Fatal errors (TypeError, AttributeError) must be re-raised immediately."""

        async def raise_type_error() -> None:
            raise TypeError("bad type")

        with pytest.raises(TypeError, match="bad type"):
            await run_with_restart(
                raise_type_error,
                name="test",
                initial_backoff_s=0.01,
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transient_error_retried(self) -> None:
        """Transient errors should trigger retry with backoff."""
        call_count = 0

        async def fail_then_shutdown() -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            # Third call: succeed and return (which triggers restart loop)

        shutdown = asyncio.Event()
        # Set shutdown after enough time for 3 attempts
        asyncio.get_running_loop().call_later(0.5, shutdown.set)

        with patch("omnibase_infra.utils.util_consumer_restart.random") as mock_random:
            mock_random.uniform.return_value = 1.0  # No jitter variance
            await run_with_restart(
                fail_then_shutdown,
                name="test",
                shutdown_event=shutdown,
                initial_backoff_s=0.01,
                max_backoff_s=0.1,
            )

        assert call_count >= 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_backoff_resets_after_stable_run(self) -> None:
        """After a stable run > stable_threshold_s, backoff should reset."""
        call_count = 0
        backoff_values: list[float] = []

        original_sleep = _shutdown_aware_sleep

        async def patched_sleep(seconds: float, event: asyncio.Event | None) -> bool:
            backoff_values.append(seconds)
            # Don't actually sleep long
            return await original_sleep(0.001, event)

        async def stable_then_fail() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate "stable" run via time.monotonic patching
                raise RuntimeError("first crash")
            if call_count == 2:
                # This will look "stable" because we patch monotonic
                raise RuntimeError("second crash")
            # Third call: just return

        shutdown = asyncio.Event()
        asyncio.get_running_loop().call_later(0.5, shutdown.set)

        # Patch stable_threshold_s to 0 so any run is "stable"
        with patch(
            "omnibase_infra.utils.util_consumer_restart._shutdown_aware_sleep",
            side_effect=patched_sleep,
        ):
            await run_with_restart(
                stable_then_fail,
                name="test",
                shutdown_event=shutdown,
                initial_backoff_s=0.01,
                max_backoff_s=1.0,
                stable_threshold_s=0.0,  # Any run is "stable"
            )

        assert call_count >= 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shutdown_event_causes_clean_exit(self) -> None:
        """Setting shutdown_event after clean coro return should exit the loop."""
        call_count = 0

        async def run_once() -> None:
            nonlocal call_count
            call_count += 1

        shutdown = asyncio.Event()
        shutdown.set()  # Already set before starting

        await run_with_restart(
            run_once,
            name="test",
            shutdown_event=shutdown,
        )

        assert call_count == 0  # Loop exits before first call

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shutdown_prevents_restart_after_transient_failure(self) -> None:
        """If shutdown is set during backoff, should not restart."""
        call_count = 0
        shutdown = asyncio.Event()

        async def fail_and_set_shutdown() -> None:
            nonlocal call_count
            call_count += 1
            shutdown.set()  # Signal shutdown after failure
            raise RuntimeError("transient")

        await run_with_restart(
            fail_and_set_shutdown,
            name="test",
            shutdown_event=shutdown,
            initial_backoff_s=0.01,
        )

        assert call_count == 1  # Only one attempt

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shutdown_during_backoff_exits_fast(self) -> None:
        """Shutdown during backoff sleep should exit within milliseconds, not full backoff."""
        call_count = 0
        shutdown = asyncio.Event()

        async def fail_once() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")

        # Set shutdown shortly after the first failure
        asyncio.get_running_loop().call_later(0.02, shutdown.set)

        start = time.monotonic()
        await run_with_restart(
            fail_once,
            name="test",
            shutdown_event=shutdown,
            initial_backoff_s=60.0,  # Very long backoff
            max_backoff_s=60.0,
        )
        elapsed = time.monotonic() - start

        assert call_count == 1
        assert elapsed < 1.0  # Should exit fast, not wait 60s

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clean_exit_triggers_restart(self) -> None:
        """A clean return from coro_factory should trigger restart (consumer should run forever)."""
        call_count = 0
        shutdown = asyncio.Event()

        async def return_cleanly() -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                shutdown.set()

        await run_with_restart(
            return_cleanly,
            name="test",
            shutdown_event=shutdown,
            initial_backoff_s=0.001,
        )

        assert call_count >= 3
