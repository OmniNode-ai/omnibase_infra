# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for circuit breaker active recovery timer.

This module validates the active recovery mechanism that transitions the circuit
breaker from OPEN to HALF_OPEN via a background asyncio task, without requiring
a caller to invoke _check_circuit_breaker().

Tests cover:
    - Auto-recovery to HALF_OPEN after reset_timeout (no passive check needed)
    - Timer cancellation on manual reset_circuit_breaker()
    - Timer idempotency (double open does not create two timers)
    - Timer cleanup via cancel_active_recovery()
    - Opt-out via enable_active_recovery=False
    - Config-based initialization with enable_active_recovery

Related:
    - mixin_async_circuit_breaker.py: Implementation under test
    - model_circuit_breaker_config.py: Config model with enable_active_recovery field
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from omnibase_infra.enums import EnumCircuitState, EnumInfraTransportType
from omnibase_infra.mixins.mixin_async_circuit_breaker import (
    MixinAsyncCircuitBreaker,
)

# =============================================================================
# Test Helper
# =============================================================================


class ActiveRecoveryTestService(MixinAsyncCircuitBreaker):
    """Test service for active recovery timer tests."""

    def __init__(
        self,
        threshold: int = 3,
        reset_timeout: float = 0.2,
        service_name: str = "test-active-recovery",
        enable_active_recovery: bool = True,
    ) -> None:
        self._init_circuit_breaker(
            threshold=threshold,
            reset_timeout=reset_timeout,
            service_name=service_name,
            transport_type=EnumInfraTransportType.HTTP,
            enable_active_recovery=enable_active_recovery,
        )

    async def record_failure(
        self, operation: str = "test_op", correlation_id: UUID | None = None
    ) -> None:
        async with self._circuit_breaker_lock:
            await self._record_circuit_failure(operation, correlation_id)

    async def reset_circuit(self) -> None:
        async with self._circuit_breaker_lock:
            await self._reset_circuit_breaker()

    async def check_circuit(
        self, operation: str = "test_op", correlation_id: UUID | None = None
    ) -> None:
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

    def get_state(self) -> EnumCircuitState:
        if self._circuit_breaker_open:
            return EnumCircuitState.OPEN
        if self._circuit_breaker_half_open:
            return EnumCircuitState.HALF_OPEN
        return EnumCircuitState.CLOSED

    def get_failure_count(self) -> int:
        return self._circuit_breaker_failures

    def has_recovery_task(self) -> bool:
        return self._cb_recovery_task is not None and not self._cb_recovery_task.done()


# =============================================================================
# Tests: Active Recovery to HALF_OPEN
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestActiveRecoveryTransition:
    """Test that the circuit auto-recovers to HALF_OPEN without callers."""

    async def test_auto_recovery_to_half_open_without_passive_check(self) -> None:
        """Open circuit, do NOT call _check_circuit_breaker, wait for timeout.

        The active recovery timer should transition to HALF_OPEN on its own.
        """
        reset_timeout = 0.15
        service = ActiveRecoveryTestService(threshold=2, reset_timeout=reset_timeout)

        # Open the circuit
        await service.record_failure()
        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN

        # Wait for the active recovery timer to fire
        await asyncio.sleep(reset_timeout + 0.1)

        # State should be HALF_OPEN -- without any call to check_circuit
        assert service.get_state() == EnumCircuitState.HALF_OPEN
        assert service.get_failure_count() == 0

    async def test_active_recovery_resets_failure_count(self) -> None:
        """Verify that active recovery resets the failure count to zero."""
        reset_timeout = 0.1
        service = ActiveRecoveryTestService(threshold=1, reset_timeout=reset_timeout)

        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN
        assert service.get_failure_count() == 1

        await asyncio.sleep(reset_timeout + 0.1)

        assert service.get_state() == EnumCircuitState.HALF_OPEN
        assert service.get_failure_count() == 0

    async def test_active_recovery_after_half_open_failure_reopens(self) -> None:
        """Full cycle: OPEN -> HALF_OPEN (active) -> fail -> OPEN -> HALF_OPEN (active).

        Verify the timer restarts when the circuit re-opens from HALF_OPEN.
        """
        reset_timeout = 0.1
        service = ActiveRecoveryTestService(threshold=1, reset_timeout=reset_timeout)

        # First open
        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN

        # Wait for active recovery
        await asyncio.sleep(reset_timeout + 0.1)
        assert service.get_state() == EnumCircuitState.HALF_OPEN

        # Fail in HALF_OPEN -> re-opens the circuit
        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN

        # Wait for active recovery again (timer should have restarted)
        await asyncio.sleep(reset_timeout + 0.1)
        assert service.get_state() == EnumCircuitState.HALF_OPEN


# =============================================================================
# Tests: Timer Cancellation on Manual Reset
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestActiveRecoveryCancellation:
    """Test that the timer is cancelled when the circuit is manually reset."""

    async def test_timer_cancelled_on_manual_reset(self) -> None:
        """Open circuit, then manually reset. Timer should be cancelled."""
        reset_timeout = 1.0  # Long timeout so timer does not fire naturally
        service = ActiveRecoveryTestService(threshold=2, reset_timeout=reset_timeout)

        # Open the circuit
        await service.record_failure()
        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN
        assert service.has_recovery_task()

        # Manual reset
        await service.reset_circuit()
        assert service.get_state() == EnumCircuitState.CLOSED
        assert not service.has_recovery_task()

    async def test_timer_cancelled_on_cancel_active_recovery(self) -> None:
        """Test the public cancel_active_recovery() method for shutdown paths."""
        reset_timeout = 1.0
        service = ActiveRecoveryTestService(threshold=1, reset_timeout=reset_timeout)

        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN
        assert service.has_recovery_task()

        # Cancel via public method
        await service.cancel_active_recovery()
        assert not service.has_recovery_task()

        # Circuit should still be OPEN (cancel does not change state)
        assert service.get_state() == EnumCircuitState.OPEN

    async def test_timer_cancelled_when_passive_check_transitions(self) -> None:
        """If a passive check beats the timer, the timer should be cancelled."""
        reset_timeout = 0.1
        service = ActiveRecoveryTestService(threshold=1, reset_timeout=reset_timeout)

        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN

        # Wait for timeout to elapse, then do passive check
        await asyncio.sleep(reset_timeout + 0.05)
        await service.check_circuit()

        # Passive check transitioned to HALF_OPEN, timer should be cleaned up
        assert service.get_state() == EnumCircuitState.HALF_OPEN
        assert not service.has_recovery_task()


# =============================================================================
# Tests: Timer Idempotency
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestActiveRecoveryIdempotency:
    """Test that opening the circuit multiple times does not create duplicate timers."""

    async def test_double_open_does_not_create_two_timers(self) -> None:
        """Record failures beyond threshold. Only one timer should exist."""
        reset_timeout = 1.0
        service = ActiveRecoveryTestService(threshold=2, reset_timeout=reset_timeout)

        # Open the circuit
        await service.record_failure()
        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN

        # Record additional failures while already open
        await service.record_failure()
        await service.record_failure()

        # Should still have exactly one recovery task
        assert service.has_recovery_task()
        task_ref = service._cb_recovery_task
        assert task_ref is not None

        # The task should be a single task, not multiple
        assert not task_ref.done()

    async def test_reopen_replaces_timer(self) -> None:
        """When circuit re-opens from HALF_OPEN, old timer is replaced by new one."""
        reset_timeout = 0.1
        service = ActiveRecoveryTestService(threshold=1, reset_timeout=reset_timeout)

        # First open
        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN
        first_task = service._cb_recovery_task

        # Wait for HALF_OPEN
        await asyncio.sleep(reset_timeout + 0.1)
        assert service.get_state() == EnumCircuitState.HALF_OPEN

        # The first task should be done
        assert first_task is not None
        assert first_task.done()

        # Re-open from HALF_OPEN
        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN

        # New timer should exist and be different from the first
        second_task = service._cb_recovery_task
        assert second_task is not None
        assert not second_task.done()
        assert second_task is not first_task


# =============================================================================
# Tests: Disabled Active Recovery
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestActiveRecoveryDisabled:
    """Test behavior when active recovery is disabled."""

    async def test_no_timer_when_disabled(self) -> None:
        """With enable_active_recovery=False, no background task is created."""
        service = ActiveRecoveryTestService(
            threshold=1, reset_timeout=0.1, enable_active_recovery=False
        )

        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN
        assert not service.has_recovery_task()

    async def test_no_auto_recovery_when_disabled(self) -> None:
        """With active recovery disabled, circuit stays OPEN past timeout."""
        reset_timeout = 0.1
        service = ActiveRecoveryTestService(
            threshold=1, reset_timeout=reset_timeout, enable_active_recovery=False
        )

        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN

        # Wait well past the reset timeout
        await asyncio.sleep(reset_timeout + 0.15)

        # Circuit should still be OPEN (no active recovery)
        assert service.get_state() == EnumCircuitState.OPEN

        # But passive check should still work
        await service.check_circuit()
        assert service.get_state() == EnumCircuitState.HALF_OPEN

    async def test_disabled_via_config_model(self) -> None:
        """Test that enable_active_recovery=False works via config model init."""
        from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

        config = ModelCircuitBreakerConfig(
            threshold=1,
            reset_timeout_seconds=0.1,
            service_name="test-disabled",
            transport_type=EnumInfraTransportType.HTTP,
            enable_active_recovery=False,
        )

        service = MixinAsyncCircuitBreaker()
        service._init_circuit_breaker_from_config(config)

        # Open the circuit
        async with service._circuit_breaker_lock:
            await service._record_circuit_failure("test_op")

        assert service._circuit_breaker_open is True
        assert service._cb_recovery_task is None


# =============================================================================
# Tests: Timer Cleanup
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestActiveRecoveryCleanup:
    """Test timer cleanup on close/shutdown scenarios."""

    async def test_cancel_active_recovery_is_safe_when_no_timer(self) -> None:
        """cancel_active_recovery() should be a no-op when no timer exists."""
        service = ActiveRecoveryTestService(threshold=5, reset_timeout=1.0)

        # No failures, no timer
        assert not service.has_recovery_task()

        # Should not raise
        await service.cancel_active_recovery()
        assert not service.has_recovery_task()

    async def test_cancel_active_recovery_is_safe_after_timer_completes(self) -> None:
        """cancel_active_recovery() should be safe after timer has already fired."""
        reset_timeout = 0.1
        service = ActiveRecoveryTestService(threshold=1, reset_timeout=reset_timeout)

        await service.record_failure()
        assert service.get_state() == EnumCircuitState.OPEN

        # Wait for timer to complete naturally
        await asyncio.sleep(reset_timeout + 0.15)
        assert service.get_state() == EnumCircuitState.HALF_OPEN

        # Cancel should be a no-op (timer already done)
        await service.cancel_active_recovery()
        assert not service.has_recovery_task()

    async def test_multiple_cancel_calls_are_safe(self) -> None:
        """Multiple calls to cancel_active_recovery() should be idempotent."""
        service = ActiveRecoveryTestService(threshold=1, reset_timeout=1.0)

        await service.record_failure()
        assert service.has_recovery_task()

        await service.cancel_active_recovery()
        await service.cancel_active_recovery()
        await service.cancel_active_recovery()

        assert not service.has_recovery_task()


__all__ = [
    "ActiveRecoveryTestService",
    "TestActiveRecoveryTransition",
    "TestActiveRecoveryCancellation",
    "TestActiveRecoveryIdempotency",
    "TestActiveRecoveryDisabled",
    "TestActiveRecoveryCleanup",
]
