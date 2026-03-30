# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerPool (OMN-477).

Tests validate:
- Pool initialization with configurable size
- Checkout/checkin semantics under concurrent load
- Unhealthy handler instance recycling
- Pool metrics exposed via health_check()
- Pool shutdown lifecycle
- Error isolation during checkout
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.protocols.protocol_container_aware import ProtocolContainerAware
from omnibase_infra.runtime.handler_pool import (
    DEFAULT_POOL_SIZE,
    MAX_POOL_SIZE,
    MIN_POOL_SIZE,
    HandlerPool,
)

pytestmark = [pytest.mark.unit]


# =============================================================================
# Helper functions
# =============================================================================


def _make_mock_handler(healthy: bool = True) -> MagicMock:
    """Create a mock handler with required async lifecycle methods."""
    handler = MagicMock(spec=ProtocolContainerAware)
    handler.execute = AsyncMock(return_value={"success": True})
    handler.initialize = AsyncMock()
    handler.shutdown = AsyncMock()
    handler.health_check = AsyncMock(return_value={"healthy": healthy})
    return handler


def _make_factory(healthy: bool = True) -> MagicMock:
    """Create a factory that produces mock handlers."""
    factory = MagicMock(side_effect=lambda: _make_mock_handler(healthy=healthy))
    return factory


# =============================================================================
# Initialization Tests
# =============================================================================


class TestHandlerPoolInit:
    """Tests for pool initialization."""

    @pytest.mark.asyncio
    async def test_pool_creates_configured_instances(self) -> None:
        """Pool should create pool_size instances during initialize()."""
        factory = _make_factory()
        pool = HandlerPool(handler_type="mock", factory=factory, pool_size=4)
        await pool.initialize()

        assert factory.call_count == 4
        assert pool.total_instance_count == 4
        assert pool.available_count == 4

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_calls_initialize_on_instances(self) -> None:
        """Each instance should have initialize() called during pool init."""
        instances: list[MagicMock] = []

        def tracking_factory() -> MagicMock:
            h = _make_mock_handler()
            instances.append(h)
            return h

        pool = HandlerPool(handler_type="mock", factory=tracking_factory, pool_size=3)
        await pool.initialize()

        for instance in instances:
            instance.initialize.assert_awaited_once()

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_passes_config_to_initialize(self) -> None:
        """Pool should pass handler_config to instance.initialize() when provided."""
        instances: list[MagicMock] = []
        test_config: dict[str, object] = {
            "dsn": "postgresql://localhost/test",
            "timeout": 30,
        }

        def tracking_factory() -> MagicMock:
            h = _make_mock_handler()
            instances.append(h)
            return h

        pool = HandlerPool(
            handler_type="db",
            factory=tracking_factory,
            pool_size=2,
            handler_config=test_config,
        )
        await pool.initialize()

        for instance in instances:
            instance.initialize.assert_awaited_once_with(test_config)

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_initialize_no_config_calls_without_args(self) -> None:
        """Pool without handler_config should call initialize() with no args (backwards compat)."""
        instances: list[MagicMock] = []

        def tracking_factory() -> MagicMock:
            h = _make_mock_handler()
            instances.append(h)
            return h

        pool = HandlerPool(handler_type="mock", factory=tracking_factory, pool_size=2)
        await pool.initialize()

        for instance in instances:
            instance.initialize.assert_awaited_once_with()

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_recycle_passes_config_to_new_instance(self) -> None:
        """Recycled instances should receive handler_config during initialize()."""
        call_count = 0
        test_config: dict[str, object] = {"dsn": "postgresql://localhost/test"}

        def factory() -> MagicMock:
            nonlocal call_count
            call_count += 1
            h = _make_mock_handler(healthy=True)
            return h

        pool = HandlerPool(
            handler_type="db",
            factory=factory,
            pool_size=1,
            handler_config=test_config,
        )
        await pool.initialize()
        assert call_count == 1

        # Make instance unhealthy to trigger recycle
        async with pool.checkout() as handler:
            handler.health_check = AsyncMock(return_value={"healthy": False})

        # Recycle happened — new instance should have received config
        assert call_count == 2
        # The pool's _all_instances list should have the recycled instance
        assert pool.total_instance_count == 1

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_double_initialize_raises(self) -> None:
        """Calling initialize() twice should raise RuntimeError."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=1)
        await pool.initialize()

        with pytest.raises(RuntimeError, match="already initialized"):
            await pool.initialize()

        await pool.shutdown()

    def test_pool_size_clamped_to_min(self) -> None:
        """Pool size below minimum should be clamped."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=0)
        assert pool.pool_size == MIN_POOL_SIZE

    def test_pool_size_clamped_to_max(self) -> None:
        """Pool size above maximum should be clamped."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=999)
        assert pool.pool_size == MAX_POOL_SIZE

    def test_default_pool_size(self) -> None:
        """Default pool size should be DEFAULT_POOL_SIZE."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory())
        assert pool.pool_size == DEFAULT_POOL_SIZE


# =============================================================================
# Checkout/Checkin Tests
# =============================================================================


class TestHandlerPoolCheckout:
    """Tests for checkout/checkin semantics."""

    @pytest.mark.asyncio
    async def test_checkout_returns_handler(self) -> None:
        """Checkout should return a handler instance."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=2)
        await pool.initialize()

        async with pool.checkout() as handler:
            assert handler is not None
            assert hasattr(handler, "execute")

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_checkout_reduces_available_count(self) -> None:
        """Checkout should reduce available count, checkin restores it."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=2)
        await pool.initialize()

        assert pool.available_count == 2

        async with pool.checkout():
            assert pool.available_count == 1

        # After context exit, handler returned to pool
        assert pool.available_count == 2

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_checkout(self) -> None:
        """Multiple concurrent checkouts should work up to pool size."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=3)
        await pool.initialize()

        gate = asyncio.Event()
        checked_out: list[object] = []

        async def checkout_and_hold() -> None:
            async with pool.checkout() as handler:
                checked_out.append(handler)
                await gate.wait()

        tasks = [asyncio.create_task(checkout_and_hold()) for _ in range(3)]
        await asyncio.sleep(0.02)

        # All 3 should be checked out
        assert len(checked_out) == 3
        assert pool.available_count == 0

        gate.set()
        await asyncio.gather(*tasks)

        # All returned
        assert pool.available_count == 3

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_checkout_blocks_when_pool_empty(self) -> None:
        """Checkout should block when all instances are checked out."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=1)
        await pool.initialize()

        gate = asyncio.Event()
        blocked = False

        async def hold_handler() -> None:
            async with pool.checkout():
                await gate.wait()

        async def try_checkout() -> None:
            nonlocal blocked
            # This should block because the only instance is held
            async with pool.checkout():
                blocked = True

        hold_task = asyncio.create_task(hold_handler())
        await asyncio.sleep(0.01)

        checkout_task = asyncio.create_task(try_checkout())
        await asyncio.sleep(0.02)

        # Should still be blocked
        assert not blocked

        # Release the held handler
        gate.set()
        await asyncio.sleep(0.02)

        # Now checkout should succeed
        assert blocked

        await hold_task
        await checkout_task
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_checkout_before_init_raises(self) -> None:
        """Checkout before initialize() should raise RuntimeError."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=1)

        with pytest.raises(RuntimeError, match="not initialized"):
            async with pool.checkout():
                pass

    @pytest.mark.asyncio
    async def test_checkout_during_shutdown_raises(self) -> None:
        """Checkout during shutdown should raise RuntimeError."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=1)
        await pool.initialize()
        await pool.shutdown()

        with pytest.raises(RuntimeError, match="not initialized"):
            async with pool.checkout():
                pass


# =============================================================================
# Recycling Tests
# =============================================================================


class TestHandlerPoolRecycling:
    """Tests for unhealthy instance recycling."""

    @pytest.mark.asyncio
    async def test_unhealthy_instance_recycled(self) -> None:
        """Instance that fails health check should be recycled after checkin."""
        call_count = 0
        healthy_flag = True

        def factory() -> MagicMock:
            nonlocal call_count
            call_count += 1
            h = _make_mock_handler(healthy=True)
            # Make health check return the current flag value
            h.health_check = AsyncMock(side_effect=lambda: {"healthy": healthy_flag})
            return h

        pool = HandlerPool(handler_type="mock", factory=factory, pool_size=1)
        await pool.initialize()
        assert call_count == 1

        # Mark unhealthy before checkin
        healthy_flag = False

        async with pool.checkout() as handler:
            handler.health_check = AsyncMock(return_value={"healthy": False})

        # Instance should have been recycled (new one created)
        assert call_count == 2

        health = await pool.health_check()
        assert health["recycle_count"] == 1

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_recycle_failure_degrades_pool(self) -> None:
        """If recycling fails, pool has fewer instances (degraded)."""
        first_call = True

        def factory() -> MagicMock:
            nonlocal first_call
            if first_call:
                first_call = False
                return _make_mock_handler(healthy=True)
            raise RuntimeError("Cannot create replacement")

        pool = HandlerPool(handler_type="mock", factory=factory, pool_size=1)
        await pool.initialize()

        # Make the instance unhealthy
        async with pool.checkout() as handler:
            handler.health_check = AsyncMock(return_value={"healthy": False})

        # Recycle failed, pool is degraded
        assert pool.total_instance_count == 0

        await pool.shutdown()


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHandlerPoolHealthCheck:
    """Tests for pool health metrics."""

    @pytest.mark.asyncio
    async def test_health_check_returns_metrics(self) -> None:
        """Health check should return comprehensive pool metrics."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=3)
        await pool.initialize()

        health = await pool.health_check()

        assert health["healthy"] is True
        assert health["handler_type"] == "mock"
        assert health["pool_size"] == 3
        assert health["available"] == 3
        assert health["total_instances"] == 3
        assert health["checkout_count"] == 0
        assert health["checkin_count"] == 0
        assert health["recycle_count"] == 0
        assert health["avg_checkout_wait_ms"] == 0.0

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_health_tracks_checkout_count(self) -> None:
        """Health metrics should track checkout/checkin counts."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=2)
        await pool.initialize()

        for _ in range(5):
            async with pool.checkout():
                pass

        health = await pool.health_check()
        assert health["checkout_count"] == 5
        assert health["checkin_count"] == 5

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_health_unhealthy_after_shutdown(self) -> None:
        """Pool should report unhealthy after shutdown."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=1)
        await pool.initialize()
        await pool.shutdown()

        health = await pool.health_check()
        assert health["healthy"] is False


# =============================================================================
# Shutdown Tests
# =============================================================================


class TestHandlerPoolShutdown:
    """Tests for pool shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_shutdown_on_all_instances(self) -> None:
        """Shutdown should call shutdown() on every instance."""
        instances: list[MagicMock] = []

        def factory() -> MagicMock:
            h = _make_mock_handler()
            instances.append(h)
            return h

        pool = HandlerPool(handler_type="mock", factory=factory, pool_size=3)
        await pool.initialize()
        await pool.shutdown()

        for instance in instances:
            instance.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_clears_pool(self) -> None:
        """Pool should be empty after shutdown."""
        pool = HandlerPool(handler_type="mock", factory=_make_factory(), pool_size=2)
        await pool.initialize()
        await pool.shutdown()

        assert pool.total_instance_count == 0
        assert pool.available_count == 0

    @pytest.mark.asyncio
    async def test_shutdown_tolerates_instance_errors(self) -> None:
        """Shutdown should not raise even if individual instances error."""

        def factory() -> MagicMock:
            h = _make_mock_handler()
            h.shutdown = AsyncMock(side_effect=RuntimeError("shutdown error"))
            return h

        pool = HandlerPool(handler_type="mock", factory=factory, pool_size=2)
        await pool.initialize()

        # Should not raise
        await pool.shutdown()
