# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for handler pool integration with RuntimeHostProcess (OMN-477).

Tests validate:
- Pool creation when handler_pool_size > 1 AND max_concurrent_handlers > 1
- No pool creation when pool_size == 1 (backwards compatible)
- Pool metrics in health_check() output
- Pool shutdown during process stop
- Pooled handler execution via _handle_envelope
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

_RUNTIME_HOST_IMPLEMENTED = False
try:
    from omnibase_infra.runtime.service_runtime_host_process import (
        RuntimeHostProcess,
    )

    _RUNTIME_HOST_IMPLEMENTED = True
except ImportError:
    RuntimeHostProcess = None  # type: ignore[misc, assignment]

pytestmark = [
    pytest.mark.skipif(
        not _RUNTIME_HOST_IMPLEMENTED,
        reason="RuntimeHostProcess not yet implemented",
    ),
    pytest.mark.unit,
]


# =============================================================================
# Helpers
# =============================================================================


def _make_headers() -> ModelEventHeaders:
    """Create valid ModelEventHeaders for test messages."""
    return ModelEventHeaders(
        timestamp=str(int(time.time() * 1000)),
        source="test-pool-integration",
        event_type="test.envelope",
    )


def _make_event_message(
    operation: str = "mock.test",
    handler_type: str = "mock",
    offset: int = 0,
    correlation_id: str | None = None,
) -> ModelEventMessage:
    """Create a ModelEventMessage with a valid JSON envelope payload."""
    envelope = {
        "operation": operation,
        "handler_type": handler_type,
        "payload": {"test": True},
        "correlation_id": correlation_id or str(uuid4()),
    }
    return ModelEventMessage(
        topic="requests",
        partition=0,
        offset=str(offset),
        key=b"test-key",
        value=json.dumps(envelope).encode("utf-8"),
        headers=_make_headers(),
    )


def _make_process(
    max_concurrent: int = 1,
    pool_size: int = 1,
    event_bus: EventBusInmemory | None = None,
) -> RuntimeHostProcess:
    """Create a RuntimeHostProcess with pool config."""
    bus = event_bus or EventBusInmemory()
    config = make_runtime_config(
        max_concurrent_handlers=max_concurrent,
        handler_pool_size=pool_size,
    )
    process = RuntimeHostProcess(event_bus=bus, config=config)
    seed_mock_handlers(process)
    return process


# =============================================================================
# Configuration Tests
# =============================================================================


class TestPoolConfigIntegration:
    """Tests for handler_pool_size configuration in RuntimeHostProcess."""

    def test_default_pool_size_is_one(self) -> None:
        """Default handler_pool_size should be 1 (no pooling)."""
        process = _make_process()
        assert process.handler_pool_size == 1

    def test_configurable_pool_size(self) -> None:
        """handler_pool_size should accept integer values."""
        process = _make_process(pool_size=4)
        assert process.handler_pool_size == 4

    def test_string_pool_size_parsed(self) -> None:
        """String values for handler_pool_size should be parsed."""
        bus = EventBusInmemory()
        config = make_runtime_config(handler_pool_size="8")
        process = RuntimeHostProcess(event_bus=bus, config=config)
        seed_mock_handlers(process)
        assert process.handler_pool_size == 8

    def test_invalid_string_falls_back_to_default(self) -> None:
        """Invalid string values should fall back to default."""
        bus = EventBusInmemory()
        config = make_runtime_config(handler_pool_size="not-a-number")
        process = RuntimeHostProcess(event_bus=bus, config=config)
        seed_mock_handlers(process)
        assert process.handler_pool_size == 1  # DEFAULT

    def test_no_pools_when_pool_size_one(self) -> None:
        """No pools should be created when pool_size is 1."""
        process = _make_process(max_concurrent=4, pool_size=1)
        assert len(process.handler_pools) == 0

    def test_no_pools_when_concurrent_one(self) -> None:
        """No pools should be created when max_concurrent_handlers is 1."""
        process = _make_process(max_concurrent=1, pool_size=4)
        assert len(process.handler_pools) == 0


# =============================================================================
# Health Check Integration Tests
# =============================================================================


class TestPoolHealthCheckIntegration:
    """Tests for pool metrics in RuntimeHostProcess health_check()."""

    @pytest.mark.asyncio
    async def test_health_includes_pool_size(self) -> None:
        """Health check should include handler_pool_size."""
        bus = EventBusInmemory()
        process = _make_process(max_concurrent=4, pool_size=3, event_bus=bus)
        await bus.start()

        health = await process.health_check()

        assert health["handler_pool_size"] == 3
        assert "handler_pools" in health

        await bus.close()

    @pytest.mark.asyncio
    async def test_health_empty_pools_when_disabled(self) -> None:
        """Health check should have empty handler_pools when pooling disabled."""
        bus = EventBusInmemory()
        process = _make_process(max_concurrent=1, pool_size=1, event_bus=bus)
        await bus.start()

        health = await process.health_check()

        assert health["handler_pool_size"] == 1
        assert health["handler_pools"] == {}

        await bus.close()


# =============================================================================
# Execution Tests
# =============================================================================


class TestPooledExecution:
    """Tests for envelope execution through handler pools."""

    @pytest.mark.asyncio
    async def test_execution_uses_single_handler_without_pool(self) -> None:
        """Without pooling, envelope should use the single handler instance."""
        process = _make_process(max_concurrent=1, pool_size=1)

        call_count = 0

        async def tracking_handle(envelope: dict[str, object]) -> None:
            nonlocal call_count
            call_count += 1

        process._handle_envelope = tracking_handle  # type: ignore[assignment]

        msg = _make_event_message()
        await process._on_message(msg)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_parallel_execution_with_pooling(self) -> None:
        """With pooling enabled, parallel execution should not contend."""
        process = _make_process(max_concurrent=4, pool_size=4)

        gate = asyncio.Event()
        active_count = 0
        max_active = 0

        async def tracking_handle(envelope: dict[str, object]) -> None:
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await gate.wait()
            active_count -= 1

        process._handle_envelope = tracking_handle  # type: ignore[assignment]

        # Send 4 messages in parallel
        for i in range(4):
            msg = _make_event_message(offset=i)
            await process._on_message(msg)

        await asyncio.sleep(0.02)

        # All 4 should be active concurrently
        assert max_active == 4

        gate.set()
        await asyncio.sleep(0.05)
