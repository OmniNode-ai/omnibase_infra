# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for parallel handler execution in RuntimeHostProcess (OMN-476).

Tests validate:
- Configurable concurrency limit via max_concurrent_handlers
- Backpressure via asyncio.Semaphore when queue is full
- Error isolation between parallel handlers
- Correlation ID tracking across parallel executions
- Sequential backwards compatibility when max_concurrent_handlers=1
- Graceful drain of in-flight parallel tasks
- Health check reporting of concurrency metrics
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

# Try to import RuntimeHostProcess
_RUNTIME_HOST_IMPLEMENTED = False
try:
    from omnibase_infra.runtime.service_runtime_host_process import (
        DEFAULT_MAX_CONCURRENT_HANDLERS,
        MAX_MAX_CONCURRENT_HANDLERS,
        MIN_MAX_CONCURRENT_HANDLERS,
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
# Helper functions
# =============================================================================


def _make_headers() -> ModelEventHeaders:
    """Create valid ModelEventHeaders for test messages."""
    return ModelEventHeaders(
        timestamp=str(int(time.time() * 1000)),
        source="test-parallel",
        event_type="test.envelope",
    )


def _make_event_message(
    operation: str = "mock.test",
    handler_type: str = "mock",
    topic: str = "requests",
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
        topic=topic,
        partition=0,
        offset=str(offset),
        key=b"test-key",
        value=json.dumps(envelope).encode("utf-8"),
        headers=_make_headers(),
    )


def _make_process(
    max_concurrent: int = 1,
    event_bus: EventBusInmemory | None = None,
    **config_overrides: object,
) -> RuntimeHostProcess:
    """Create a RuntimeHostProcess with parallel handler config."""
    bus = event_bus or EventBusInmemory()
    config = make_runtime_config(
        max_concurrent_handlers=max_concurrent, **config_overrides
    )
    process = RuntimeHostProcess(event_bus=bus, config=config)
    seed_mock_handlers(process)
    return process


# =============================================================================
# Configuration Tests
# =============================================================================


class TestParallelHandlerConfig:
    """Tests for max_concurrent_handlers configuration."""

    def test_default_concurrency_is_one(self) -> None:
        """Default should be 1 (sequential) for backwards compatibility."""
        process = _make_process()
        assert process.max_concurrent_handlers == 1

    def test_configurable_concurrency(self) -> None:
        """max_concurrent_handlers should accept integer values."""
        process = _make_process(max_concurrent=10)
        assert process.max_concurrent_handlers == 10

    def test_string_config_parsed(self) -> None:
        """String values for max_concurrent_handlers should be parsed."""
        bus = EventBusInmemory()
        config = make_runtime_config(max_concurrent_handlers="8")
        process = RuntimeHostProcess(event_bus=bus, config=config)
        seed_mock_handlers(process)
        assert process.max_concurrent_handlers == 8

    def test_invalid_string_falls_back_to_default(self) -> None:
        """Invalid string values should fall back to default."""
        bus = EventBusInmemory()
        config = make_runtime_config(max_concurrent_handlers="not-a-number")
        process = RuntimeHostProcess(event_bus=bus, config=config)
        seed_mock_handlers(process)
        assert process.max_concurrent_handlers == DEFAULT_MAX_CONCURRENT_HANDLERS

    def test_zero_clamped_to_minimum(self) -> None:
        """Values below minimum should be clamped to 1."""
        process = _make_process(max_concurrent=0)
        assert process.max_concurrent_handlers == MIN_MAX_CONCURRENT_HANDLERS

    def test_negative_clamped_to_minimum(self) -> None:
        """Negative values should be clamped to 1."""
        process = _make_process(max_concurrent=-5)
        assert process.max_concurrent_handlers == MIN_MAX_CONCURRENT_HANDLERS

    def test_exceeds_max_clamped(self) -> None:
        """Values above maximum should be clamped."""
        process = _make_process(max_concurrent=999)
        assert process.max_concurrent_handlers == MAX_MAX_CONCURRENT_HANDLERS


# =============================================================================
# Sequential Backwards Compatibility Tests
# =============================================================================


class TestSequentialBackwardsCompat:
    """Tests that max_concurrent_handlers=1 preserves sequential behavior."""

    @pytest.mark.asyncio
    async def test_sequential_processes_in_order(self) -> None:
        """With concurrency=1, messages should be processed sequentially."""
        process = _make_process(max_concurrent=1)

        order: list[int] = []

        async def tracking_handle(envelope: dict[str, object]) -> None:
            payload = envelope.get("payload", {})
            idx = payload.get("idx", -1) if isinstance(payload, dict) else -1
            order.append(idx)

        process._handle_envelope = tracking_handle  # type: ignore[assignment]

        for i in range(2):
            envelope = {
                "operation": "mock.test",
                "handler_type": "mock",
                "payload": {"idx": i},
                "correlation_id": str(uuid4()),
            }
            msg = ModelEventMessage(
                topic="requests",
                partition=0,
                offset=str(i),
                key=b"key",
                value=json.dumps(envelope).encode("utf-8"),
                headers=_make_headers(),
            )
            await process._on_message(msg)

        # Sequential: both processed immediately in order
        assert order == [0, 1]

    @pytest.mark.asyncio
    async def test_no_tasks_created_in_sequential_mode(self) -> None:
        """Sequential mode should not create asyncio tasks."""
        process = _make_process(max_concurrent=1)
        process._handle_envelope = AsyncMock()  # type: ignore[assignment]
        msg = _make_event_message()
        await process._on_message(msg)
        assert process.in_flight_task_count == 0


# =============================================================================
# Parallel Execution Tests
# =============================================================================


class TestParallelExecution:
    """Tests for parallel handler execution with max_concurrent_handlers > 1."""

    @pytest.mark.asyncio
    async def test_parallel_dispatches_tasks(self) -> None:
        """With concurrency > 1, messages should be dispatched as tasks."""
        process = _make_process(max_concurrent=4)

        gate = asyncio.Event()
        call_count = 0

        async def slow_handle(envelope: dict[str, object]) -> None:
            nonlocal call_count
            call_count += 1
            await gate.wait()

        process._handle_envelope = slow_handle  # type: ignore[assignment]

        for i in range(3):
            msg = _make_event_message(offset=i)
            await process._on_message(msg)

        # Yield to event loop so tasks start executing
        await asyncio.sleep(0.01)

        # All 3 should be dispatched as tasks (not yet completed, blocked on gate)
        assert process.in_flight_task_count == 3
        assert call_count == 3  # All started concurrently

        gate.set()
        await asyncio.sleep(0.05)
        assert process.in_flight_task_count == 0

    @pytest.mark.asyncio
    async def test_backpressure_blocks_at_limit(self) -> None:
        """When concurrency limit is reached, new messages should block."""
        process = _make_process(max_concurrent=2)

        gate = asyncio.Event()
        started_count = 0

        async def blocking_handle(envelope: dict[str, object]) -> None:
            nonlocal started_count
            started_count += 1
            await gate.wait()

        process._handle_envelope = blocking_handle  # type: ignore[assignment]

        # Send 2 messages - should fill the semaphore
        for i in range(2):
            msg = _make_event_message(offset=i)
            await process._on_message(msg)

        # Yield to let tasks start
        await asyncio.sleep(0.01)

        assert started_count == 2
        assert process.in_flight_task_count == 2

        # Third message should block (semaphore full)
        third_msg = _make_event_message(offset=2)
        blocked_task = asyncio.create_task(process._on_message(third_msg))

        # Give the event loop a chance to process
        await asyncio.sleep(0.02)
        # The third message's handler shouldn't have started yet
        assert started_count == 2

        # Release all slots
        gate.set()
        await asyncio.sleep(0.05)

        await blocked_task
        assert started_count == 3

    @pytest.mark.asyncio
    async def test_error_isolation(self) -> None:
        """Error in one parallel handler should not affect others."""
        process = _make_process(max_concurrent=4)

        gate = asyncio.Event()
        results: list[str] = []

        async def handler_that_may_fail(envelope: dict[str, object]) -> None:
            idx = envelope.get("payload", {}).get("idx", 0)  # type: ignore[union-attr]
            if idx == 1:
                raise RuntimeError("Handler 1 failed!")
            await gate.wait()
            results.append(f"done-{idx}")

        process._handle_envelope = handler_that_may_fail  # type: ignore[assignment]

        # Send 3 messages - message 1 will fail
        for i in range(3):
            envelope = {
                "operation": "mock.test",
                "handler_type": "mock",
                "payload": {"idx": i},
                "correlation_id": str(uuid4()),
            }
            msg = ModelEventMessage(
                topic="requests",
                partition=0,
                offset=str(i),
                key=b"key",
                value=json.dumps(envelope).encode("utf-8"),
                headers=_make_headers(),
            )
            await process._on_message(msg)

        # Yield to let tasks start and the failing one to error
        await asyncio.sleep(0.02)

        # Release gate for remaining tasks
        gate.set()
        await asyncio.sleep(0.05)

        # Messages 0 and 2 should complete successfully despite message 1 failing
        assert "done-0" in results
        assert "done-2" in results
        assert process.in_flight_task_count == 0

    @pytest.mark.asyncio
    async def test_correlation_ids_isolated(self) -> None:
        """Each parallel handler should track its own correlation ID."""
        process = _make_process(max_concurrent=4)

        seen_ids: list[str] = []
        gate = asyncio.Event()

        async def tracking_handle(envelope: dict[str, object]) -> None:
            cid = str(envelope.get("correlation_id", ""))
            seen_ids.append(cid)
            await gate.wait()

        process._handle_envelope = tracking_handle  # type: ignore[assignment]

        # Send messages with distinct correlation IDs
        expected_ids = []
        for i in range(3):
            cid = str(uuid4())
            expected_ids.append(cid)
            msg = _make_event_message(offset=i, correlation_id=cid)
            await process._on_message(msg)

        # Yield to let tasks start
        await asyncio.sleep(0.02)

        # All 3 correlation IDs should be tracked independently
        assert len(seen_ids) == 3
        for eid in expected_ids:
            assert eid in seen_ids

        gate.set()
        await asyncio.sleep(0.05)


# =============================================================================
# Drain Tests
# =============================================================================


class TestParallelDrain:
    """Tests for graceful drain of in-flight parallel tasks."""

    @pytest.mark.asyncio
    async def test_drain_waits_for_tasks(self) -> None:
        """drain_in_flight_tasks should wait for all tasks to complete."""
        process = _make_process(max_concurrent=4)

        gate = asyncio.Event()

        async def slow_handle(envelope: dict[str, object]) -> None:
            await gate.wait()

        process._handle_envelope = slow_handle  # type: ignore[assignment]

        # Send messages
        for i in range(3):
            msg = _make_event_message(offset=i)
            await process._on_message(msg)

        await asyncio.sleep(0.01)
        assert process.in_flight_task_count == 3

        # Start drain in background
        drain_task = asyncio.create_task(process.drain_in_flight_tasks(timeout=5.0))

        # Release tasks
        await asyncio.sleep(0.01)
        gate.set()

        count = await drain_task
        assert count == 3
        assert process.in_flight_task_count == 0

    @pytest.mark.asyncio
    async def test_drain_timeout_cancels_tasks(self) -> None:
        """Tasks exceeding drain timeout should be cancelled."""
        process = _make_process(max_concurrent=4)

        async def never_finish(envelope: dict[str, object]) -> None:
            await asyncio.sleep(999)

        process._handle_envelope = never_finish  # type: ignore[assignment]

        msg = _make_event_message()
        await process._on_message(msg)
        await asyncio.sleep(0.01)

        assert process.in_flight_task_count == 1

        # Drain with very short timeout
        count = await process.drain_in_flight_tasks(timeout=0.05)
        assert count == 1

        # Task should be cancelled
        await asyncio.sleep(0.02)
        assert process.in_flight_task_count == 0

    @pytest.mark.asyncio
    async def test_drain_empty_returns_zero(self) -> None:
        """Drain with no in-flight tasks should return 0 immediately."""
        process = _make_process(max_concurrent=4)
        count = await process.drain_in_flight_tasks()
        assert count == 0


# =============================================================================
# Health Check Tests
# =============================================================================


class TestParallelHealthCheck:
    """Tests for health check reporting of concurrency metrics."""

    @pytest.mark.asyncio
    async def test_health_includes_concurrency_info(self) -> None:
        """Health check should report max_concurrent_handlers and in_flight_tasks."""
        bus = EventBusInmemory()
        process = _make_process(max_concurrent=8, event_bus=bus)
        await bus.start()

        health = await process.health_check()

        assert health["max_concurrent_handlers"] == 8
        assert health["in_flight_tasks"] == 0

        await bus.close()

    @pytest.mark.asyncio
    async def test_health_reports_in_flight_count(self) -> None:
        """Health check should reflect current in-flight task count."""
        bus = EventBusInmemory()
        await bus.start()
        process = _make_process(max_concurrent=4, event_bus=bus)

        gate = asyncio.Event()

        async def slow_handle(envelope: dict[str, object]) -> None:
            await gate.wait()

        process._handle_envelope = slow_handle  # type: ignore[assignment]

        msg = _make_event_message()
        await process._on_message(msg)
        await asyncio.sleep(0.01)

        health = await process.health_check()
        assert health["in_flight_tasks"] == 1

        gate.set()
        await asyncio.sleep(0.05)

        health = await process.health_check()
        assert health["in_flight_tasks"] == 0

        await bus.close()


# =============================================================================
# Property Tests
# =============================================================================


class TestParallelProperties:
    """Tests for public properties related to parallel execution."""

    def test_max_concurrent_handlers_property(self) -> None:
        """max_concurrent_handlers property should reflect config."""
        process = _make_process(max_concurrent=16)
        assert process.max_concurrent_handlers == 16

    def test_in_flight_task_count_initially_zero(self) -> None:
        """in_flight_task_count should be 0 when no tasks are running."""
        process = _make_process(max_concurrent=4)
        assert process.in_flight_task_count == 0
