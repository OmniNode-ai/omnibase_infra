# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for event bus performance tests.

Provides fixtures for performance testing including:
- Pre-configured event bus instances with various settings
- Sample event payloads and messages
- Latency measurement utilities
- Concurrent subscriber simulation

Usage:
    Fixtures are automatically available to all tests in this package.

Supported Event Bus Implementations:
    The ONEX infrastructure supports multiple event bus implementations:
    - InMemoryEventBus: Used for unit and performance tests (this module)
    - KafkaEventBus: Used for integration and E2E tests with real Kafka/Redpanda

    This module uses InMemoryEventBus for deterministic performance benchmarking.
    For Kafka-based testing, see tests/integration/event_bus/conftest.py and
    tests/integration/registration/e2e/conftest.py.

Related:
    - OMN-57: Event bus performance testing requirements
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

# -----------------------------------------------------------------------------
# Module-Level Markers
# -----------------------------------------------------------------------------

pytestmark = [
    pytest.mark.performance,
]

# -----------------------------------------------------------------------------
# Event Bus Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
async def event_bus() -> AsyncGenerator[InMemoryEventBus, None]:
    """Create and start an InMemoryEventBus for testing.

    Yields:
        Started InMemoryEventBus instance.
    """
    bus = InMemoryEventBus(
        environment="perf-test",
        group="benchmark",
        max_history=10000,
    )
    await bus.start()
    yield bus
    await bus.close()


@pytest.fixture
async def high_volume_event_bus() -> AsyncGenerator[InMemoryEventBus, None]:
    """Create InMemoryEventBus with high history capacity for volume testing.

    Yields:
        InMemoryEventBus with 100k history capacity.
    """
    bus = InMemoryEventBus(
        environment="high-volume",
        group="stress-test",
        max_history=100000,
    )
    await bus.start()
    yield bus
    await bus.close()


@pytest.fixture
async def low_latency_event_bus() -> AsyncGenerator[InMemoryEventBus, None]:
    """Create InMemoryEventBus optimized for low latency testing.

    Yields:
        InMemoryEventBus with minimal history for lower overhead.
    """
    bus = InMemoryEventBus(
        environment="low-latency",
        group="latency-test",
        max_history=100,  # Small history for minimal overhead
    )
    await bus.start()
    yield bus
    await bus.close()


# -----------------------------------------------------------------------------
# Message Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_message_bytes() -> bytes:
    """Create sample message payload as bytes.

    Returns:
        Sample JSON-encoded message bytes.
    """
    return b'{"event_type": "test_event", "data": {"key": "value", "count": 42}}'


@pytest.fixture
def large_message_bytes() -> bytes:
    """Create a larger message payload for stress testing.

    Returns:
        1KB message payload.
    """
    # Create ~1KB payload
    data = "x" * 1000
    return f'{{"event_type": "large_event", "data": "{data}"}}'.encode()


@pytest.fixture
def sample_headers() -> ModelEventHeaders:
    """Create sample event headers.

    Returns:
        ModelEventHeaders configured for testing.
    """
    return ModelEventHeaders(
        source="perf-test",
        event_type="benchmark_event",
        priority="normal",
        content_type="application/json",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
    )


# -----------------------------------------------------------------------------
# Subscriber Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def counting_handler() -> tuple[
    Callable[[ModelEventMessage], Awaitable[None]],
    Callable[[], int],
]:
    """Create a handler that counts received messages.

    Returns:
        Tuple of (handler_callback, get_count_function).
    """
    count = 0
    lock = asyncio.Lock()

    async def handler(msg: ModelEventMessage) -> None:
        nonlocal count
        async with lock:
            count += 1

    def get_count() -> int:
        return count

    return handler, get_count


@pytest.fixture
def latency_tracking_handler() -> tuple[
    Callable[[ModelEventMessage], Awaitable[None]],
    Callable[[], list[float]],
]:
    """Create a handler that tracks message receipt timestamps.

    Returns:
        Tuple of (handler_callback, get_timestamps_function).
    """
    timestamps: list[float] = []
    lock = asyncio.Lock()

    async def handler(msg: ModelEventMessage) -> None:
        receipt_time = time.perf_counter()
        async with lock:
            timestamps.append(receipt_time)

    def get_timestamps() -> list[float]:
        return timestamps.copy()

    return handler, get_timestamps


@pytest.fixture
def slow_handler() -> Callable[[ModelEventMessage], Awaitable[None]]:
    """Create a handler with artificial delay for backpressure testing.

    Returns:
        Handler that sleeps for 1ms per message.
    """

    async def handler(msg: ModelEventMessage) -> None:
        await asyncio.sleep(0.001)  # 1ms delay

    return handler


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def generate_unique_topic() -> str:
    """Generate a unique topic name for test isolation.

    Returns:
        Unique topic string.
    """
    return f"perf-test.{uuid4().hex[:8]}"


def generate_batch_messages(count: int, topic: str) -> list[tuple[str, bytes, bytes]]:
    """Generate a batch of test messages.

    Args:
        count: Number of messages to generate.
        topic: Topic name for all messages.

    Returns:
        List of (topic, key, value) tuples.
    """
    return [
        (topic, f"key-{i}".encode(), f'{{"index": {i}}}'.encode()) for i in range(count)
    ]
