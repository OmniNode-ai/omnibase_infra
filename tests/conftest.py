"""
Shared pytest fixtures and configuration for infrastructure tests.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_async_pool():
    """Create a mock asyncpg pool for testing."""
    pool = AsyncMock()
    pool.get_size = Mock(return_value=5)
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    pool.close = AsyncMock()
    return pool


@pytest.fixture
def mock_async_connection():
    """Create a mock asyncpg connection for testing."""
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchval = AsyncMock()
    conn.fetchrow = AsyncMock()
    return conn


@pytest.fixture
def mock_kafka_producer():
    """Create a mock Kafka producer for testing."""
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send = AsyncMock()
    return producer


@pytest.fixture
def mock_kafka_consumer():
    """Create a mock Kafka consumer for testing."""
    consumer = AsyncMock()
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    consumer.subscribe = AsyncMock()
    consumer.getmany = AsyncMock(return_value={})
    return consumer


@pytest.fixture(autouse=True)
def reset_structlog():
    """Reset structlog configuration between tests."""
    import structlog

    # Clear any existing configuration
    structlog.reset_defaults()

    yield

    # Clean up after test
    structlog.reset_defaults()


@pytest.fixture(autouse=True)
def reset_logger_factory():
    """Reset LoggerFactory between tests."""
    from omnibase_infra.infrastructure.observability.structured_logger import (
        LoggerFactory,
    )

    LoggerFactory.clear_cache()
    LoggerFactory._config = None

    yield

    LoggerFactory.clear_cache()
    LoggerFactory._config = None


@pytest.fixture
def mock_time():
    """Mock time.time and time.monotonic for consistent testing."""
    import time
    from unittest.mock import patch

    current_time = [1000.0]

    def mock_time_func():
        return current_time[0]

    def advance_time(seconds):
        current_time[0] += seconds

    with patch("time.time", side_effect=mock_time_func):
        with patch("time.monotonic", side_effect=mock_time_func):
            yield advance_time
