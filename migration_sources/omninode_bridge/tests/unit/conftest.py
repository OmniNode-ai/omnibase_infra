"""
Fast unit test configuration - minimal fixtures for speed.

This configuration avoids:
- Container infrastructure loading
- Complex fixture chains
- Heavy dependency imports
- Coverage on entire codebase

Used only for unit tests that don't need integration infrastructure.
"""

import os
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def mock_kafka_client():
    """Fast mock KafkaClient for unit tests."""
    mock = AsyncMock()
    # Use environment-based configuration for consistency
    mock.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    mock._connected = False
    mock.producer = None
    return mock


@pytest.fixture
def mock_postgres_client():
    """Fast mock PostgresClient for unit tests."""
    mock = AsyncMock()
    mock.pool = None
    mock._connected = False
    return mock


@pytest.fixture
def mock_audit_logger():
    """Fast mock audit logger for unit tests."""
    mock = Mock()
    return mock


# Disable expensive auto-use fixtures from main conftest.py
@pytest.fixture(autouse=True, scope="session")
def skip_expensive_fixtures():
    """Skip expensive fixtures for unit tests."""
    pass
