#!/usr/bin/env python3
"""
Pytest fixtures for NodeDistributedLockEffect tests.

Provides fixtures for:
- Mock PostgreSQL connection pool
- Node instances
- Contract creation
- Test data
"""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

from ..models.enum_lock_operation import EnumLockOperation
from ..models.model_config import ModelDistributedLockConfig
from ..models.model_request import ModelDistributedLockRequest
from ..node import NodeDistributedLockEffect


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_postgres_pool():
    """Create mock PostgreSQL connection pool."""
    pool = AsyncMock()

    # Mock acquire context manager
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    pool.acquire.return_value.__aexit__.return_value = None

    # Mock transaction context manager
    conn.transaction.return_value.__aenter__.return_value = None
    conn.transaction.return_value.__aexit__.return_value = None

    # Mock fetchval for advisory lock
    conn.fetchval.return_value = None

    # Mock fetchrow for lock queries
    conn.fetchrow.return_value = None

    # Mock execute for lock operations
    conn.execute.return_value = "UPDATE 1"

    # Mock fetch for cleanup queries
    conn.fetch.return_value = []

    return pool


@pytest.fixture
def config():
    """Create test configuration."""
    return ModelDistributedLockConfig(
        postgres_host="localhost",
        postgres_port=5432,
        postgres_database="test_db",
        postgres_user="test_user",
        postgres_password="test_password",
        postgres_min_connections=2,
        postgres_max_connections=5,
        default_lease_duration=30.0,
        cleanup_interval=60.0,
        max_acquire_attempts=3,
        acquire_retry_delay=0.1,
    )


@pytest.fixture
def container(config):
    """Create ONEX container with test configuration."""
    return ModelContainer(
        value=config.model_dump(),
        container_type="config",
    )


@pytest.fixture
async def node(
    container, mock_postgres_pool, monkeypatch
) -> AsyncGenerator[NodeDistributedLockEffect, None]:
    """
    Create NodeDistributedLockEffect instance with mocked PostgreSQL pool.

    Patches asyncpg.create_pool to return mock pool.
    """

    # Patch asyncpg.create_pool
    async def mock_create_pool(*args, **kwargs):
        return mock_postgres_pool

    monkeypatch.setattr("asyncpg.create_pool", mock_create_pool)

    # Create node
    node_instance = NodeDistributedLockEffect(container)
    await node_instance.initialize()

    yield node_instance

    # Cleanup
    await node_instance.shutdown()


@pytest.fixture
def correlation_id():
    """Generate test correlation ID."""
    return uuid4()


@pytest.fixture
def execution_id():
    """Generate test execution ID."""
    return uuid4()


@pytest.fixture
def lock_request(correlation_id, execution_id):
    """Create test lock request."""
    return ModelDistributedLockRequest(
        operation=EnumLockOperation.ACQUIRE,
        lock_name="test_lock",
        owner_id="test_owner",
        lease_duration=30.0,
        correlation_id=correlation_id,
        execution_id=execution_id,
    )


@pytest.fixture
def acquire_contract(correlation_id):
    """Create test contract for lock acquisition."""
    return ModelContractEffect(
        name="acquire_lock",
        version={"major": 1, "minor": 0, "patch": 0},
        description="Acquire distributed lock",
        node_type="EFFECT",
        input_model="ModelDistributedLockRequest",
        output_model="ModelDistributedLockResponse",
        input_state={
            "operation": "acquire",
            "lock_name": "test_lock",
            "owner_id": "test_owner",
            "lease_duration": 30.0,
        },
        correlation_id=correlation_id,
    )


@pytest.fixture
def release_contract(correlation_id):
    """Create test contract for lock release."""
    return ModelContractEffect(
        name="release_lock",
        version={"major": 1, "minor": 0, "patch": 0},
        description="Release distributed lock",
        node_type="EFFECT",
        input_model="ModelDistributedLockRequest",
        output_model="ModelDistributedLockResponse",
        input_state={
            "operation": "release",
            "lock_name": "test_lock",
            "owner_id": "test_owner",
        },
        correlation_id=correlation_id,
    )


@pytest.fixture
def extend_contract(correlation_id):
    """Create test contract for lock extension."""
    return ModelContractEffect(
        name="extend_lock",
        version={"major": 1, "minor": 0, "patch": 0},
        description="Extend lock lease",
        node_type="EFFECT",
        input_model="ModelDistributedLockRequest",
        output_model="ModelDistributedLockResponse",
        input_state={
            "operation": "extend",
            "lock_name": "test_lock",
            "owner_id": "test_owner",
            "extension_duration": 30.0,
        },
        correlation_id=correlation_id,
    )


@pytest.fixture
def query_contract(correlation_id):
    """Create test contract for lock query."""
    return ModelContractEffect(
        name="query_lock",
        version={"major": 1, "minor": 0, "patch": 0},
        description="Query lock status",
        node_type="EFFECT",
        input_model="ModelDistributedLockRequest",
        output_model="ModelDistributedLockResponse",
        input_state={
            "operation": "query",
            "lock_name": "test_lock",
        },
        correlation_id=correlation_id,
    )


@pytest.fixture
def cleanup_contract(correlation_id):
    """Create test contract for cleanup."""
    return ModelContractEffect(
        name="cleanup_locks",
        version={"major": 1, "minor": 0, "patch": 0},
        description="Clean up expired locks",
        node_type="EFFECT",
        input_model="ModelDistributedLockRequest",
        output_model="ModelDistributedLockResponse",
        input_state={
            "operation": "cleanup",
            "lock_name": "*",
            "max_age_seconds": 3600.0,
        },
        correlation_id=correlation_id,
    )


__all__ = [
    "event_loop",
    "mock_postgres_pool",
    "config",
    "container",
    "node",
    "correlation_id",
    "execution_id",
    "lock_request",
    "acquire_contract",
    "release_contract",
    "extend_contract",
    "query_contract",
    "cleanup_contract",
]
