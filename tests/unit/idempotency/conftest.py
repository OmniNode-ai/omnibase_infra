# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for idempotency store tests.

Provides common fixtures for StoreIdempotencyPostgres testing including
configuration objects and initialized store instances with mocked asyncpg pools.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from omnibase_infra.idempotency import (
    ModelPostgresIdempotencyStoreConfig,
    StoreIdempotencyPostgres,
)


@pytest.fixture
def postgres_config() -> ModelPostgresIdempotencyStoreConfig:
    """Create PostgreSQL idempotency store configuration for tests.

    Returns minimal configuration suitable for most test cases.
    """
    return ModelPostgresIdempotencyStoreConfig(
        dsn="postgresql://user:pass@localhost:5432/testdb",
    )


@pytest.fixture
def postgres_config_extended() -> ModelPostgresIdempotencyStoreConfig:
    """Create extended PostgreSQL idempotency store configuration.

    Returns configuration with explicit pool and timeout settings,
    suitable for initialization tests.
    """
    return ModelPostgresIdempotencyStoreConfig(
        dsn="postgresql://user:pass@localhost:5432/testdb",
        table_name="idempotency_records",
        pool_min_size=1,
        pool_max_size=5,
        command_timeout=30.0,
    )


@pytest.fixture
def postgres_store(
    postgres_config: ModelPostgresIdempotencyStoreConfig,
) -> StoreIdempotencyPostgres:
    """Create PostgreSQL idempotency store for tests.

    Returns an uninitialized store instance.
    """
    return StoreIdempotencyPostgres(postgres_config)


@pytest.fixture
def store(
    postgres_config: ModelPostgresIdempotencyStoreConfig,
) -> StoreIdempotencyPostgres:
    """Alias for postgres_store fixture for convenience.

    Returns an uninitialized store instance.
    """
    return StoreIdempotencyPostgres(postgres_config)


@pytest.fixture
def postgres_store_extended(
    postgres_config_extended: ModelPostgresIdempotencyStoreConfig,
) -> StoreIdempotencyPostgres:
    """Create PostgreSQL idempotency store with extended config.

    Returns an uninitialized store instance with extended configuration.
    """
    return StoreIdempotencyPostgres(postgres_config_extended)


@pytest.fixture
async def initialized_postgres_store(
    postgres_config: ModelPostgresIdempotencyStoreConfig,
) -> AsyncIterator[StoreIdempotencyPostgres]:
    """Create and initialize PostgreSQL idempotency store with mocked pool.

    Yields an initialized store with mocked asyncpg pool for testing
    store operations without requiring an actual PostgreSQL database.

    The store is properly shut down after the test completes.
    """
    store = StoreIdempotencyPostgres(postgres_config)
    mock_pool = MagicMock(spec=asyncpg.Pool)
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock()

    with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_pool
        await store.initialize()

    yield store
    await store.shutdown()
