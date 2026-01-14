# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S106
# S106 disabled: Test password fixtures are intentional for integration testing
"""Pytest fixtures for projector integration tests.

This module provides shared fixtures for projector integration tests using
testcontainers to spin up real PostgreSQL instances. These fixtures ensure
proper isolation and cleanup for each test.

Fixture Scoping Strategy
------------------------
Session-scoped:
    - postgres_container: PostgreSQL testcontainer (expensive startup)

Function-scoped:
    - pg_pool: Fresh connection pool with clean schema per test
    - projector: ProjectorRegistration instance
    - reader: ProjectionReaderRegistration instance

Usage:
    The fixtures handle:
    1. Container lifecycle management (start/stop)
    2. Schema creation from SQL files
    3. Connection pool management
    4. Cleanup between tests
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import TYPE_CHECKING

import asyncpg
import pytest
from testcontainers.postgres import PostgresContainer

if TYPE_CHECKING:
    from omnibase_infra.projectors import (
        ProjectionReaderRegistration,
        ProjectorRegistration,
    )


# Path to SQL schema file
SCHEMA_FILE = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "schemas"
    / "schema_registration_projection.sql"
)


def _check_docker_available() -> bool:
    """Check if Docker daemon is available and running.

    Returns:
        bool: True if Docker is available, False otherwise.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            shell=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


# Check Docker availability at module import time
DOCKER_AVAILABLE = _check_docker_available()


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Session-scoped fixture indicating Docker availability.

    Returns:
        bool: True if Docker daemon is available.
    """
    return DOCKER_AVAILABLE


@pytest.fixture(scope="session")
def postgres_container(
    docker_available: bool,
) -> Generator[PostgresContainer, None, None]:
    """Session-scoped PostgreSQL testcontainer.

    Starts a PostgreSQL container once per test session. The container
    is shared across all tests for performance. Individual tests get
    isolated through schema reset in the pg_pool fixture.

    Args:
        docker_available: Whether Docker daemon is available.

    Yields:
        PostgresContainer with PostgreSQL running.

    Raises:
        pytest.skip: If Docker is not available.
    """
    if not docker_available:
        pytest.skip("Docker daemon not available for testcontainers")

    container = PostgresContainer(
        image="postgres:16-alpine",
        username="test_user",
        password="test_password",
        dbname="test_projections",
    )

    # Start container
    container.start()

    yield container

    # Cleanup: stop container
    container.stop()


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Create event loop policy for async tests.

    This fixture ensures we have a consistent event loop policy
    across the test session for asyncio operations.

    Returns:
        asyncio.DefaultEventLoopPolicy instance.
    """
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def pg_pool(
    postgres_container: PostgresContainer,
) -> AsyncGenerator[asyncpg.Pool, None]:
    """Function-scoped asyncpg connection pool with clean schema.

    Creates a fresh connection pool for each test and initializes
    the schema. Cleans up the table data between tests to ensure
    isolation.

    Args:
        postgres_container: PostgreSQL testcontainer fixture.

    Yields:
        asyncpg.Pool connected to the test database.
    """
    # Get connection URL from container
    connection_url = postgres_container.get_connection_url()

    # Convert from psycopg2 format to asyncpg format
    # psycopg2: postgresql+psycopg2://user:pass@host:port/db
    # asyncpg:  postgresql://user:pass@host:port/db
    dsn = connection_url.replace("postgresql+psycopg2://", "postgresql://")

    # Create pool
    pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=5,
    )

    # Initialize schema
    schema_sql = SCHEMA_FILE.read_text()

    async with pool.acquire() as conn:
        await conn.execute(schema_sql)

    yield pool

    # Cleanup: truncate table for test isolation
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE registration_projections CASCADE")

    # Close pool
    await pool.close()


@pytest.fixture
def projector(pg_pool: asyncpg.Pool) -> ProjectorRegistration:
    """Function-scoped ProjectorRegistration instance.

    Args:
        pg_pool: asyncpg connection pool fixture.

    Returns:
        ProjectorRegistration configured with the test pool.
    """
    from omnibase_infra.projectors import ProjectorRegistration

    return ProjectorRegistration(pg_pool)


@pytest.fixture
def reader(pg_pool: asyncpg.Pool) -> ProjectionReaderRegistration:
    """Function-scoped ProjectionReaderRegistration instance.

    Args:
        pg_pool: asyncpg connection pool fixture.

    Returns:
        ProjectionReaderRegistration configured with the test pool.
    """
    from omnibase_infra.projectors import ProjectionReaderRegistration

    return ProjectionReaderRegistration(pg_pool)
