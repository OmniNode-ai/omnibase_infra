# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# S106 disabled: Test password fixtures are intentional for integration testing
"""Pytest fixtures for projector integration tests.

This module provides shared fixtures for projector integration tests using
testcontainers to spin up real PostgreSQL instances. These fixtures ensure
proper isolation and cleanup for each test.

IMPORTANT: Event Loop Scope Configuration (pytest-asyncio 0.25+)
================================================================

When using session-scoped async fixtures with pytest-asyncio 0.25+, you MUST
configure the event loop scope to prevent "attached to a different loop" errors.

**For modules importing these fixtures**, add to the importing module:

.. code-block:: python

    pytestmark = [pytest.mark.asyncio(loop_scope="session")]

Or use module scope if only module-scoped fixtures are needed:

.. code-block:: python

    pytestmark = [pytest.mark.asyncio(loop_scope="module")]

**Why**: pytest-asyncio 0.25+ defaults to function-scoped event loops. Session/module
scoped async fixtures need matching loop scope to avoid RuntimeError when sharing
async resources across tests.

Fixture Scoping Strategy
------------------------
Session-scoped:
    - postgres_container: PostgreSQL testcontainer (expensive startup)
    - event_loop_policy: Asyncio event loop policy

Function-scoped:
    - pg_pool: Fresh connection pool with clean schema per test
    - projector: ProjectorShell instance (contract-driven)
    - reader: ProjectionReaderRegistration instance

Usage:
    The fixtures handle:
    1. Container lifecycle management (start/stop)
    2. Schema creation from SQL files
    3. Connection pool management
    4. Cleanup between tests

Related Tickets:
    - OMN-1169: ProjectorShell for contract-driven projections
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import TYPE_CHECKING

import asyncpg
import pytest
import yaml
from testcontainers.postgres import PostgresContainer

from omnibase_core.models.projectors import ModelProjectorContract
from omnibase_infra.projectors.contracts import REGISTRATION_PROJECTOR_CONTRACT

if TYPE_CHECKING:
    # TYPE_CHECKING imports: These imports are only used for type annotations.
    # They are NOT imported at runtime, which:
    # 1. Avoids circular import issues (projector modules may import test utilities)
    # 2. Allows type hints without requiring all projector dependencies at collection time
    # 3. Enables IDE autocompletion and type checking for fixture return types
    from omnibase_infra.projectors import ProjectionReaderRegistration
    from omnibase_infra.runtime import ProjectorShell

    # Legacy type alias for backward compatibility
    # ProjectorRegistration has been superseded by ProjectorShell
    ProjectorRegistration = object  # type: ignore[misc]


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

    IMPORTANT: Event Loop Scope Configuration (pytest-asyncio 0.25+)
    =================================================================

    This fixture ensures we have a consistent event loop policy across the test
    session for asyncio operations. Starting with pytest-asyncio 0.25, the default
    event loop scope changed from "module" to "function", which breaks session/module
    scoped async fixtures.

    **Why This Matters:**

    - Session-scoped fixtures (``postgres_container``) are created on one event loop
    - Without proper loop scope config, each test gets a NEW event loop
    - Sharing async resources across different loops causes RuntimeError

    **Symptoms Without Proper Configuration:**

    .. code-block:: text

        RuntimeError: Task <Task pending ...> got Future <Future ...>
        attached to a different loop

    **Configuration Options:**

    1. **pytestmark with loop_scope** (preferred for module-scoped fixtures):

       .. code-block:: python

           pytestmark = [pytest.mark.asyncio(loop_scope="module")]

    2. **pytest.ini or pyproject.toml** (global default):

       .. code-block:: toml

           [tool.pytest.ini_options]
           asyncio_default_fixture_loop_scope = "module"

    **Reference:**
        https://pytest-asyncio.readthedocs.io/en/latest/concepts.html#event-loop-scope

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
async def projector(pg_pool: asyncpg.Pool) -> ProjectorShell:
    """Function-scoped ProjectorShell instance loaded from contract.

    Uses ProjectorPluginLoader to load the registration projector from
    its YAML contract definition. This ensures the test uses the same
    contract-driven configuration as production.

    Args:
        pg_pool: asyncpg connection pool fixture.

    Returns:
        ProjectorShell configured with the test pool and registration contract.

    Related:
        - OMN-1169: ProjectorShell for contract-driven projections
        - OMN-1168: ProjectorPluginLoader contract discovery
    """
    from omnibase_infra.runtime import ProjectorPluginLoader

    # Use exported constant for canonical contract location
    loader = ProjectorPluginLoader(pool=pg_pool)
    projector = await loader.load_from_contract(REGISTRATION_PROJECTOR_CONTRACT)

    # Type narrowing - loader with pool returns ProjectorShell, not placeholder
    from omnibase_infra.runtime import ProjectorShell

    assert isinstance(projector, ProjectorShell), (
        "Expected ProjectorShell instance when pool is provided"
    )
    return projector


@pytest.fixture
def reader(pg_pool: asyncpg.Pool) -> ProjectionReaderRegistration:
    """Function-scoped ProjectionReaderRegistration instance.

    Note: The reader is kept as ProjectionReaderRegistration (not a generic shell)
    because it provides domain-specific query methods for registration state
    (get_by_state, get_overdue_ack_registrations, capability queries, etc.).

    Args:
        pg_pool: asyncpg connection pool fixture.

    Returns:
        ProjectionReaderRegistration configured with the test pool.
    """
    from omnibase_infra.projectors import ProjectionReaderRegistration

    return ProjectionReaderRegistration(pg_pool)


@pytest.fixture
def legacy_projector(pg_pool: asyncpg.Pool) -> ProjectorRegistration:
    """Function-scoped legacy ProjectorRegistration instance.

    This fixture provides the legacy ProjectorRegistration for tests that
    still require the old persist() interface.

    Note:
        The ProjectorRegistration class has been superseded by ProjectorShell.
        This fixture attempts to import the legacy class and skips the test
        if it's not available. Tests should be migrated to use ProjectorShell.

    Args:
        pg_pool: asyncpg connection pool fixture.

    Returns:
        ProjectorRegistration configured with the test pool.

    Raises:
        pytest.skip: If ProjectorRegistration is not available.
    """
    try:
        from omnibase_infra.projectors.projector_registration import (
            ProjectorRegistration,
        )
    except ImportError:
        pytest.skip(
            "ProjectorRegistration not available - "
            "tests should be migrated to use ProjectorShell"
        )

    return ProjectorRegistration(pg_pool)


@pytest.fixture
def contract() -> ModelProjectorContract:
    """Load the registration projector contract.

    Uses the exported REGISTRATION_PROJECTOR_CONTRACT constant to ensure
    tests always use the canonical contract location.

    Note:
        The contract YAML may contain extended fields (e.g., partial_updates)
        that are not part of the base ModelProjectorContract model. These are
        stripped before validation. The partial_updates definitions are used
        for documentation and runtime behavior but not validated by the
        base contract model.

    Returns:
        Parsed ModelProjectorContract from YAML.

    Raises:
        pytest.fail: If contract file doesn't exist.
    """
    if not REGISTRATION_PROJECTOR_CONTRACT.exists():
        pytest.fail(f"Contract file not found: {REGISTRATION_PROJECTOR_CONTRACT}")

    with open(REGISTRATION_PROJECTOR_CONTRACT) as f:
        data = yaml.safe_load(f)

    # Strip extended fields not in base ModelProjectorContract
    # partial_updates is an extension for OMN-1170 that defines partial update operations
    data.pop("partial_updates", None)

    # Handle composite key fields: ModelProjectorContract expects strings, but the
    # contract YAML uses lists for composite primary/upsert keys.
    # Convert first element of list to string for model validation.
    # The full composite key information is preserved in the SQL schema.
    if isinstance(data.get("projection_schema", {}).get("primary_key"), list):
        pk_list = data["projection_schema"]["primary_key"]
        data["projection_schema"]["primary_key"] = (
            pk_list[0] if pk_list else "entity_id"
        )

    if isinstance(data.get("behavior", {}).get("upsert_key"), list):
        upsert_list = data["behavior"]["upsert_key"]
        data["behavior"]["upsert_key"] = upsert_list[0] if upsert_list else None

    return ModelProjectorContract.model_validate(data)
