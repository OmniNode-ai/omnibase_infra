# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pytest configuration and fixtures for DLQ tracking integration tests.  # ai-slop-ok: pre-existing

This module provides fixtures for testing the DLQ PostgreSQL tracking service.
Environment variables should be set via docker-compose.yml or .env file.

CI/CD Graceful Skip Behavior
============================  # ai-slop-ok: pre-existing

These integration tests are designed to skip gracefully when infrastructure
is unavailable, enabling CI/CD pipelines to run without hard failures.

Skip Conditions:
    - Skips if OMNIBASE_INFRA_DB_URL not set

Environment Variables
=====================  # ai-slop-ok: pre-existing

    OMNIBASE_INFRA_DB_URL: Full PostgreSQL DSN (required, no fallback)
        Example: postgresql://postgres:secret@localhost:5432/omnibase_infra
        Tests skip if this variable is not set or malformed.

Related Ticket: OMN-1032 - Complete DLQ Replay PostgreSQL Tracking Integration
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

# Module-level logger for test cleanup diagnostics
logger = logging.getLogger(__name__)

from omnibase_infra.dlq import (
    ModelDlqTrackingConfig,
    ServiceDlqTracking,
)

# =============================================================================
# Database Environment Configuration
# =============================================================================
# Read configuration from environment variables (set via docker-compose or .env)
#
# Cross-Module Import: Shared Test Helpers
# From tests/helpers/util_postgres:
#   - PostgresConfig: Configuration dataclass for PostgreSQL connections
#
# This ensures consistent infrastructure endpoint configuration across all
# DLQ integration tests. See tests/infrastructure_config.py for full
# documentation on environment variable overrides and CI/CD graceful skip behavior.
# =============================================================================
from tests.helpers.util_postgres import PostgresConfig

# Use shared PostgresConfig for consistent configuration management
_postgres_config = PostgresConfig.from_env()

# Export individual values for use in availability checks and diagnostics
POSTGRES_HOST = _postgres_config.host
POSTGRES_PORT = str(_postgres_config.port)
POSTGRES_USER = _postgres_config.user
POSTGRES_PASSWORD = _postgres_config.password

# Defensive check: warn if PostgreSQL is not configured at all
if not _postgres_config.is_configured:
    import warnings

    warnings.warn(
        "PostgreSQL not configured - DLQ tracking integration tests will be skipped. "
        "Set OMNIBASE_INFRA_DB_URL in your .env file or environment to enable "
        "database tests.",
        UserWarning,
        stacklevel=1,
    )

# Check if PostgreSQL is available using the shared config
POSTGRES_AVAILABLE = _postgres_config.is_configured


def _build_postgres_dsn() -> str:
    """Build PostgreSQL DSN by delegating to PostgresConfig.build_dsn().

    Returns:
        PostgreSQL connection string in standard format.

    Raises:
        ProtocolConfigurationError: If configuration is incomplete
            (host, password, or database missing).
    """
    return _postgres_config.build_dsn()


def _build_test_table_ddl(table_name: str) -> tuple[str, str, str]:
    """Build the DLQ replay-history DDL for a per-test table.

    ServiceDlqTracking no longer creates its table at runtime (OMN-12633) — the
    canonical ``dlq_replay_history`` table is provisioned by the forward
    migration runner (``docker/migrations/forward/086_create_dlq_replay_history.sql``).
    Integration tests use a unique per-test table for isolation, so the fixture
    must provision that table itself.  The DDL mirrors the canonical schema in
    ``src/omnibase_infra/schemas/schema_dlq_replay_history.sql``.

    Args:
        table_name: Validated test table identifier (alphanumeric + underscore).

    Returns:
        Tuple of (create-table SQL, message-id-index SQL, timestamp-index SQL).
    """
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id UUID PRIMARY KEY,
            original_message_id UUID NOT NULL,
            replay_correlation_id UUID NOT NULL,
            original_topic VARCHAR(255) NOT NULL,
            target_topic VARCHAR(255) NOT NULL,
            replay_status VARCHAR(50) NOT NULL,
            replay_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            success BOOLEAN NOT NULL,
            error_message TEXT,
            dlq_offset BIGINT NOT NULL,
            dlq_partition INTEGER NOT NULL,
            retry_count INTEGER NOT NULL DEFAULT 0
        )
    """
    create_message_id_index_sql = f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_message_id
        ON {table_name}(original_message_id)
    """
    create_timestamp_index_sql = f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp
        ON {table_name}(replay_timestamp)
    """
    return create_table_sql, create_message_id_index_sql, create_timestamp_index_sql


async def _provision_test_table(dsn: str, table_name: str) -> None:
    """Provision a per-test DLQ replay-history table over a short-lived connection.

    ServiceDlqTracking no longer creates its table at runtime (OMN-12633), so any
    test that constructs a service directly must provision the table first.

    Args:
        dsn: PostgreSQL DSN to connect with.
        table_name: Validated test table identifier.
    """
    import asyncpg

    create_table_sql, message_id_index_sql, timestamp_index_sql = _build_test_table_ddl(
        table_name
    )
    conn = await asyncpg.connect(dsn=dsn)
    try:
        async with conn.transaction():
            await conn.execute(create_table_sql)
            await conn.execute(message_id_index_sql)
            await conn.execute(timestamp_index_sql)
    finally:
        await conn.close()


async def _drop_test_table(dsn: str, table_name: str) -> None:
    """Drop a per-test DLQ replay-history table, logging on failure.

    Args:
        dsn: PostgreSQL DSN to connect with.
        table_name: Validated test table identifier.
    """
    import asyncpg

    try:
        conn = await asyncpg.connect(dsn=dsn)
        try:
            await conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        finally:
            await conn.close()
    except Exception as e:  # noqa: BLE001 — boundary: logs warning and degrades
        logger.warning(
            "Cleanup failed for DLQ tracking table %s: %s",
            table_name,
            e,
            exc_info=True,
        )


# =============================================================================
# DLQ Tracking Fixtures
# =============================================================================


@pytest.fixture
def dlq_tracking_config() -> ModelDlqTrackingConfig:
    """Create test configuration for DLQ tracking service.

    This fixture generates a unique table name for each test run to ensure
    test isolation and prevent conflicts between parallel test executions.

    Skip Conditions (CI/CD Graceful Degradation):
        - Skips if PostgreSQL is not available (OMNIBASE_INFRA_DB_URL not set)

    Returns:
        ModelDlqTrackingConfig with test-specific table name.

    Example:
        >>> config = dlq_tracking_config()
        >>> config.storage_table  # 'dlq_replay_history_test_a1b2c3d4'
    """
    if not POSTGRES_AVAILABLE:
        pytest.skip("PostgreSQL not available (set OMNIBASE_INFRA_DB_URL)")

    return ModelDlqTrackingConfig(
        dsn=_build_postgres_dsn(),
        storage_table=f"dlq_replay_history_test_{uuid4().hex[:8]}",
        pool_min_size=1,
        pool_max_size=3,
        command_timeout=30.0,
    )


@pytest_asyncio.fixture
async def dlq_tracking_service(
    dlq_tracking_config: ModelDlqTrackingConfig,
) -> AsyncGenerator[ServiceDlqTracking, None]:
    """Create and initialize DLQ tracking service for tests.

    This fixture handles the complete lifecycle of the DLQ tracking service:
    1. Creates service instance with test configuration
    2. Initializes connection pool and creates test table
    3. Yields service for test execution
    4. Cleans up by dropping test table and shutting down service

    Cleanup Behavior:
        - Drops the test-specific table after test completion
        - Closes connection pool via shutdown()
        - Idempotent: safe even if test already caused cleanup

    Args:
        dlq_tracking_config: Test configuration with unique table name.

    Yields:
        Initialized ServiceDlqTracking ready for testing.

    Example:
        >>> async def test_record_replay(dlq_tracking_service):
        ...     record = ModelDlqReplayRecord(...)
        ...     await dlq_tracking_service.record_replay_attempt(record)
        ...     # Table is automatically cleaned up after test
    """
    # Provision the per-test table before initialising the service.
    # ServiceDlqTracking no longer creates its table at runtime (OMN-12633);
    # the canonical table is owned by the forward migration runner.  Tests use
    # an isolated per-test table, so the fixture provisions it from the same
    # DDL shape as the canonical schema.
    await _provision_test_table(
        dlq_tracking_config.dsn, dlq_tracking_config.storage_table
    )

    service = ServiceDlqTracking(dlq_tracking_config)
    await service.initialize()

    yield service

    # Always attempt shutdown
    try:
        await service.shutdown()
    except Exception as e:  # noqa: BLE001 — boundary: logs warning and degrades
        logger.warning(
            "Cleanup failed for ServiceDlqTracking shutdown: %s",
            e,
            exc_info=True,
        )

    # Cleanup: drop test table after the pool is closed
    await _drop_test_table(dlq_tracking_config.dsn, dlq_tracking_config.storage_table)


@pytest_asyncio.fixture
async def provision_dlq_table() -> AsyncGenerator[
    Callable[[ModelDlqTrackingConfig], Awaitable[None]], None
]:
    """Provide a callable that provisions a per-test DLQ table for a config.

    For tests that construct ``ServiceDlqTracking`` directly (rather than using
    the ``dlq_tracking_service`` fixture) but still need the backing table to
    exist — ServiceDlqTracking no longer creates it at runtime (OMN-12633).

    The fixture tracks every provisioned table and drops it on teardown.

    Example:
        >>> async def test_x(dlq_tracking_config, provision_dlq_table):
        ...     await provision_dlq_table(dlq_tracking_config)
        ...     service = ServiceDlqTracking(dlq_tracking_config)
        ...     await service.initialize()
    """
    provisioned: list[ModelDlqTrackingConfig] = []

    async def _provision(config: ModelDlqTrackingConfig) -> None:
        await _provision_test_table(config.dsn, config.storage_table)
        provisioned.append(config)

    yield _provision

    for config in provisioned:
        await _drop_test_table(config.dsn, config.storage_table)


@pytest.fixture
def unique_message_id() -> UUID:
    """Generate a unique message ID for test isolation.

    Returns:
        UUID for use as original_message_id in tests.
    """
    return uuid4()
