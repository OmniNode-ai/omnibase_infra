# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest configuration and fixtures for DLQ tracking integration tests.

This module provides fixtures for testing the DLQ PostgreSQL tracking service.
Environment variables should be set via docker-compose.yml or .env file.

CI/CD Graceful Skip Behavior
============================

These integration tests are designed to skip gracefully when infrastructure
is unavailable, enabling CI/CD pipelines to run without hard failures.

Skip Conditions:
    - Skips if POSTGRES_HOST not set
    - Skips if POSTGRES_PASSWORD not set

Environment Variables
=====================

    OMNIBASE_INFRA_DB_URL: Full PostgreSQL DSN (preferred, overrides individual vars)
        Example: postgresql://postgres:secret@localhost:5436/omninode_bridge

    Fallback (used only if OMNIBASE_INFRA_DB_URL is not set):
    POSTGRES_HOST: PostgreSQL server hostname (required - skip if not set)
        Example: localhost or your-server-ip
    POSTGRES_PORT: PostgreSQL server port (default: 5436)
    POSTGRES_USER: Database username (default: postgres)
    POSTGRES_PASSWORD: Database password (required - tests skip if not set)

Related Ticket: OMN-1032 - Complete DLQ Replay PostgreSQL Tracking Integration
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
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
# From tests/infrastructure_config.py:
#   - REMOTE_INFRA_HOST: Default infrastructure server hostname (localhost by default)
#     Used as fallback when POSTGRES_HOST is not set.
#
# This ensures consistent infrastructure endpoint configuration across all
# DLQ integration tests. See tests/infrastructure_config.py for full
# documentation on environment variable overrides and CI/CD graceful skip behavior.
# =============================================================================
from tests.helpers.util_postgres import PostgresConfig
from tests.infrastructure_config import REMOTE_INFRA_HOST

# Use shared PostgresConfig for consistent configuration management
_postgres_config = PostgresConfig.from_env(fallback_host=REMOTE_INFRA_HOST)

# Export individual values for use in availability checks and diagnostics
POSTGRES_HOST = _postgres_config.host
POSTGRES_PORT = str(_postgres_config.port)
POSTGRES_USER = _postgres_config.user
POSTGRES_PASSWORD = _postgres_config.password

# Defensive check: warn if POSTGRES_PASSWORD is missing or empty
if not POSTGRES_PASSWORD:
    import warnings

    warnings.warn(
        "POSTGRES_PASSWORD environment variable not set or empty - DLQ tracking "
        "integration tests will be skipped. Set POSTGRES_PASSWORD in your .env file "
        "or environment to enable database tests.",
        UserWarning,
        stacklevel=1,
    )

# Check if PostgreSQL is available using the shared config
POSTGRES_AVAILABLE = _postgres_config.is_configured


def _build_postgres_dsn() -> str:
    """Build PostgreSQL DSN from environment variables.

    Returns:
        PostgreSQL connection string in standard format.

    Note:
        This function should only be called after verifying
        POSTGRES_PASSWORD is set.
    """
    return _postgres_config.build_dsn()


# =============================================================================
# DLQ Tracking Fixtures
# =============================================================================


@pytest.fixture
def dlq_tracking_config() -> ModelDlqTrackingConfig:
    """Create test configuration for DLQ tracking service.

    This fixture generates a unique table name for each test run to ensure
    test isolation and prevent conflicts between parallel test executions.

    Skip Conditions (CI/CD Graceful Degradation):
        - Skips if POSTGRES_PASSWORD environment variable is not set

    Returns:
        ModelDlqTrackingConfig with test-specific table name.

    Example:
        >>> config = dlq_tracking_config()
        >>> config.storage_table  # 'dlq_replay_history_test_a1b2c3d4'
    """
    if not POSTGRES_PASSWORD:
        pytest.skip("POSTGRES_PASSWORD not set")

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
    service = ServiceDlqTracking(dlq_tracking_config)
    await service.initialize()

    yield service

    # Cleanup: drop test table and shutdown
    # Use try/finally to ensure cleanup even if test fails
    try:
        if service._pool is not None:
            async with service._pool.acquire() as conn:
                await conn.execute(
                    f"DROP TABLE IF EXISTS {dlq_tracking_config.storage_table}"
                )
    except Exception as e:
        logger.warning(
            "Cleanup failed for DLQ tracking table %s: %s",
            dlq_tracking_config.storage_table,
            e,
            exc_info=True,
        )

    # Always attempt shutdown
    try:
        await service.shutdown()
    except Exception as e:
        logger.warning(
            "Cleanup failed for ServiceDlqTracking shutdown: %s",
            e,
            exc_info=True,
        )


@pytest.fixture
def unique_message_id() -> UUID:
    """Generate a unique message ID for test isolation.

    Returns:
        UUID for use as original_message_id in tests.
    """
    return uuid4()
