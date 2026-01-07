# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest configuration and fixtures for handler integration tests.

This module provides fixtures for testing infrastructure handlers.
Environment variables should be set via docker-compose.yml or .env file.

CI/CD Graceful Skip Behavior
============================

These integration tests are designed to skip gracefully when infrastructure
is unavailable, enabling CI/CD pipelines to run without hard failures. This
allows the test suite to be run in environments without access to external
infrastructure (e.g., GitHub Actions without VPN access to internal servers).

Skip Conditions by Handler:

    **PostgreSQL (DbHandler)**:
        - Skips if POSTGRES_HOST not set
        - Skips if POSTGRES_PASSWORD not set
        - Tests use module-level ``pytestmark`` with ``pytest.mark.skipif``

    **Vault (VaultHandler)**:
        - Skips if VAULT_ADDR not set (environment variable)
        - Skips if VAULT_TOKEN not set
        - Skips if Vault server is unreachable (health check fails)
        - Two-phase skip: first checks env vars, then checks reachability

    **Consul (ConsulHandler)**:
        - Skips if CONSUL_HOST not set
        - Skips if Consul server is unreachable (TCP connection fails)
        - Uses socket-based reachability check at module import time

    **HTTP (HttpRestHandler)**:
        - No skip conditions - uses pytest-httpserver for local mock testing
        - Always runs regardless of external infrastructure

Example CI/CD Behavior::

    # In CI without infrastructure access:
    $ pytest tests/integration/handlers/ -v
    tests/.../test_db_handler_integration.py::TestDbHandlerConnection::test_db_describe SKIPPED
    tests/.../test_vault_handler_integration.py::TestVaultHandlerConnection::test_vault_describe SKIPPED
    tests/.../test_consul_handler_integration.py::TestConsulHandlerConnection::test_consul_describe SKIPPED
    tests/.../test_http_handler_integration.py::TestHttpRestHandlerIntegration::test_simple_get_request PASSED

    # With infrastructure access (using REMOTE_INFRA_HOST server):
    $ export POSTGRES_HOST=$REMOTE_INFRA_HOST POSTGRES_PASSWORD=xxx ...
    $ pytest tests/integration/handlers/ -v
    tests/.../test_db_handler_integration.py::TestDbHandlerConnection::test_db_describe PASSED

HTTP Handlers
=============

Uses pytest-httpserver for local mock server testing without external dependencies.
Requirements: pytest-httpserver must be installed: pip install pytest-httpserver

Database Handlers
=================

Environment Variables (required):
    POSTGRES_HOST: PostgreSQL hostname (required)
    POSTGRES_PASSWORD: Database password (required)
Environment Variables (optional):
    POSTGRES_PORT: PostgreSQL port (default: 5432)
    POSTGRES_DATABASE: Database name (default: omninode_bridge)
    POSTGRES_USER: Database username (default: postgres)

DSN Format: postgresql://{user}:{password}@{host}:{port}/{database}

Vault Handlers
==============

Environment Variables (required):
    VAULT_ADDR: Vault server URL (required) - must be a valid URL (e.g., http://localhost:8200)
    VAULT_TOKEN: Vault authentication token (required)
Environment Variables (optional):
    VAULT_NAMESPACE: Vault namespace (for Enterprise)

Error Types for Missing/Invalid Configuration:
    - Missing VAULT_ADDR: RuntimeHostError with message "Missing 'url' in config"
    - Invalid VAULT_ADDR format: ProtocolConfigurationError from Pydantic validation
    - Missing VAULT_TOKEN: RuntimeHostError with message "Missing 'token' in config"
    - Invalid VAULT_TOKEN: InfraAuthenticationError when Vault rejects the token

Consul Handlers
===============

Environment Variables (required):
    CONSUL_HOST: Consul hostname (required)
Environment Variables (optional):
    CONSUL_PORT: Consul port (default: 8500)
    CONSUL_SCHEME: HTTP scheme (default: http)
    CONSUL_TOKEN: ACL token for authentication
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import pytest

# Module-level logger for test cleanup diagnostics
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pytest_httpserver import HTTPServer

    from omnibase_infra.handlers import ConsulHandler, DbHandler, VaultHandler
    from omnibase_infra.models.types import JsonType


# =============================================================================
# Remote Infrastructure Configuration
# =============================================================================
# The ONEX development infrastructure server hosts shared services:
# - PostgreSQL (port 5436)
# - Consul (port 28500)
# - Vault (port 8200)
# - Kafka/Redpanda (port 29092)
#
# This server provides a shared development environment for integration testing
# against real infrastructure components. The default IP is configured in
# tests/infrastructure_config.py and can be overridden via REMOTE_INFRA_HOST.
#
# For local development or CI/CD environments without access to the remote
# infrastructure, set individual *_HOST environment variables to override
# with localhost or Docker container hostnames. Tests will gracefully skip
# if the required infrastructure is unavailable.
#
# Environment Variable Overrides:
#   - Set REMOTE_INFRA_HOST to override the infrastructure server IP
#   - Set POSTGRES_HOST=localhost for local PostgreSQL
#   - Set CONSUL_HOST=localhost for local Consul
#   - Set VAULT_ADDR=http://localhost:8200 for local Vault
#   - Leave unset to skip infrastructure-dependent tests in CI
#
# See tests/infrastructure_config.py for full documentation.
from tests.infrastructure_config import REMOTE_INFRA_HOST

# Backwards compatibility alias (deprecated - use REMOTE_INFRA_HOST instead)
REMOTE_INFRASTRUCTURE_IP = REMOTE_INFRA_HOST


# =============================================================================
# Environment Variable Utilities
# =============================================================================


def _safe_int_env(name: str, default: int) -> int:
    """Safely get integer environment variable with fallback.

    Args:
        name: Environment variable name.
        default: Default value if env var is not set or invalid.

    Returns:
        Integer value from environment or default if not set/invalid.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# =============================================================================
# Database Environment Configuration
# =============================================================================

# Read configuration from environment variables (set via docker-compose or .env)
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "omninode_bridge")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Defensive check: warn if POSTGRES_PASSWORD is missing or empty to avoid silent failures
# Handles None, empty string, and whitespace-only values
if not POSTGRES_PASSWORD or not POSTGRES_PASSWORD.strip():
    import warnings

    warnings.warn(
        "POSTGRES_PASSWORD environment variable not set or empty - database integration "
        "tests will be skipped. Set POSTGRES_PASSWORD in your .env file or environment "
        "to enable database tests.",
        UserWarning,
        stacklevel=1,
    )
    # Normalize to None for consistent availability check
    POSTGRES_PASSWORD = None

# Check if PostgreSQL is available based on host and password being set
POSTGRES_AVAILABLE = POSTGRES_HOST is not None and POSTGRES_PASSWORD is not None


def _build_postgres_dsn() -> str:
    """Build PostgreSQL DSN from environment variables.

    Returns:
        PostgreSQL connection string in standard format.

    Note:
        This function should only be called after verifying
        POSTGRES_PASSWORD is set.
    """
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"
    )


# =============================================================================
# HTTP Handler Fixtures
# =============================================================================


@pytest.fixture
def http_handler_config() -> dict[str, object]:
    """Provide default configuration for HttpRestHandler in integration tests.

    Returns:
        Configuration dict with reasonable size limits for testing.
    """
    return {
        "max_request_size": 1024 * 1024,  # 1 MB
        "max_response_size": 10 * 1024 * 1024,  # 10 MB
    }


@pytest.fixture
def small_response_config() -> dict[str, object]:
    """Provide configuration with small response size limit for testing limits.

    Returns:
        Configuration dict with small response size limit.
    """
    return {
        "max_request_size": 1024 * 1024,  # 1 MB
        "max_response_size": 100,  # 100 bytes - for testing size limits
    }


# =============================================================================
# Database Handler Fixtures
# =============================================================================


@pytest.fixture
def db_config() -> dict[str, JsonType]:
    """Provide database configuration for DbHandler.

    This fixture enables graceful skip behavior for CI/CD environments
    where database infrastructure may not be available.

    Skip Conditions (CI/CD Graceful Degradation):
        - Skips immediately if POSTGRES_PASSWORD environment variable is not set
        - Combined with module-level pytestmark skipif for POSTGRES_HOST

    Returns:
        Configuration dict with 'dsn' key for DbHandler.initialize().

    Note:
        Tests using this fixture should also use @pytest.mark.skipif
        or combine with POSTGRES_AVAILABLE check at the module level.
        The module-level skip is preferred for cleaner test output.

    Example:
        >>> # In CI without database access:
        >>> # Test is skipped with message "POSTGRES_PASSWORD not set"
        >>> # In development with database:
        >>> config = db_config()  # Returns valid DSN configuration
    """
    if not POSTGRES_PASSWORD:
        pytest.skip("POSTGRES_PASSWORD not set")

    return {
        "dsn": _build_postgres_dsn(),
        "timeout": 30.0,
    }


@pytest.fixture
def unique_table_name() -> str:
    """Generate a unique test table name for isolation.

    Returns:
        Unique table name prefixed with 'test_' and containing a UUID.

    Example:
        >>> table = unique_table_name()
        >>> # Returns something like 'test_a1b2c3d4e5f6'
    """
    return f"test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
async def initialized_db_handler(
    db_config: dict[str, JsonType],
) -> AsyncGenerator[DbHandler, None]:
    """Provide an initialized DbHandler instance with automatic cleanup.

    Creates a DbHandler, initializes it with the test configuration,
    yields it for the test, then ensures proper cleanup via shutdown().

    Cleanup Behavior:
        - Calls handler.shutdown() after test completion
        - Shutdown is idempotent (safe to call multiple times)
        - Ignores any cleanup errors to prevent test pollution
        - Closes connection pool and releases all resources

    Yields:
        Initialized DbHandler ready for database operations.

    Note:
        This fixture handles cleanup automatically. Tests should not
        call shutdown() manually unless testing shutdown behavior.
        If a test calls shutdown(), the fixture's cleanup will simply
        detect the handler is already shut down and complete gracefully.

    Example:
        >>> async def test_with_db(initialized_db_handler):
        ...     result = await initialized_db_handler.execute(envelope)
        ...     # No need to call shutdown - fixture handles it
    """
    from omnibase_infra.handlers import DbHandler

    handler = DbHandler()
    await handler.initialize(db_config)

    yield handler

    # Cleanup: ensure handler is properly shut down
    # Idempotent: safe even if test already called shutdown()
    try:
        await handler.shutdown()
    except Exception as e:
        logger.warning(
            "Cleanup failed for DbHandler shutdown: %s",
            e,
            exc_info=True,
        )


@pytest.fixture
async def cleanup_table(
    initialized_db_handler: DbHandler,
) -> AsyncGenerator[list[str], None]:
    """Fixture to track and cleanup test tables with idempotent deletion.

    Yields a list where tests can append table names they create.
    After the test completes, all listed tables are dropped.

    Cleanup Behavior:
        - Uses DROP TABLE IF EXISTS (idempotent - safe if table doesn't exist)
        - Iterates through all tracked tables regardless of individual failures
        - Ignores cleanup errors to prevent test pollution
        - Runs after test completion (success or failure)

    Test Isolation:
        This fixture enables test isolation by ensuring each test's tables
        are cleaned up, preventing data leakage between tests. Combined
        with unique_table_name fixture, this guarantees no table conflicts.

    Yields:
        List to which tests can append table names for cleanup.

    Example:
        >>> async def test_create_table(initialized_db_handler, cleanup_table):
        ...     table = "test_my_table"
        ...     cleanup_table.append(table)
        ...     await initialized_db_handler.execute(...)
        ...     # Table will be dropped after test, even if test fails
    """
    tables_to_cleanup: list[str] = []

    yield tables_to_cleanup

    # Cleanup: drop all tables that were tracked
    # Idempotent: DROP TABLE IF EXISTS succeeds even for non-existent tables
    for table in tables_to_cleanup:
        try:
            envelope = {
                "operation": "db.execute",
                "payload": {
                    "sql": f'DROP TABLE IF EXISTS "{table}"',
                    "parameters": [],
                },
            }
            await initialized_db_handler.execute(envelope)
        except Exception as e:
            logger.warning(
                "Cleanup failed for table %s: %s",
                table,
                e,
                exc_info=True,
            )


# =============================================================================
# Vault Environment Configuration
# =============================================================================

# Get Vault configuration from environment (set via docker-compose or .env)
VAULT_ADDR = os.getenv("VAULT_ADDR")
VAULT_TOKEN = os.getenv("VAULT_TOKEN")
VAULT_NAMESPACE = os.getenv("VAULT_NAMESPACE")

# Defensive check: warn if VAULT_TOKEN is missing or empty to avoid silent failures
# Handles None, empty string, and whitespace-only values
if not VAULT_TOKEN or not VAULT_TOKEN.strip():
    import warnings

    warnings.warn(
        "VAULT_TOKEN environment variable not set or empty - Vault integration tests "
        "will be skipped. Set VAULT_TOKEN in your .env file or environment to enable "
        "Vault tests.",
        UserWarning,
        stacklevel=1,
    )
    # Normalize to None for consistent availability check
    VAULT_TOKEN = None

# Vault is available if address and token are set
VAULT_AVAILABLE = VAULT_ADDR is not None and VAULT_TOKEN is not None


def _check_vault_reachable() -> bool:
    """Check if Vault server is reachable.

    Makes a simple HTTP request to Vault health endpoint to verify connectivity.

    Returns:
        bool: True if Vault is reachable, False otherwise.
    """
    if not VAULT_AVAILABLE:
        return False

    import urllib.request
    from urllib.error import URLError

    try:
        # Use health check endpoint (doesn't require auth)
        health_url = f"{VAULT_ADDR}/v1/sys/health"
        req = urllib.request.Request(health_url, method="GET")  # noqa: S310
        req.add_header("X-Vault-Request", "true")

        with urllib.request.urlopen(req, timeout=5) as response:  # noqa: S310
            # 200 = initialized, unsealed, active
            # 429 = standby (but reachable)
            # 472 = DR secondary
            # 473 = performance standby
            # 501 = uninitialized
            # 503 = sealed
            return response.status in (200, 429, 472, 473, 501, 503)
    except (URLError, TimeoutError, OSError):
        return False


# Check Vault reachability at module import time
VAULT_REACHABLE = _check_vault_reachable()


# =============================================================================
# Vault Handler Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def vault_available() -> bool:
    """Session-scoped fixture indicating Vault availability.

    This fixture enables graceful skip behavior for CI/CD environments
    where Vault infrastructure may not be available. Tests can use this
    fixture to conditionally skip based on infrastructure availability.

    Skip Conditions (Two-Phase Check):
        Phase 1 - Environment Variables:
            - Returns False if VAULT_ADDR not set
            - Returns False if VAULT_TOKEN not set

        Phase 2 - Reachability:
            - Returns False if Vault health endpoint is unreachable
            - Uses HTTP request to /v1/sys/health with 5-second timeout
            - Accepts various status codes (200, 429, 472, 473, 501, 503)
              as "reachable" since they indicate the server is responding

    Returns:
        bool: True if Vault is available for testing.

    CI/CD Behavior:
        In CI environments without Vault access, this returns False,
        causing tests to be skipped gracefully without failures.

    Example:
        >>> @pytest.mark.skipif(not vault_available(), reason="Vault unavailable")
        >>> async def test_vault_secret_read(vault_handler):
        ...     # This test skips in CI without Vault
        ...     pass
    """
    return VAULT_AVAILABLE and VAULT_REACHABLE


@pytest.fixture
def vault_config() -> dict[str, JsonType]:
    """Get Vault configuration from environment variables.

    Returns:
        Configuration dict for VaultHandler.initialize()

    Note:
        This fixture does not skip tests if Vault is unavailable.
        Use the vault_available fixture or module-level pytestmark
        for skipping tests.
    """
    config: dict[str, JsonType] = {
        "url": VAULT_ADDR,
        "token": VAULT_TOKEN,
        "timeout_seconds": 30.0,
        "verify_ssl": False,  # Allow self-signed certs in dev/test
        "circuit_breaker_enabled": True,
        "circuit_breaker_failure_threshold": 5,
        "circuit_breaker_reset_timeout_seconds": 30.0,
    }

    if VAULT_NAMESPACE:
        config["namespace"] = VAULT_NAMESPACE

    return config


@pytest.fixture
async def vault_handler(
    vault_config: dict[str, JsonType],
) -> AsyncGenerator[VaultHandler, None]:
    """Create and initialize VaultHandler for integration testing with automatic cleanup.

    Yields an initialized VaultHandler instance and ensures proper cleanup.

    Cleanup Behavior:
        - Calls handler.shutdown() after test completion
        - Closes HTTP client connections to Vault
        - Idempotent: safe to call shutdown() multiple times
        - Ignores cleanup errors to prevent test pollution

    Args:
        vault_config: Vault configuration fixture.

    Yields:
        Initialized VaultHandler instance.

    Note:
        This fixture handles cleanup automatically. Tests should not
        call shutdown() manually unless testing shutdown behavior.
    """
    from omnibase_infra.handlers import VaultHandler

    handler = VaultHandler()
    await handler.initialize(vault_config)

    yield handler

    # Cleanup: ensure handler is shutdown
    # Idempotent: safe even if test already called shutdown()
    try:
        await handler.shutdown()
    except Exception as e:
        logger.warning(
            "Cleanup failed for VaultHandler shutdown: %s",
            e,
            exc_info=True,
        )


# =============================================================================
# Consul Environment Configuration
# =============================================================================

# Read Consul configuration from environment (set via docker-compose or .env)
CONSUL_HOST = os.getenv("CONSUL_HOST")
CONSUL_PORT = _safe_int_env("CONSUL_PORT", 8500)
CONSUL_SCHEME = os.getenv("CONSUL_SCHEME", "http")
CONSUL_TOKEN = os.getenv("CONSUL_TOKEN")


def _check_consul_reachable() -> bool:
    """Check if Consul server is reachable.

    Makes a TCP connection to verify connectivity.

    Returns:
        bool: True if Consul is reachable, False otherwise.
    """
    if CONSUL_HOST is None:
        return False

    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            result = sock.connect_ex((CONSUL_HOST, CONSUL_PORT))
            return result == 0
    except (OSError, TimeoutError):
        return False


# Check Consul reachability at module import time
CONSUL_AVAILABLE = _check_consul_reachable()


# =============================================================================
# Consul Handler Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def consul_available() -> bool:
    """Session-scoped fixture indicating Consul availability.

    This fixture enables graceful skip behavior for CI/CD environments
    where Consul infrastructure may not be available. The check is
    performed at module import time using a TCP socket connection.

    Skip Conditions:
        - Returns False if CONSUL_HOST environment variable not set
        - Returns False if TCP connection to CONSUL_HOST:CONSUL_PORT fails
        - Uses 5-second timeout for connection attempts

    Returns:
        bool: True if Consul is available for testing.

    CI/CD Behavior:
        In CI environments without Consul access, this returns False,
        causing tests to be skipped gracefully without failures.

    Example:
        >>> @pytest.mark.skipif(not consul_available(), reason="Consul unavailable")
        >>> async def test_consul_kv_put(consul_handler):
        ...     # This test skips in CI without Consul
        ...     pass
    """
    return CONSUL_AVAILABLE


@pytest.fixture
def consul_config() -> dict[str, JsonType]:
    """Provide Consul configuration for ConsulHandler.

    Returns:
        Configuration dict for ConsulHandler.initialize()
    """
    config: dict[str, JsonType] = {
        "host": CONSUL_HOST,
        "port": CONSUL_PORT,
        "scheme": CONSUL_SCHEME,
        "timeout_seconds": 30.0,
        "max_concurrent_operations": 5,
        "circuit_breaker_enabled": True,
        "circuit_breaker_failure_threshold": 3,
        "circuit_breaker_reset_timeout_seconds": 30.0,
    }

    # Add token if provided
    if CONSUL_TOKEN:
        config["token"] = CONSUL_TOKEN

    return config


@pytest.fixture
def unique_kv_key() -> str:
    """Generate unique KV key for test isolation.

    Returns:
        Unique key path prefixed with test namespace.
    """
    return f"integration-test/consul/{uuid.uuid4().hex[:12]}"


@pytest.fixture
def unique_service_name() -> str:
    """Generate unique service name for test isolation.

    Returns:
        Unique service name for registration tests.
    """
    return f"integration-test-svc-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def initialized_consul_handler(
    consul_config: dict[str, JsonType],
) -> AsyncGenerator[ConsulHandler, None]:
    """Provide an initialized ConsulHandler instance with automatic cleanup.

    Creates a ConsulHandler, initializes it with the test configuration,
    yields it for the test, then ensures proper cleanup via shutdown().

    Cleanup Behavior:
        - Calls handler.shutdown() after test completion
        - Closes HTTP client connections to Consul
        - Idempotent: safe to call shutdown() multiple times
        - Ignores cleanup errors to prevent test pollution

    Note:
        This fixture does NOT clean up KV keys or registered services.
        Use unique_kv_key and unique_service_name fixtures for test data
        that needs cleanup, and handle cleanup in individual tests or
        use dedicated cleanup fixtures.

    Args:
        consul_config: Consul configuration fixture.

    Yields:
        Initialized ConsulHandler ready for Consul operations.
    """
    from omnibase_infra.handlers import ConsulHandler

    handler = ConsulHandler()
    await handler.initialize(consul_config)

    yield handler

    # Cleanup: ensure handler is properly shut down
    # Idempotent: safe even if test already called shutdown()
    try:
        await handler.shutdown()
    except Exception as e:
        logger.warning(
            "Cleanup failed for ConsulHandler shutdown: %s",
            e,
            exc_info=True,
        )
