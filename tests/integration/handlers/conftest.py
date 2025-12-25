# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest configuration and fixtures for handler integration tests.

This module provides fixtures for testing infrastructure handlers.
Environment variables should be set via docker-compose.yml or .env file.

HTTP Handlers:
    Uses pytest-httpserver for local mock server testing without external dependencies.
    Requirements: pytest-httpserver must be installed: pip install pytest-httpserver

Database Handlers:
    Environment Variables (required):
        POSTGRES_HOST: PostgreSQL hostname (required)
        POSTGRES_PASSWORD: Database password (required)
    Environment Variables (optional):
        POSTGRES_PORT: PostgreSQL port (default: 5432)
        POSTGRES_DATABASE: Database name (default: omninode_bridge)
        POSTGRES_USER: Database username (default: postgres)

    DSN Format: postgresql://{user}:{password}@{host}:{port}/{database}

Vault Handlers:
    Environment Variables (required):
        VAULT_ADDR: Vault server URL (required)
        VAULT_TOKEN: Vault authentication token (required)
    Environment Variables (optional):
        VAULT_NAMESPACE: Vault namespace (for Enterprise)

Consul Handlers:
    Environment Variables (required):
        CONSUL_HOST: Consul hostname (required)
    Environment Variables (optional):
        CONSUL_PORT: Consul port (default: 8500)
        CONSUL_SCHEME: HTTP scheme (default: http)
        CONSUL_TOKEN: ACL token for authentication
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from omnibase_core.types import JsonValue
    from pytest_httpserver import HTTPServer

    from omnibase_infra.handlers import ConsulHandler, DbHandler, VaultHandler


# =============================================================================
# Database Environment Configuration
# =============================================================================

# Read configuration from environment variables (set via docker-compose or .env)
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "omninode_bridge")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

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
def db_config() -> dict[str, JsonValue]:
    """Provide database configuration for DbHandler.

    Returns:
        Configuration dict with 'dsn' key for DbHandler.initialize().

    Note:
        Tests using this fixture should also use @pytest.mark.skipif
        or combine with POSTGRES_AVAILABLE check.
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
    db_config: dict[str, JsonValue],
) -> AsyncGenerator[DbHandler, None]:
    """Provide an initialized DbHandler instance.

    Creates a DbHandler, initializes it with the test configuration,
    yields it for the test, then ensures proper cleanup via shutdown().

    Yields:
        Initialized DbHandler ready for database operations.

    Note:
        This fixture handles cleanup automatically. Tests should not
        call shutdown() manually unless testing shutdown behavior.
    """
    from omnibase_infra.handlers import DbHandler

    handler = DbHandler()
    await handler.initialize(db_config)

    yield handler

    # Cleanup: ensure handler is properly shut down
    try:
        await handler.shutdown()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
async def cleanup_table(
    initialized_db_handler: DbHandler,
) -> AsyncGenerator[list[str], None]:
    """Fixture to track and cleanup test tables.

    Yields a list where tests can append table names they create.
    After the test completes, all listed tables are dropped.

    Yields:
        List to which tests can append table names for cleanup.

    Example:
        >>> async def test_create_table(initialized_db_handler, cleanup_table):
        ...     table = "test_my_table"
        ...     cleanup_table.append(table)
        ...     await initialized_db_handler.execute(...)
        ...     # Table will be dropped after test
    """
    tables_to_cleanup: list[str] = []

    yield tables_to_cleanup

    # Cleanup: drop all tables that were tracked
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
        except Exception:
            pass  # Ignore cleanup errors


# =============================================================================
# Vault Environment Configuration
# =============================================================================

# Get Vault configuration from environment (set via docker-compose or .env)
VAULT_ADDR = os.getenv("VAULT_ADDR")
VAULT_TOKEN = os.getenv("VAULT_TOKEN")
VAULT_NAMESPACE = os.getenv("VAULT_NAMESPACE")

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

    Returns True only if:
    1. VAULT_TOKEN environment variable is set
    2. Vault server is reachable at VAULT_ADDR

    Returns:
        bool: True if Vault is available for testing.
    """
    return VAULT_AVAILABLE and VAULT_REACHABLE


@pytest.fixture
def vault_config() -> dict[str, JsonValue]:
    """Get Vault configuration from environment variables.

    Returns:
        Configuration dict for VaultHandler.initialize()

    Note:
        This fixture does not skip tests if Vault is unavailable.
        Use the vault_available fixture or module-level pytestmark
        for skipping tests.
    """
    config: dict[str, JsonValue] = {
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
    vault_config: dict[str, JsonValue],
) -> AsyncGenerator[VaultHandler, None]:
    """Create and initialize VaultHandler for integration testing.

    Yields an initialized VaultHandler instance and ensures proper cleanup.

    Args:
        vault_config: Vault configuration fixture.

    Yields:
        Initialized VaultHandler instance.
    """
    from omnibase_infra.handlers import VaultHandler

    handler = VaultHandler()
    await handler.initialize(vault_config)

    yield handler

    # Cleanup: ensure handler is shutdown
    try:
        await handler.shutdown()
    except Exception:
        pass  # Ignore cleanup errors


# =============================================================================
# Consul Environment Configuration
# =============================================================================

# Read Consul configuration from environment (set via docker-compose or .env)
CONSUL_HOST = os.getenv("CONSUL_HOST")
CONSUL_PORT = int(os.getenv("CONSUL_PORT", "8500"))
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

    Returns True if Consul server is reachable at configured address.

    Returns:
        bool: True if Consul is available for testing.
    """
    return CONSUL_AVAILABLE


@pytest.fixture
def consul_config() -> dict[str, JsonValue]:
    """Provide Consul configuration for ConsulHandler.

    Returns:
        Configuration dict for ConsulHandler.initialize()
    """
    config: dict[str, JsonValue] = {
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
    consul_config: dict[str, JsonValue],
) -> AsyncGenerator[ConsulHandler, None]:
    """Provide an initialized ConsulHandler instance.

    Creates a ConsulHandler, initializes it with the test configuration,
    yields it for the test, then ensures proper cleanup via shutdown().

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
    try:
        await handler.shutdown()
    except Exception:
        pass  # Ignore cleanup errors
