# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Central configuration for test infrastructure endpoints.

This module provides a single source of truth for infrastructure server
configuration used in integration tests. It enables:

1. **Environment Variable Override**: All infrastructure endpoints can be
   overridden via environment variables for different deployment scenarios.

2. **Graceful Skip Behavior**: Tests skip gracefully when infrastructure
   is unavailable (common in CI/CD pipelines without VPN access).

3. **Documentation**: Clear documentation of the development infrastructure
   server and its purpose.

Development Infrastructure Server
=================================

The default values point to the ONEX development/staging infrastructure
server, which hosts shared services for integration testing:

    - PostgreSQL: Database for state persistence
    - Consul: Service discovery and configuration
    - Vault: Secret management
    - Kafka/Redpanda: Event streaming

This server provides a shared development environment for integration testing
against real infrastructure components. It is NOT accessible from public CI/CD
runners (e.g., GitHub Actions) without VPN access.

Environment Variable Overrides
==============================

For local development or alternative infrastructure, set these environment
variables to override the defaults:

    REMOTE_INFRA_HOST: Override the infrastructure server IP
        Default: localhost (CI-friendly default)
        Example: REMOTE_INFRA_HOST=your-server-ip

    Individual service overrides (take precedence over REMOTE_INFRA_HOST):
        POSTGRES_HOST: PostgreSQL server hostname
        CONSUL_HOST: Consul server hostname
        VAULT_ADDR: Vault server URL (full URL including scheme and port)
        KAFKA_BOOTSTRAP_SERVERS: Kafka bootstrap servers

CI/CD Graceful Skip Behavior
============================

When infrastructure is unavailable (e.g., CI/CD without VPN):

    1. Tests using infrastructure fixtures will be skipped automatically
    2. Skip messages indicate which environment variable to set
    3. No test failures - just skipped tests with clear reasons

Example Output in CI::

    $ pytest tests/integration/handlers/ -v
    test_db_health_check SKIPPED (PostgreSQL not available - POSTGRES_PASSWORD not set)
    test_vault_health_check SKIPPED (Vault not available - VAULT_TOKEN not set)
    test_consul_health_check SKIPPED (Consul not available - cannot connect)

Usage in Tests
==============

Import the configuration constants from this module:

    >>> from tests.infrastructure_config import (
    ...     REMOTE_INFRA_HOST,
    ...     DEFAULT_POSTGRES_PORT,
    ...     DEFAULT_CONSUL_PORT,
    ...     DEFAULT_VAULT_PORT,
    ... )

For environment-aware configuration with fallbacks:

    >>> from tests.infrastructure_config import get_postgres_host
    >>> host = get_postgres_host()  # Returns POSTGRES_HOST or REMOTE_INFRA_HOST

Related Files
=============

    - tests/integration/handlers/conftest.py: Uses these constants for fixtures
    - tests/integration/handlers/README.md: Documents infrastructure setup
    - .env.example: Template for environment configuration
"""

from __future__ import annotations

import os

# =============================================================================
# Remote Infrastructure Server Configuration
# =============================================================================
# The ONEX development infrastructure server hosts shared services:
#   - PostgreSQL (port 5436)
#   - Consul (port 28500)
#   - Vault (port 8200)
#   - Kafka/Redpanda (port 29092)
#
# This server provides a shared development environment for integration testing
# against real infrastructure components. For CI/CD environments without access
# to this server, tests will skip gracefully.
#
# To override for local development or alternative infrastructure:
#   export REMOTE_INFRA_HOST=localhost
#   export REMOTE_INFRA_HOST=your-server-ip
# =============================================================================

# Default infrastructure server host for CI environments
# Override with REMOTE_INFRA_HOST environment variable
# Default is "localhost" for CI compatibility - tests skip gracefully when unreachable
# For ONEX dev server access, set: export REMOTE_INFRA_HOST=<your-infra-server-ip>
_DEFAULT_REMOTE_INFRA_HOST = "localhost"

REMOTE_INFRA_HOST: str = os.getenv("REMOTE_INFRA_HOST", _DEFAULT_REMOTE_INFRA_HOST)
"""Infrastructure server hostname/IP.

This is the primary infrastructure server used for integration tests.
Can be overridden via the REMOTE_INFRA_HOST environment variable.

Default: localhost (CI-friendly default, tests skip when unreachable)
"""

# =============================================================================
# Default Service Ports
# =============================================================================
# These are the standard ports for each service on the infrastructure server.
# Individual service configurations may use different ports - check .env.example.

DEFAULT_POSTGRES_PORT: int = 5436
"""PostgreSQL port on the infrastructure server (external port)."""

DEFAULT_CONSUL_PORT: int = 28500
"""Consul HTTP API port on the infrastructure server."""

DEFAULT_VAULT_PORT: int = 8200
"""Vault HTTP API port on the infrastructure server."""

DEFAULT_KAFKA_PORT: int = 29092
"""Kafka/Redpanda external port on the infrastructure server."""


# =============================================================================
# Helper Functions for Environment-Aware Configuration
# =============================================================================


def get_postgres_host() -> str | None:
    """Get PostgreSQL host with fallback to infrastructure server.

    Resolution order:
        1. POSTGRES_HOST environment variable (if set)
        2. None (to indicate not configured - tests should skip)

    Note: Does NOT fall back to REMOTE_INFRA_HOST automatically.
    Tests should explicitly require POSTGRES_HOST to be set.

    Returns:
        PostgreSQL hostname if configured, None otherwise.
    """
    return os.getenv("POSTGRES_HOST")


def get_consul_host() -> str | None:
    """Get Consul host with fallback to infrastructure server.

    Resolution order:
        1. CONSUL_HOST environment variable (if set)
        2. None (to indicate not configured - tests should skip)

    Note: Does NOT fall back to REMOTE_INFRA_HOST automatically.
    Tests should explicitly require CONSUL_HOST to be set.

    Returns:
        Consul hostname if configured, None otherwise.
    """
    return os.getenv("CONSUL_HOST")


def get_vault_addr() -> str | None:
    """Get Vault address (full URL).

    Resolution order:
        1. VAULT_ADDR environment variable (if set)
        2. None (to indicate not configured - tests should skip)

    Note: Does NOT fall back to REMOTE_INFRA_HOST automatically.
    Tests should explicitly require VAULT_ADDR to be set.

    Returns:
        Vault URL (e.g., 'http://localhost:8200') if configured, None otherwise.
    """
    return os.getenv("VAULT_ADDR")


def get_kafka_bootstrap_servers() -> str | None:
    """Get Kafka bootstrap servers.

    Resolution order:
        1. KAFKA_BOOTSTRAP_SERVERS environment variable (if set)
        2. None (to indicate not configured - tests should skip)

    Note: Does NOT fall back to REMOTE_INFRA_HOST automatically.
    Tests should explicitly require KAFKA_BOOTSTRAP_SERVERS to be set.

    Returns:
        Kafka bootstrap servers (e.g., 'localhost:29092') if configured,
        None otherwise.
    """
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS")


def build_default_vault_url() -> str:
    """Build default Vault URL using infrastructure server.

    Uses REMOTE_INFRA_HOST and DEFAULT_VAULT_PORT to construct a Vault URL.
    This is useful for documentation and examples.

    Returns:
        Default Vault URL string.

    Example:
        >>> build_default_vault_url()
        'http://localhost:8200'
    """
    return f"http://{REMOTE_INFRA_HOST}:{DEFAULT_VAULT_PORT}"


def build_default_kafka_servers() -> str:
    """Build default Kafka bootstrap servers string.

    Uses REMOTE_INFRA_HOST and DEFAULT_KAFKA_PORT to construct the bootstrap
    servers string. This is useful for documentation and examples.

    Returns:
        Default Kafka bootstrap servers string.

    Example:
        >>> build_default_kafka_servers()
        'localhost:29092'
    """
    return f"{REMOTE_INFRA_HOST}:{DEFAULT_KAFKA_PORT}"
