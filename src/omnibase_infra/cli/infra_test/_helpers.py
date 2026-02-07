# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared helpers for the infra-test CLI.

Centralises environment-resolution functions used by multiple subcommands
(``verify``, ``run``, ``introspect``).
"""

from __future__ import annotations

import os
from urllib.parse import quote_plus


def get_broker() -> str:
    """Resolve Kafka broker address from environment.

    Returns:
        Kafka bootstrap server address.
    """
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")


def get_consul_addr() -> str:
    """Resolve Consul HTTP address from environment.

    Returns:
        Full Consul HTTP URL (e.g. ``http://localhost:8500``).
    """
    host = os.getenv("CONSUL_HOST", "localhost")
    port = os.getenv("CONSUL_PORT", "8500")
    scheme = os.getenv("CONSUL_SCHEME", "http")
    return f"{scheme}://{host}:{port}"


def get_postgres_dsn() -> str:
    """Build PostgreSQL DSN from environment variables.

    Defaults are for local E2E test environments only -- never use in
    production.  All values (including the fallback password) are overridden
    via environment variables in real deployments.

    Returns:
        PostgreSQL connection string.
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    db = os.getenv("POSTGRES_DATABASE", "omninode_bridge")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "test-password")
    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"
