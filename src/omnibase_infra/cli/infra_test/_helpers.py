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
    if scheme not in ("http", "https"):
        raise ValueError(f"CONSUL_SCHEME must be 'http' or 'https', got {scheme!r}.")
    if not port.isdigit():
        raise ValueError(f"CONSUL_PORT must be numeric, got {port!r}.")
    return f"{scheme}://{host}:{port}"


def get_postgres_dsn() -> str:
    """Build PostgreSQL DSN from environment variables.

    Resolution order:
        1. ``OMNIBASE_INFRA_DB_URL`` -- full DSN, returned as-is.
        2. Individual ``POSTGRES_*`` variables assembled into a DSN.

    Defaults are for local E2E test environments only -- never use in
    production.  All values (including the fallback password) are overridden
    via environment variables in real deployments.

    Returns:
        PostgreSQL connection string.

    Raises:
        ValueError: If no database name is configured or inputs are invalid.
    """
    # Prefer explicit DSN when available (OMN-2146)
    db_url = os.getenv("OMNIBASE_INFRA_DB_URL", "")
    if db_url:
        return db_url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    db = os.getenv("POSTGRES_DATABASE", "")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "test-password")
    if not db:
        raise ValueError(
            "No database configured. Set OMNIBASE_INFRA_DB_URL or POSTGRES_DATABASE."
        )
    if "@" in host:
        raise ValueError(
            f"POSTGRES_HOST contains '@' ({host!r}), which would produce a malformed DSN."
        )
    if not port.isdigit():
        raise ValueError(f"POSTGRES_PORT must be numeric, got {port!r}.")
    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{quote_plus(db)}"
