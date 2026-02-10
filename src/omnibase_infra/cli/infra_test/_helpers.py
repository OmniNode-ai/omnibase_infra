# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared helpers for the infra-test CLI.

Centralises environment-resolution functions used by multiple subcommands
(``verify``, ``run``, ``introspect``).
"""

from __future__ import annotations

import os
from urllib.parse import urlparse


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
    """Get PostgreSQL DSN from OMNIBASE_INFRA_DB_URL.

    Raises:
        ValueError: If OMNIBASE_INFRA_DB_URL is not set (fail-fast).

    Returns:
        PostgreSQL connection string.
    """
    db_url = os.getenv("OMNIBASE_INFRA_DB_URL")
    if not db_url:
        msg = (
            "OMNIBASE_INFRA_DB_URL is required but not set. "
            "Set it to a PostgreSQL DSN, e.g. "
            "postgresql://user:pass@host:5432/omnibase_infra"
        )
        raise ValueError(msg)

    # Validate DSN scheme to catch obvious misconfigurations early
    if not db_url.startswith(("postgresql://", "postgres://")):
        msg = (
            f"OMNIBASE_INFRA_DB_URL has invalid scheme. "
            f"Expected 'postgresql://' or 'postgres://', "
            f"got: {urlparse(db_url).scheme or '(none)'}://"
        )
        raise ValueError(msg)

    # Validate DSN contains a database name (path component)
    parsed = urlparse(db_url)
    database = (parsed.path or "").lstrip("/")
    if not database:
        msg = (
            "OMNIBASE_INFRA_DB_URL is missing a database name. "
            "Example: postgresql://user:pass@host:5432/omnibase_infra"
        )
        raise ValueError(msg)

    return db_url
