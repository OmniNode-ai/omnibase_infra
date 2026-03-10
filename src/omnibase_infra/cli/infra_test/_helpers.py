# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Shared helpers for the infra-test CLI.

Centralises environment-resolution functions used by multiple subcommands
(``verify``, ``run``, ``introspect``).
"""

from __future__ import annotations

import os

from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)


def get_broker() -> str:
    """Resolve Kafka broker address from environment.

    Returns:
        Kafka bootstrap server address.
    """
    return os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"
    )  # kafka-fallback-ok — integration test default


def get_postgres_dsn() -> str:
    """Get PostgreSQL DSN from OMNIBASE_INFRA_DB_URL.

    Raises:
        ValueError: If OMNIBASE_INFRA_DB_URL is not set or invalid.

    Returns:
        Validated PostgreSQL connection string.
    """
    db_url = os.getenv("OMNIBASE_INFRA_DB_URL")
    if not db_url:
        msg = (
            "OMNIBASE_INFRA_DB_URL is required but not set. "
            "For host-side integration tests set: "
            "OMNIBASE_INFRA_DB_URL=postgresql://postgres:PASSWORD@localhost:5436/omnibase_infra "
            "(use localhost:5436 — the Docker-exposed port, not postgres:5432 which is Docker-internal only)."
        )
        raise ValueError(msg)

    return ModelPostgresPoolConfig.validate_dsn(db_url)
