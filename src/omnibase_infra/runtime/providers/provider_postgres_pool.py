# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PostgreSQL connection pool provider.

Creates asyncpg connection pools from environment-driven configuration.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import logging
import ssl as ssl_lib

import asyncpg

from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)

logger = logging.getLogger(__name__)

# Modes requiring an explicit SSLContext built from ssl_ca_file, because
# asyncpg has no direct kwarg for a CA bundle path (only a DSN query param or
# the PGSSLROOTCERT env var, which this repo does not use for secrets/config).
_CONTEXT_SSL_MODES = frozenset({"verify-ca", "verify-full"})


def _resolve_ssl_context(
    config: ModelPostgresPoolConfig,
) -> ssl_lib.SSLContext | None:
    """Build an explicit SSLContext for verify-ca/verify-full sslmode.

    Returns:
        None for every other sslmode (including unset) — the caller falls
        back to the raw sslmode string for those (see ``config.ssl_mode``),
        which asyncpg resolves and verifies itself.
    """
    if config.ssl_mode not in _CONTEXT_SSL_MODES:
        return None
    # Validator on ModelPostgresPoolConfig guarantees ssl_ca_file is set here.
    context = ssl_lib.create_default_context(cafile=config.ssl_ca_file)
    context.check_hostname = config.ssl_mode == "verify-full"
    return context


class ProviderPostgresPool:
    """Creates and manages asyncpg connection pools.

    Pools are created from POSTGRES_* environment variables and shared
    across all contracts that declare postgres_pool dependencies.
    """

    def __init__(self, config: ModelPostgresPoolConfig) -> None:
        """Initialize the PostgreSQL pool provider.

        Args:
            config: PostgreSQL pool configuration (host, port, credentials, pool sizes).
        """
        self._config = config

    async def create(self) -> asyncpg.Pool:
        """Create an asyncpg connection pool.

        Returns:
            asyncpg.Pool instance.

        Raises:
            Exception: If pool creation fails (connection error, auth error, etc.)
        """
        logger.info(
            "Creating PostgreSQL connection pool",
            extra={
                "host": self._config.host,
                "port": self._config.port,
                "database": self._config.database,
                "min_size": self._config.min_size,
                "max_size": self._config.max_size,
                "ssl_mode": self._config.ssl_mode or "(unset)",
            },
        )

        # asyncpg resolves 'disable'/'allow'/'prefer'/'require' strings itself;
        # verify-ca/verify-full need the explicit SSLContext built above. ""
        # (unset) becomes None, preserving the pre-OMN-14597 call exactly.
        ssl_context = _resolve_ssl_context(self._config)
        ssl_param = (
            ssl_context if ssl_context is not None else self._config.ssl_mode or None
        )

        pool = await asyncpg.create_pool(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            database=self._config.database,
            min_size=self._config.min_size,
            max_size=self._config.max_size,
            ssl=ssl_param,
        )

        logger.info("PostgreSQL connection pool created successfully")
        return pool

    @staticmethod
    async def close(resource: asyncpg.Pool | None) -> None:
        """Close an asyncpg connection pool.

        Args:
            resource: The asyncpg.Pool to close.
        """
        if resource is not None and hasattr(resource, "close"):
            await resource.close()
            logger.info("PostgreSQL connection pool closed")


__all__ = ["ProviderPostgresPool"]
