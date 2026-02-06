# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL connection pool provider.

Creates asyncpg connection pools from environment-driven configuration.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)

logger = logging.getLogger(__name__)


class ProviderPostgresPool:
    """Creates and manages asyncpg connection pools.

    Pools are created from POSTGRES_* environment variables and shared
    across all contracts that declare postgres_pool dependencies.
    """

    def __init__(self, config: ModelPostgresPoolConfig) -> None:
        self._config = config

    # ONEX_EXCLUDE: any_type - returns asyncpg.Pool which is not a standard type
    async def create(self) -> Any:
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
            },
        )

        pool = await asyncpg.create_pool(
            self._config.dsn,
            min_size=self._config.min_size,
            max_size=self._config.max_size,
        )

        logger.info("PostgreSQL connection pool created successfully")
        return pool

    @staticmethod
    # ONEX_EXCLUDE: any_type - resource is asyncpg.Pool, typed as Any for provider interface
    async def close(resource: Any) -> None:
        """Close an asyncpg connection pool.

        Args:
            resource: The asyncpg.Pool to close.
        """
        if resource is not None and hasattr(resource, "close"):
            await resource.close()
            logger.info("PostgreSQL connection pool closed")


__all__ = ["ProviderPostgresPool"]
