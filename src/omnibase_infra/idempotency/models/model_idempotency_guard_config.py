# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Idempotency Guard Configuration Model.

This module provides the Pydantic configuration model for the idempotency
guard, which controls how idempotency checking is applied to message handlers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelIdempotencyGuardConfig(BaseModel):
    """Configuration for the idempotency guard.

    The idempotency guard is a decorator or middleware that automatically
    applies idempotency checking to message handlers. This configuration
    controls its behavior, including which store backend to use and which
    operations to skip.

    Attributes:
        enabled: Whether idempotency checking is enabled.
            When False, all messages are processed without deduplication.
            Default: True.
        store_type: Type of idempotency store to use.
            "postgres" uses PostgreSQL for persistent storage.
            "memory" uses in-memory storage (for testing only).
            Default: "postgres".
        domain_from_operation: Whether to extract domain from operation prefix.
            When True, operations like "db.query" will use "db" as the domain.
            This enables per-domain idempotency partitioning.
            Default: True.
        skip_operations: List of operations to skip idempotency checking for.
            Useful for health checks, metrics endpoints, or other operations
            that should always be processed.
            Default: [].

    Example:
        >>> config = ModelIdempotencyGuardConfig(
        ...     enabled=True,
        ...     store_type="postgres",
        ...     domain_from_operation=True,
        ...     skip_operations=["health.check", "metrics.collect"],
        ... )
        >>> print(config.store_type)
        postgres

        >>> # Check if an operation should be skipped
        >>> "health.check" in config.skip_operations
        True
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    enabled: bool = Field(
        default=True,
        description="Whether idempotency checking is enabled",
    )
    store_type: Literal["postgres", "memory"] = Field(
        default="postgres",
        description="Type of idempotency store to use ('postgres' for production, 'memory' for testing)",
    )
    domain_from_operation: bool = Field(
        default=True,
        description="Extract domain from operation prefix (e.g., 'db.query' -> 'db')",
    )
    skip_operations: list[str] = Field(
        default_factory=list,
        description="Operations to skip idempotency checking for (e.g., health checks)",
    )

    def should_check_idempotency(self, operation: str) -> bool:
        """Determine if idempotency should be checked for an operation.

        Args:
            operation: The operation name to check.

        Returns:
            True if idempotency should be checked, False if skipped.

        Example:
            >>> config = ModelIdempotencyGuardConfig(
            ...     enabled=True,
            ...     skip_operations=["health.check"],
            ... )
            >>> config.should_check_idempotency("db.query")
            True
            >>> config.should_check_idempotency("health.check")
            False
        """
        if not self.enabled:
            return False
        return operation not in self.skip_operations

    def extract_domain(self, operation: str) -> str | None:
        """Extract domain from operation name if configured.

        Args:
            operation: The operation name (e.g., "db.query", "orders.create").

        Returns:
            The extracted domain if domain_from_operation is True and
            operation contains a dot separator, otherwise None.

        Example:
            >>> config = ModelIdempotencyGuardConfig(domain_from_operation=True)
            >>> config.extract_domain("db.query")
            'db'
            >>> config.extract_domain("simple_operation")
            None

            >>> config_no_domain = ModelIdempotencyGuardConfig(domain_from_operation=False)
            >>> config_no_domain.extract_domain("db.query")
            None
        """
        if not self.domain_from_operation:
            return None
        if "." in operation:
            return operation.split(".")[0]
        return None


__all__: list[str] = ["ModelIdempotencyGuardConfig"]
