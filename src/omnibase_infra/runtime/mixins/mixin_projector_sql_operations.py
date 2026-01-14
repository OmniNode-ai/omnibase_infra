# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""SQL Operation Mixin for Projector Implementations.

Provides SQL generation and execution methods for projector shells. This mixin
extracts database-specific operations from ProjectorShell to keep the main
class focused on projection logic and under the method count limit.

Features:
    - INSERT, UPSERT, APPEND operations
    - Parameterized SQL for injection protection
    - Configurable query timeouts
    - Row count parsing from asyncpg results

See Also:
    - ProjectorShell: Main projector class that uses this mixin
    - ModelProjectorContract: Contract model defining projection behavior

Related Tickets:
    - OMN-1169: ProjectorShell contract-driven projections

.. versionadded:: 0.7.0
    Extracted from ProjectorShell as part of OMN-1169 class decomposition.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

import asyncpg

from omnibase_infra.models.projectors.util_sql_identifiers import quote_identifier

if TYPE_CHECKING:
    from omnibase_core.models.projectors import (
        ModelProjectorContract,
    )

logger = logging.getLogger(__name__)


class ProtocolProjectorContext(Protocol):
    """Protocol for projector context required by SQL operations mixin.

    This protocol defines the minimum interface that a projector must
    implement to use MixinProjectorSqlOperations.
    """

    @property
    def _contract(self) -> ModelProjectorContract:
        """The projector contract defining projection behavior."""
        ...

    @property
    def _pool(self) -> asyncpg.Pool:
        """The asyncpg connection pool for database operations."""
        ...

    @property
    def _query_timeout(self) -> float:
        """Query timeout in seconds."""
        ...

    @property
    def projector_id(self) -> str:
        """Unique identifier for the projector."""
        ...


class MixinProjectorSqlOperations:
    """SQL operation mixin for projector implementations.

    Provides INSERT, UPSERT, and APPEND database operations with:
    - Parameterized SQL for injection protection
    - Empty SET clause handling for upsert edge cases
    - Configurable query timeouts
    - Row count parsing

    This mixin expects the implementing class to provide:
    - ``_contract``: ModelProjectorContract instance
    - ``_pool``: asyncpg.Pool for database connections
    - ``_query_timeout``: float timeout in seconds
    - ``projector_id``: str identifier for logging

    Example:
        >>> class ProjectorShell(MixinProjectorSqlOperations):
        ...     def __init__(self, contract, pool, timeout):
        ...         self._contract = contract
        ...         self._pool = pool
        ...         self._query_timeout = timeout
        ...
        ...     @property
        ...     def projector_id(self):
        ...         return str(self._contract.projector_id)

    Note:
        This mixin expects the implementing class to provide the attributes
        documented in ProtocolProjectorContext. The ``projector_id`` attribute
        is not declared here as it may be implemented as a property.
    """

    # Type hints for expected attributes from implementing class
    _contract: ModelProjectorContract
    _pool: asyncpg.Pool
    _query_timeout: float

    @property
    def projector_id(self) -> str:
        """Unique identifier for the projector (expected from implementing class)."""
        raise NotImplementedError("projector_id must be implemented by subclass")

    async def _upsert(
        self,
        values: dict[str, object],
        correlation_id: UUID,
    ) -> int:
        """Execute upsert (INSERT ON CONFLICT DO UPDATE).

        Uses the contract's upsert_key for conflict detection. When all columns
        are part of the upsert key (i.e., no updatable columns), uses
        DO NOTHING to avoid generating invalid SQL with empty SET clause.

        Args:
            values: Column name to value mapping.
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows affected.
        """
        schema = self._contract.projection_schema
        behavior = self._contract.behavior
        table_quoted = quote_identifier(schema.table)
        upsert_key = behavior.upsert_key or schema.primary_key
        upsert_key_quoted = quote_identifier(upsert_key)

        # Build column lists
        columns = list(values.keys())
        if not columns:
            logger.warning(
                "No columns to upsert",
                extra={
                    "projector_id": self.projector_id,
                    "correlation_id": str(correlation_id),
                },
            )
            return 0

        # Build parameterized INSERT ... ON CONFLICT DO UPDATE
        column_list = ", ".join(quote_identifier(col) for col in columns)
        param_list = ", ".join(f"${i + 1}" for i in range(len(columns)))
        updatable_columns = [col for col in columns if col != upsert_key]

        # S608: Safe - identifiers quoted via quote_identifier(), not user input
        if updatable_columns:
            # Normal case: columns to update on conflict
            update_list = ", ".join(
                f"{quote_identifier(col)} = EXCLUDED.{quote_identifier(col)}"
                for col in updatable_columns
            )
            sql = f"""
                INSERT INTO {table_quoted} ({column_list})
                VALUES ({param_list})
                ON CONFLICT ({upsert_key_quoted}) DO UPDATE SET {update_list}
            """  # noqa: S608
        else:
            # Edge case: all columns are part of primary key - no columns to update
            # Use DO NOTHING to avoid invalid SQL with empty SET clause
            sql = f"""
                INSERT INTO {table_quoted} ({column_list})
                VALUES ({param_list})
                ON CONFLICT ({upsert_key_quoted}) DO NOTHING
            """  # noqa: S608

        params = list(values.values())

        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, *params, timeout=self._query_timeout)

        # Parse row count from result (e.g., "INSERT 0 1" -> 1)
        return self._parse_row_count(result)

    async def _insert(
        self,
        values: dict[str, object],
        correlation_id: UUID,
    ) -> int:
        """Execute INSERT (fail on conflict).

        Args:
            values: Column name to value mapping.
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows affected.

        Raises:
            asyncpg.UniqueViolationError: On conflict (handled by caller
                based on projection mode).
        """
        schema = self._contract.projection_schema
        table_quoted = quote_identifier(schema.table)

        columns = list(values.keys())
        if not columns:
            logger.warning(
                "No columns to insert",
                extra={
                    "projector_id": self.projector_id,
                    "correlation_id": str(correlation_id),
                },
            )
            return 0

        column_list = ", ".join(quote_identifier(col) for col in columns)
        param_list = ", ".join(f"${i + 1}" for i in range(len(columns)))

        # S608: Safe - identifiers quoted via quote_identifier(), not user input
        sql = f"INSERT INTO {table_quoted} ({column_list}) VALUES ({param_list})"  # noqa: S608

        params = list(values.values())

        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, *params, timeout=self._query_timeout)

        return self._parse_row_count(result)

    async def _append(
        self,
        values: dict[str, object],
        correlation_id: UUID,
    ) -> int:
        """Execute INSERT (always append, event-log style).

        Similar to insert, but semantically indicates this is an
        append-only projection where conflicts are unexpected.

        Args:
            values: Column name to value mapping.
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows affected.
        """
        # Implementation is same as insert - semantic difference only
        return await self._insert(values, correlation_id)

    def _parse_row_count(self, result: str) -> int:
        """Parse row count from asyncpg execute result.

        Args:
            result: Result string from conn.execute (e.g., "INSERT 0 1").

        Returns:
            Number of rows affected.
        """
        # asyncpg returns strings like "INSERT 0 1", "UPDATE 3", etc.
        # The last number is the row count
        parts = result.split()
        if parts and parts[-1].isdigit():
            return int(parts[-1])
        return 0


__all__ = [
    "MixinProjectorSqlOperations",
]
