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
from datetime import datetime
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.projectors.util_sql_identifiers import quote_identifier
from omnibase_infra.utils.util_datetime import ensure_timezone_aware

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

    def normalize_value(
        self,
        value: object,
        column_name: str | None = None,
    ) -> object:
        """Normalize a value before SQL persistence.

        Performs value normalization to ensure data consistency and prevent
        common issues when persisting to PostgreSQL:

        - **Datetime validation**: Ensures datetime values are timezone-aware.
          Naive datetimes (without tzinfo) are automatically converted to UTC
          with a warning log. This prevents subtle bugs when storing in
          TIMESTAMPTZ columns.

        - **Pass-through for other types**: Non-datetime values are returned
          unchanged.

        Args:
            value: The value to normalize. Can be any type.
            column_name: Optional column name for context in warning messages.
                Helps identify the source of naive datetimes in logs.

        Returns:
            The normalized value. For datetimes, returns a timezone-aware
            datetime. For other types, returns the original value unchanged.

        Example:
            >>> from datetime import datetime, UTC
            >>> mixin = MixinProjectorSqlOperations()
            >>>
            >>> # Aware datetime passes through
            >>> aware_dt = datetime.now(UTC)
            >>> mixin.normalize_value(aware_dt, "updated_at") == aware_dt
            True
            >>>
            >>> # Naive datetime is converted to UTC (with warning log)
            >>> naive_dt = datetime(2025, 1, 15, 12, 0, 0)
            >>> result = mixin.normalize_value(naive_dt, "created_at")
            >>> result.tzinfo is not None
            True
            >>>
            >>> # Non-datetime values pass through unchanged
            >>> mixin.normalize_value("test", "name")
            'test'
            >>> mixin.normalize_value(123, "count")
            123

        Warning:
            Naive datetimes are automatically converted to UTC to prevent
            database errors, but this may mask timezone bugs in your code.
            The warning log helps identify these issues. Prefer using
            ``datetime.now(UTC)`` explicitly in your code.

        Related:
            - ensure_timezone_aware: The underlying datetime validation utility
            - OMN-1170: Declarative contract projections
            - PR #146: Datetime validation improvements
        """
        # Handle datetime timezone validation
        if isinstance(value, datetime):
            return ensure_timezone_aware(
                value,
                assume_utc=True,
                warn_on_naive=True,
                context=column_name,
            )

        # Pass through other types unchanged
        return value

    def _normalize_values(
        self,
        values: dict[str, object],
    ) -> dict[str, object]:
        """Normalize all values in a dictionary before SQL persistence.

        Applies normalize_value() to each value in the dictionary, using the
        key as the column_name context for any warnings.

        Args:
            values: Dictionary of column names to values.

        Returns:
            New dictionary with all values normalized.

        Example:
            >>> from datetime import datetime, UTC
            >>> mixin = MixinProjectorSqlOperations()
            >>> values = {
            ...     "name": "test",
            ...     "created_at": datetime(2025, 1, 15),  # Naive - will be converted
            ...     "updated_at": datetime.now(UTC),      # Aware - passes through
            ... }
            >>> normalized = mixin._normalize_values(values)
            >>> normalized["created_at"].tzinfo is not None  # Now timezone-aware
            True
        """
        return {
            column: self.normalize_value(value, column_name=column)
            for column, value in values.items()
        }

    async def _upsert(
        self,
        values: dict[str, object],
        correlation_id: UUID,
        event_type: str | None = None,
    ) -> int:
        """Execute upsert (INSERT ON CONFLICT DO UPDATE).

        Uses the contract's upsert_key for conflict detection. Supports both
        single-column keys (str) and composite keys (list[str]). When all columns
        are part of the upsert key (i.e., no updatable columns), uses
        DO NOTHING to avoid generating invalid SQL with empty SET clause.

        Args:
            values: Column name to value mapping.
            correlation_id: Correlation ID for tracing.
            event_type: The event type being projected (for logging context).

        Returns:
            Number of rows affected.
        """
        # Normalize values (datetime timezone validation, etc.)
        values = self._normalize_values(values)

        schema = self._contract.projection_schema
        behavior = self._contract.behavior
        table_quoted = quote_identifier(schema.table)

        # Normalize upsert_key to list for uniform handling
        # schema.primary_key is a str field in omnibase_core.ModelProjectorSchema
        upsert_key = behavior.upsert_key or schema.primary_key
        upsert_key_list = [upsert_key] if isinstance(upsert_key, str) else upsert_key
        upsert_key_set = set(upsert_key_list)

        # Build quoted upsert key column list for ON CONFLICT clause
        upsert_key_quoted = ", ".join(quote_identifier(col) for col in upsert_key_list)

        # Build column lists
        columns = list(values.keys())
        if not columns:
            self._log_empty_columns_skip("upsert", event_type, correlation_id)
            return 0

        # Build parameterized INSERT ... ON CONFLICT DO UPDATE
        column_list = ", ".join(quote_identifier(col) for col in columns)
        param_list = ", ".join(f"${i + 1}" for i in range(len(columns)))
        # Exclude all upsert key columns from updatable columns
        updatable_columns = [col for col in columns if col not in upsert_key_set]

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
        event_type: str | None = None,
    ) -> int:
        """Execute INSERT (fail on conflict).

        Args:
            values: Column name to value mapping.
            correlation_id: Correlation ID for tracing.
            event_type: The event type being projected (for logging context).

        Returns:
            Number of rows affected.

        Raises:
            asyncpg.UniqueViolationError: On conflict (handled by caller
                based on projection mode).
        """
        # Normalize values (datetime timezone validation, etc.)
        values = self._normalize_values(values)

        schema = self._contract.projection_schema
        table_quoted = quote_identifier(schema.table)

        columns = list(values.keys())
        if not columns:
            self._log_empty_columns_skip("insert", event_type, correlation_id)
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
        event_type: str | None = None,
    ) -> int:
        """Execute INSERT (always append, event-log style).

        Similar to insert, but semantically indicates this is an
        append-only projection where conflicts are unexpected.

        Args:
            values: Column name to value mapping.
            correlation_id: Correlation ID for tracing.
            event_type: The event type being projected (for logging context).

        Returns:
            Number of rows affected.
        """
        # Implementation is same as insert - semantic difference only
        return await self._insert(values, correlation_id, event_type)

    def _log_empty_columns_skip(
        self,
        operation: str,
        event_type: str | None,
        correlation_id: UUID,
    ) -> None:
        """Log appropriate message when skipping due to empty columns.

        Differentiates between expected skips (on_event filtering) and
        unexpected skips (empty value extraction) to help identify
        configuration errors.

        Args:
            operation: The SQL operation being attempted (upsert/insert).
            event_type: The event type being projected.
            correlation_id: Correlation ID for tracing.
        """
        extra = {
            "projector_id": self.projector_id,
            "operation": operation,
            "correlation_id": str(correlation_id),
        }

        if event_type is None:
            # No event type context - use warning as we can't determine reason
            logger.warning(
                "No columns to %s - missing event type context",
                operation,
                extra=extra,
            )
            return

        extra["event_type"] = event_type
        schema = self._contract.projection_schema

        # Count columns that SHOULD have values for this event type
        # (no on_event filter OR matching on_event filter)
        expected_columns = [
            col
            for col in schema.columns
            if col.on_event is None or col.on_event == event_type
        ]

        if not expected_columns:
            # All columns have on_event filters that don't match this event type
            # This is expected behavior when using event-specific column filtering
            logger.info(
                "Skipping projection for event type '%s' - no columns match on_event filters",
                event_type,
                extra=extra,
            )
        else:
            # Columns should have values but all were extracted as empty
            # This likely indicates a configuration error (wrong source paths)
            logger.warning(
                "Skipping projection - all column values extracted as empty. "
                "Check source paths in contract for event type '%s'",
                event_type,
                extra=extra,
            )

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

    async def _partial_upsert(
        self,
        aggregate_id: UUID,
        values: dict[str, object],
        correlation_id: UUID,
        conflict_columns: list[str] | None = None,
    ) -> bool:
        """Execute a partial UPSERT (INSERT ON CONFLICT DO UPDATE) on specific columns.

        Inserts a new row if no row exists with the given conflict key(s), or updates
        only the specified columns if a row already exists. Supports both single and
        composite conflict keys for flexible schema support.

        This method is designed for state transition operations where:
        - A new entity may be created if it doesn't exist
        - An existing entity should be updated with new state

        Unlike partial_update() which only does UPDATE, partial_upsert() handles
        both INSERT and UPDATE cases atomically.

        Args:
            aggregate_id: The primary aggregate identifier (for logging).
            values: Dictionary mapping column names to their values.
                MUST include all conflict columns specified.
                Column names are validated and quoted for SQL safety.
                Values are passed as parameterized query arguments.
            correlation_id: Correlation ID for distributed tracing.
            conflict_columns: Optional list of column names for ON CONFLICT clause.
                If not provided, defaults to the contract's primary_key.
                Use this for composite unique constraints (e.g., ["entity_id", "domain"]).

        Returns:
            True if a row was inserted or updated successfully.
            False only if the upsert produced no rows (edge case).

        Raises:
            ProtocolConfigurationError: If values dict is empty or missing required
                conflict columns.

        Note:
            The values dict MUST include all conflict columns. This method
            is for cases where you want to upsert a subset of columns while
            ensuring the row exists.

        Example:
            >>> # Upsert with composite conflict key
            >>> result = await projector.partial_upsert(
            ...     aggregate_id=node_id,
            ...     values={
            ...         "entity_id": node_id,
            ...         "domain": "registration",
            ...         "current_state": "pending_registration",
            ...         "node_type": "effect",
            ...         "updated_at": datetime.now(UTC),
            ...     },
            ...     correlation_id=correlation_id,
            ...     conflict_columns=["entity_id", "domain"],
            ... )
        """
        if not values:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="partial_upsert",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "values dict cannot be empty for partial_upsert",
                context=context,
            )

        # Normalize values (datetime timezone validation, etc.)
        values = self._normalize_values(values)

        schema = self._contract.projection_schema
        # Normalize primary_key to list (handles both str and list[str] from omnibase_core)
        pk = schema.primary_key
        pk_list: list[str] = pk if isinstance(pk, list) else [pk]

        # Determine conflict columns (single or composite)
        conflict_cols = conflict_columns if conflict_columns else pk_list

        # Verify all conflict columns are in values
        for col in conflict_cols:
            if col not in values:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="partial_upsert",
                    correlation_id=correlation_id,
                )
                raise ProtocolConfigurationError(
                    f"values dict must include conflict column '{col}' for partial_upsert",
                    context=context,
                )

        table_quoted = quote_identifier(schema.table)

        # Build conflict clause (handles both single and composite keys)
        conflict_list = ", ".join(quote_identifier(col) for col in conflict_cols)

        # Build column lists
        columns = list(values.keys())
        column_list = ", ".join(quote_identifier(col) for col in columns)
        param_list = ", ".join(f"${i + 1}" for i in range(len(columns)))

        # Build UPDATE SET clause (exclude conflict columns from update)
        updatable_columns = [col for col in columns if col not in conflict_cols]

        # First conflict column for RETURNING clause
        first_conflict_quoted = quote_identifier(conflict_cols[0])

        if updatable_columns:
            update_list = ", ".join(
                f"{quote_identifier(col)} = EXCLUDED.{quote_identifier(col)}"
                for col in updatable_columns
            )
            # S608: Safe - identifiers quoted via quote_identifier(), not user input
            sql = f"""
                INSERT INTO {table_quoted} ({column_list})
                VALUES ({param_list})
                ON CONFLICT ({conflict_list}) DO UPDATE SET {update_list}
                RETURNING {first_conflict_quoted}
            """  # noqa: S608
        else:
            # Edge case: only conflict columns provided - DO NOTHING on conflict
            # NOTE: RETURNING with DO NOTHING returns NO ROWS when a conflict occurs
            # (i.e., row already exists). This is expected PostgreSQL behavior.
            # We handle this at the call site by returning True when result is None,
            # since for upsert semantics "row exists" is success.
            sql = f"""
                INSERT INTO {table_quoted} ({column_list})
                VALUES ({param_list})
                ON CONFLICT ({conflict_list}) DO NOTHING
                RETURNING {first_conflict_quoted}
            """  # noqa: S608

        params = list(values.values())

        logger.debug(
            "Executing partial upsert",
            extra={
                "projector_id": self.projector_id,
                "aggregate_id": str(aggregate_id),
                "columns": columns,
                "conflict_columns": conflict_cols,
                "correlation_id": str(correlation_id),
            },
        )

        async with self._pool.acquire() as conn:
            result = await conn.fetchrow(sql, *params, timeout=self._query_timeout)

        if result:
            logger.debug(
                "Partial upsert completed",
                extra={
                    "projector_id": self.projector_id,
                    "aggregate_id": str(aggregate_id),
                    "correlation_id": str(correlation_id),
                },
            )
            return True

        # DO NOTHING case - row already exists with same values
        logger.debug(
            "Partial upsert returned no result (DO NOTHING case)",
            extra={
                "projector_id": self.projector_id,
                "aggregate_id": str(aggregate_id),
                "correlation_id": str(correlation_id),
            },
        )
        return True  # Row exists, which is success for upsert semantics

    async def _partial_update(
        self,
        aggregate_id: UUID,
        updates: dict[str, object],
        correlation_id: UUID,
    ) -> bool:
        """Execute a partial UPDATE on specific columns.

        Updates only the specified columns for the row matching the aggregate_id.
        Uses the contract's primary_key for the WHERE clause.

        This method is designed for lightweight operations like:
        - Updating heartbeat timestamps
        - Setting timeout marker columns
        - Updating single fields without full row replacement

        Unlike project() which performs full event-driven projection, partial_update()
        directly updates specified columns without event type filtering or value
        extraction from event envelopes.

        Args:
            aggregate_id: The primary key value identifying the row to update.
            updates: Dictionary mapping column names to their new values.
                Column names are validated and quoted for SQL safety.
                Values are passed as parameterized query arguments.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            True if a row was updated (found and modified).
            False if no row was found matching the aggregate_id.

        Raises:
            ProtocolConfigurationError: If updates dict is empty, or if the
                schema has a composite primary key. Composite PK schemas
                cannot use this method because using only part of the key
                could cause unintended multi-row updates.

        Note:
            This method does NOT check whether column names exist in the contract
            schema - it trusts the caller to provide valid column names. This
            enables updating columns that may not be in the projection schema
            (e.g., internal tracking columns like updated_at).

            **Composite Primary Key Restriction**: Schemas with composite primary
            keys (e.g., ``["entity_id", "domain"]``) are explicitly rejected to
            prevent accidental multi-row updates. For composite key schemas, use
            ``_partial_upsert()`` with explicit ``conflict_columns`` parameter,
            or build a custom UPDATE query with a full WHERE clause.

        Example:
            >>> # Update heartbeat tracking fields
            >>> updated = await projector.partial_update(
            ...     aggregate_id=node_id,
            ...     updates={
            ...         "last_heartbeat_at": datetime.now(UTC),
            ...         "liveness_deadline": datetime.now(UTC) + timedelta(seconds=90),
            ...         "updated_at": datetime.now(UTC),
            ...     },
            ...     correlation_id=correlation_id,
            ... )
            >>> if not updated:
            ...     logger.warning("Entity not found for heartbeat update")
        """
        if not updates:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="partial_update",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "updates dict cannot be empty for partial_update",
                context=context,
            )

        # Normalize values (datetime timezone validation, etc.)
        updates = self._normalize_values(updates)

        schema = self._contract.projection_schema
        table_quoted = quote_identifier(schema.table)

        # Validate primary key is single-column to prevent multi-row updates
        pk = schema.primary_key
        if isinstance(pk, list) and len(pk) > 1:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="partial_update",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"partial_update does not support composite primary keys. "
                f"Schema '{schema.table}' has composite PK: {pk}. "
                f"Use _partial_upsert() with explicit conflict_columns parameter instead, "
                f"or build a custom UPDATE query with full WHERE clause.",
                context=context,
            )

        pk_col = pk[0] if isinstance(pk, list) else pk
        pk_quoted = quote_identifier(pk_col)

        # Build SET clause with parameterized placeholders
        # Column names are quoted for SQL safety; values use $1, $2, etc.
        columns = list(updates.keys())
        set_clauses = []
        for i, col in enumerate(columns):
            col_quoted = quote_identifier(col)
            set_clauses.append(f"{col_quoted} = ${i + 1}")

        set_clause = ", ".join(set_clauses)

        # Primary key is the last parameter
        pk_param_index = len(columns) + 1

        # Build UPDATE query
        # S608: Safe - identifiers quoted via quote_identifier(), values parameterized
        sql = f"""
            UPDATE {table_quoted}
            SET {set_clause}
            WHERE {pk_quoted} = ${pk_param_index}
        """  # noqa: S608

        # Build parameter list: values first, then aggregate_id
        params = list(updates.values()) + [aggregate_id]

        logger.debug(
            "Executing partial update",
            extra={
                "projector_id": self.projector_id,
                "aggregate_id": str(aggregate_id),
                "columns": columns,
                "correlation_id": str(correlation_id),
            },
        )

        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, *params, timeout=self._query_timeout)

        rows_affected = self._parse_row_count(result)

        if rows_affected > 0:
            logger.debug(
                "Partial update completed",
                extra={
                    "projector_id": self.projector_id,
                    "aggregate_id": str(aggregate_id),
                    "rows_affected": rows_affected,
                    "correlation_id": str(correlation_id),
                },
            )
            return True

        logger.debug(
            "Partial update found no matching row",
            extra={
                "projector_id": self.projector_id,
                "aggregate_id": str(aggregate_id),
                "correlation_id": str(correlation_id),
            },
        )
        return False


__all__ = [
    "MixinProjectorSqlOperations",
]
