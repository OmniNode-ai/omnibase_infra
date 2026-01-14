# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Generic Contract-Driven Projector Shell.

Implements ProtocolEventProjector for contract-based event-to-state projection.
All behavior is driven by ModelProjectorContract - NO domain-specific logic.

Features:
    - Event type matching via envelope metadata, payload attribute, or classname
    - Dynamic column value extraction from nested event payloads
    - Three projection modes: upsert, insert_only, append
    - Parameterized SQL queries for injection protection
    - Bulk state queries for N+1 optimization
    - Configurable query timeouts

See Also:
    - ProtocolEventProjector: Protocol definition from omnibase_infra.protocols
    - ModelProjectorContract: Contract model from omnibase_core
    - ProjectorPluginLoader: Loader that instantiates ProjectorShell

Related Tickets:
    - OMN-1169: ProjectorShell contract-driven projections (implemented)

.. versionadded:: 0.7.0
    Created as part of OMN-1169 projector shell implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

import asyncpg
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.projectors import (
    ModelProjectionResult,
    ModelProjectorContract,
)
from pydantic import BaseModel

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.models.projectors.util_sql_identifiers import quote_identifier

if TYPE_CHECKING:
    from omnibase_core.models.projectors.model_projector_column import (
        ModelProjectorColumn,
    )

logger = logging.getLogger(__name__)


class ProjectorShell:
    """Generic contract-driven projector implementation.

    Transforms events into persistent state projections based on a
    ModelProjectorContract definition. All behavior is declarative -
    no domain-specific logic in this class.

    The projector supports three projection modes:
        - upsert: INSERT or UPDATE based on upsert_key (default)
        - insert_only: INSERT only, fail on conflict
        - append: Always INSERT, event-log style

    Thread Safety:
        This implementation is coroutine-safe for concurrent async calls.
        Uses asyncpg connection pool for connection management.

    Security:
        All queries use parameterized statements for SQL injection protection.
        Table and column names are validated by the contract model validators
        and quoted using ``quote_identifier()`` for safe SQL generation.

    Query Timeout:
        Configurable via ``query_timeout_seconds`` parameter. Defaults to 30
        seconds. Set to None to disable timeout (not recommended for production).

    Default Values:
        Column defaults specified in the contract (``column.default``) are treated
        as **runtime literal values**, not SQL expressions. They are inserted as
        parameter values, not embedded in SQL. For database-level defaults (e.g.,
        ``CURRENT_TIMESTAMP``), use PostgreSQL column defaults instead.

    Example:
        >>> from omnibase_core.models.projectors import ModelProjectorContract
        >>> contract = ModelProjectorContract.model_validate(yaml_data)
        >>> pool = await asyncpg.create_pool(dsn)
        >>> projector = ProjectorShell(contract, pool, query_timeout_seconds=10.0)
        >>> result = await projector.project(event_envelope, correlation_id)
        >>> if result.success:
        ...     print(f"Projected {result.rows_affected} rows")

    Related:
        - OMN-1169: ProjectorShell contract-driven projections (implemented)
        - OMN-1168: ProjectorPluginLoader contract discovery
    """

    # Default query timeout in seconds (30s is reasonable for projections)
    DEFAULT_QUERY_TIMEOUT_SECONDS: float = 30.0

    def __init__(
        self,
        contract: ModelProjectorContract,
        pool: asyncpg.Pool,
        query_timeout_seconds: float | None = None,
    ) -> None:
        """Initialize projector shell with contract and database pool.

        Args:
            contract: The projector contract defining projection behavior.
                All projection rules (table, columns, modes) come from this.
            pool: asyncpg connection pool for database access.
                Pool should be created by the caller (e.g., from container).
            query_timeout_seconds: Timeout for individual database queries in
                seconds. Defaults to 30.0 seconds. Set to None to disable
                timeout (not recommended for production).
        """
        self._contract = contract
        self._pool = pool
        self._query_timeout = (
            query_timeout_seconds
            if query_timeout_seconds is not None
            else self.DEFAULT_QUERY_TIMEOUT_SECONDS
        )
        logger.debug(
            "ProjectorShell initialized for projector '%s'",
            contract.projector_id,
            extra={
                "projector_id": contract.projector_id,
                "aggregate_type": contract.aggregate_type,
                "consumed_events": contract.consumed_events,
                "mode": contract.behavior.mode,
                "query_timeout_seconds": self._query_timeout,
            },
        )

    @property
    def projector_id(self) -> str:
        """Unique identifier from contract."""
        return str(self._contract.projector_id)

    @property
    def aggregate_type(self) -> str:
        """Aggregate type from contract."""
        return str(self._contract.aggregate_type)

    @property
    def consumed_events(self) -> list[str]:
        """Event types from contract."""
        return list(self._contract.consumed_events)

    @property
    def contract(self) -> ModelProjectorContract:
        """Access the underlying contract."""
        return self._contract

    @property
    def is_placeholder(self) -> bool:
        """Whether this is a placeholder implementation.

        Returns:
            False, as this is a full implementation.
        """
        return False

    async def project(
        self,
        event: ModelEventEnvelope[object],
        correlation_id: UUID,
    ) -> ModelProjectionResult:
        """Project event to persistence store.

        Transforms the event into a database row based on the contract
        configuration. The projection mode (upsert, insert_only, append)
        determines how conflicts are handled.

        Args:
            event: The event envelope to project. The payload is accessed
                via dot-notation paths defined in the contract columns.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            ModelProjectionResult with success status and rows affected.
            Returns skipped=True if event type is not in consumed_events.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If projection times out.
            RuntimeHostError: For other database errors.
        """
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="project",
            target_name=f"projector.{self.projector_id}",
            correlation_id=correlation_id,
        )

        # Get event type from envelope
        event_type = self._get_event_type(event)

        # Check if this projector consumes this event type
        if event_type not in self._contract.consumed_events:
            logger.debug(
                "Skipping event type '%s' not in consumed_events",
                event_type,
                extra={
                    "projector_id": self.projector_id,
                    "event_type": event_type,
                    "consumed_events": self._contract.consumed_events,
                    "correlation_id": str(correlation_id),
                },
            )
            return ModelProjectionResult(success=True, skipped=True, rows_affected=0)

        # Extract column values from event
        values = self._extract_values(event, event_type)

        # Execute projection based on mode
        try:
            rows_affected = await self._execute_projection(values, correlation_id)

            logger.debug(
                "Projection completed",
                extra={
                    "projector_id": self.projector_id,
                    "event_type": event_type,
                    "rows_affected": rows_affected,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelProjectionResult(
                success=True,
                skipped=False,
                rows_affected=rows_affected,
            )

        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                f"Failed to connect to database for projection: {self.projector_id}",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            raise InfraTimeoutError(
                f"Projection timed out for: {self.projector_id}",
                context=ctx,
            ) from e

        except asyncpg.UniqueViolationError as e:
            # Only handle UniqueViolationError for insert_only mode
            # For upsert mode, this should never happen (ON CONFLICT handles it)
            # For append mode, this indicates an unexpected duplicate
            if self._contract.behavior.mode == "insert_only":
                logger.warning(
                    "Unique constraint violation during insert_only projection",
                    extra={
                        "projector_id": self.projector_id,
                        "event_type": event_type,
                        "correlation_id": str(correlation_id),
                    },
                )
                return ModelProjectionResult(
                    success=False,
                    skipped=False,
                    rows_affected=0,
                    error="Unique constraint violation: duplicate key for insert_only mode",
                )
            # Wrap as RuntimeHostError for other modes - unexpected behavior
            # should not expose raw asyncpg exceptions at runtime boundary
            raise RuntimeHostError(
                f"Unexpected unique constraint violation in "
                f"{self._contract.behavior.mode} mode for projector: {self.projector_id}",
                context=ctx,
            ) from e

        except Exception as e:
            raise RuntimeHostError(
                f"Failed to execute projection: {type(e).__name__}",
                context=ctx,
            ) from e

    async def get_state(
        self,
        aggregate_id: UUID,
        correlation_id: UUID,
    ) -> dict[str, object] | None:
        """Get current projected state for an aggregate.

        Queries the projection table for the current state of the
        specified aggregate. Uses configurable query timeout.

        Args:
            aggregate_id: The unique identifier of the aggregate.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Dictionary mapping column names (str) to their values if found,
            None if no state exists. Values are typed as ``object`` because
            asyncpg can return various PostgreSQL types (str, int, float,
            datetime, UUID, etc.) and the schema is defined at runtime via
            the projector contract.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If query times out.
            RuntimeHostError: For other database errors.
        """
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_state",
            target_name=f"projector.{self.projector_id}",
            correlation_id=correlation_id,
        )

        schema = self._contract.projection_schema
        table_quoted = quote_identifier(schema.table)
        pk_quoted = quote_identifier(schema.primary_key)

        # Build SELECT query - table/column names from trusted contract
        # S608: Safe - identifiers quoted via quote_identifier(), not user input
        query = f"SELECT * FROM {table_quoted} WHERE {pk_quoted} = $1"  # noqa: S608

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    query, aggregate_id, timeout=self._query_timeout
                )

            if row is None:
                logger.debug(
                    "No state found for aggregate",
                    extra={
                        "projector_id": self.projector_id,
                        "aggregate_id": str(aggregate_id),
                        "correlation_id": str(correlation_id),
                    },
                )
                return None

            # Convert asyncpg Record to dict
            result: dict[str, object] = dict(row)
            logger.debug(
                "State retrieved for aggregate",
                extra={
                    "projector_id": self.projector_id,
                    "aggregate_id": str(aggregate_id),
                    "correlation_id": str(correlation_id),
                },
            )
            return result

        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                f"Failed to connect to database for state query: {self.projector_id}",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            raise InfraTimeoutError(
                f"State query timed out for: {self.projector_id}",
                context=ctx,
            ) from e

        except Exception as e:
            raise RuntimeHostError(
                f"Failed to get state: {type(e).__name__}",
                context=ctx,
            ) from e

    async def get_states(
        self,
        aggregate_ids: list[UUID],
        correlation_id: UUID,
    ) -> dict[UUID, dict[str, object]]:
        """Get current projected states for multiple aggregates.

        Bulk query for N+1 optimization. Fetches states for all provided
        aggregate IDs in a single database query.

        Args:
            aggregate_ids: List of unique aggregate identifiers to query.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Dictionary mapping aggregate_id (UUID) to its state (dict).
            Aggregates with no state are omitted from the result.
            Empty dict if no aggregate_ids provided or none found.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If query times out.
            RuntimeHostError: For other database errors.

        Example:
            >>> states = await projector.get_states(
            ...     [order_id_1, order_id_2, order_id_3],
            ...     correlation_id,
            ... )
            >>> for order_id, state in states.items():
            ...     print(f"Order {order_id}: {state['status']}")
        """
        if not aggregate_ids:
            return {}

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_states",
            target_name=f"projector.{self.projector_id}",
            correlation_id=correlation_id,
        )

        schema = self._contract.projection_schema
        table_quoted = quote_identifier(schema.table)
        pk_quoted = quote_identifier(schema.primary_key)

        # Build SELECT query with IN clause for bulk fetch
        # S608: Safe - identifiers quoted via quote_identifier(), not user input
        query = f"SELECT * FROM {table_quoted} WHERE {pk_quoted} = ANY($1)"  # noqa: S608

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    query, aggregate_ids, timeout=self._query_timeout
                )

            # Build result dict keyed by aggregate ID
            result: dict[UUID, dict[str, object]] = {}
            pk_column = schema.primary_key
            for row in rows:
                row_dict: dict[str, object] = dict(row)
                aggregate_id = row_dict.get(pk_column)
                if isinstance(aggregate_id, UUID):
                    result[aggregate_id] = row_dict

            logger.debug(
                "Bulk state retrieval completed",
                extra={
                    "projector_id": self.projector_id,
                    "requested_count": len(aggregate_ids),
                    "found_count": len(result),
                    "correlation_id": str(correlation_id),
                },
            )
            return result

        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                f"Failed to connect to database for bulk state query: {self.projector_id}",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            raise InfraTimeoutError(
                f"Bulk state query timed out for: {self.projector_id}",
                context=ctx,
            ) from e

        except Exception as e:
            raise RuntimeHostError(
                f"Failed to get states: {type(e).__name__}",
                context=ctx,
            ) from e

    def _get_event_type(self, event: ModelEventEnvelope[object]) -> str:
        """Extract event type from envelope.

        Event type is resolved in the following order:
        1. envelope.metadata.tags['event_type'] if present
        2. payload.event_type attribute if present
        3. payload class name

        Args:
            event: The event envelope to extract type from.

        Returns:
            Event type string.
        """
        # Check metadata tags first
        if event.metadata and event.metadata.tags:
            event_type_tag = event.metadata.tags.get("event_type")
            if event_type_tag is not None:
                return str(event_type_tag)

        # Check payload attribute
        payload = event.payload
        if hasattr(payload, "event_type"):
            event_type_attr = payload.event_type
            if event_type_attr:
                return str(event_type_attr)

        # Fall back to class name
        return type(payload).__name__

    def _extract_values(
        self,
        event: ModelEventEnvelope[object],
        event_type: str,
    ) -> dict[str, object]:
        """Extract column values from event based on contract schema.

        Iterates through the contract's column definitions and resolves
        each column's source path to extract the value from the event.

        Args:
            event: The event envelope containing the data.
            event_type: The resolved event type for filtering.

        Returns:
            Dictionary mapping column names to their extracted values.
        """
        values: dict[str, object] = {}
        schema = self._contract.projection_schema

        for column in schema.columns:
            # Skip columns with on_event filter that doesn't match
            if column.on_event is not None and column.on_event != event_type:
                continue

            # Resolve the source path
            value = self._resolve_path(event, column.source)

            # Use default if value is None and default is specified
            if value is None and column.default is not None:
                value = column.default

            values[column.name] = value

        return values

    def _resolve_path(
        self,
        root: object,
        path: str,
    ) -> object | None:
        """Resolve a dot-notation path to extract a value.

        Supports navigation through:
        - Dictionary keys
        - Object attributes
        - Pydantic model fields (via model_dump())

        Args:
            root: The root object to start navigation from.
            path: Dot-notation path (e.g., "event.payload.node_name").

        Returns:
            The resolved value, or None if path resolution fails.

        Example:
            >>> event = ModelEventEnvelope(payload={"node_name": "test"})
            >>> self._resolve_path(event, "payload.node_name")
            'test'
        """
        parts = path.split(".")
        current: object = root

        for part in parts:
            if current is None:
                return None

            # Try dictionary access
            if isinstance(current, dict):
                current = current.get(part)
                continue

            # Try attribute access first (avoids Pydantic model_dump side effects)
            if hasattr(current, part):
                current = getattr(current, part)
                continue

            # Fall back to Pydantic model_dump for nested access
            if isinstance(current, BaseModel):
                dumped = current.model_dump()
                current = dumped.get(part)
                continue

            # Path resolution failed
            logger.debug(
                "Path resolution failed at part '%s'",
                part,
                extra={
                    "path": path,
                    "current_type": type(current).__name__,
                },
            )
            return None

        return current

    async def _execute_projection(
        self,
        values: dict[str, object],
        correlation_id: UUID,
    ) -> int:
        """Execute the projection based on behavior mode.

        Dispatches to the appropriate SQL execution method based on
        the contract's behavior.mode setting.

        Args:
            values: Column name to value mapping.
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows affected.

        Raises:
            asyncpg exceptions on database errors.
        """
        mode = self._contract.behavior.mode

        if mode == "upsert":
            return await self._upsert(values, correlation_id)
        elif mode == "insert_only":
            return await self._insert(values, correlation_id)
        elif mode == "append":
            return await self._append(values, correlation_id)
        else:
            # This should never happen due to contract validation
            raise ValueError(f"Unknown projection mode: {mode}")

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

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ProjectorShell("
            f"id={self.projector_id!r}, "
            f"aggregate_type={self.aggregate_type!r}, "
            f"events={len(self.consumed_events)}, "
            f"mode={self._contract.behavior.mode!r})"
        )


__all__ = [
    "ProjectorShell",
]
