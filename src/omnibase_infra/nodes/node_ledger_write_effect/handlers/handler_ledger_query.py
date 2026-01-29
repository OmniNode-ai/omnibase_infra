# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for ledger query operations with internal routing.

This handler provides query operations for the event ledger, supporting
queries by correlation_id and time_range. Both operations share:
    - Input validation and normalization
    - DB connection/session lifecycle (via HandlerDb composition)
    - Pagination and ordering rules
    - Error mapping and handling
    - Consistent response surface

The operation suffix drives internal routing to private query methods.

Design Decision - Single Handler with Internal Routing:
    Two handlers looks "clean" until you realize you now have to duplicate:
    validation, DB session wiring, paging defaults, error mapping, metrics,
    tracing, and auth checks. That's the stuff that actually rots. The query
    shape is the only thing that differs.

    Only split into two handlers if the two modes diverge materially in
    non-shared behavior (different indexes, different auth model, different
    response shape, different pagination contract).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.nodes.node_ledger_write_effect.models import (
    ModelLedgerEntry,
    ModelLedgerQuery,
    ModelLedgerQueryResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.handlers.handler_db import HandlerDb

logger = logging.getLogger(__name__)

# Handler ID for ModelHandlerOutput
HANDLER_ID_LEDGER_QUERY: str = "ledger-query-handler"

# Default pagination limits
_DEFAULT_LIMIT: int = 100
_MAX_LIMIT: int = 10000

# SQL for correlation_id queries
# Uses partial index idx_event_ledger_correlation_id
_SQL_QUERY_BY_CORRELATION_ID = """
SELECT
    ledger_entry_id,
    topic,
    partition,
    kafka_offset,
    encode(event_key, 'base64') as event_key,
    encode(event_value, 'base64') as event_value,
    onex_headers,
    envelope_id,
    correlation_id,
    event_type,
    source,
    event_timestamp,
    ledger_written_at
FROM event_ledger
WHERE correlation_id = $1
ORDER BY COALESCE(event_timestamp, ledger_written_at) DESC
LIMIT $2
OFFSET $3
"""

# SQL for counting correlation_id matches (for pagination metadata)
_SQL_COUNT_BY_CORRELATION_ID = """
SELECT COUNT(*) as total
FROM event_ledger
WHERE correlation_id = $1
"""

# SQL for time range queries
# Uses index idx_event_ledger_topic_timestamp for topic-scoped queries
# Falls back to idx_event_ledger_event_timestamp for unscoped queries
_SQL_QUERY_BY_TIME_RANGE_BASE = """
SELECT
    ledger_entry_id,
    topic,
    partition,
    kafka_offset,
    encode(event_key, 'base64') as event_key,
    encode(event_value, 'base64') as event_value,
    onex_headers,
    envelope_id,
    correlation_id,
    event_type,
    source,
    event_timestamp,
    ledger_written_at
FROM event_ledger
WHERE COALESCE(event_timestamp, ledger_written_at) >= $1
  AND COALESCE(event_timestamp, ledger_written_at) < $2
"""

_SQL_COUNT_BY_TIME_RANGE_BASE = """
SELECT COUNT(*) as total
FROM event_ledger
WHERE COALESCE(event_timestamp, ledger_written_at) >= $1
  AND COALESCE(event_timestamp, ledger_written_at) < $2
"""


class HandlerLedgerQuery:
    """Handler for querying events from the audit ledger.

    This handler implements query operations for ProtocolLedgerPersistence,
    composing with HandlerDb for PostgreSQL operations. It provides:

    - Query by correlation_id (distributed tracing)
    - Query by time_range (replay, audit, debugging)
    - Optional filters by event_type and topic
    - Pagination with limit/offset
    - Consistent response surface via ModelLedgerQueryResult

    Internal Routing:
        Based on the operation field in the envelope:
        - "ledger.query" with correlation_id → _query_by_correlation_id()
        - "ledger.query" with start_time/end_time → _query_by_time_range()
        - Or use the explicit typed methods directly

    Attributes:
        handler_type: EnumHandlerType.INFRA_HANDLER
        handler_category: EnumHandlerTypeCategory.EFFECT

    Example:
        >>> handler = HandlerLedgerQuery(container, db_handler)
        >>> await handler.initialize({})
        >>> # Query by correlation_id
        >>> entries = await handler.query_by_correlation_id(corr_id, limit=50)
        >>> # Query by time range
        >>> entries = await handler.query_by_time_range(start, end, event_type="NodeRegistered")
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        db_handler: HandlerDb,
    ) -> None:
        """Initialize the ledger query handler.

        Args:
            container: ONEX dependency injection container.
            db_handler: Initialized HandlerDb instance for PostgreSQL operations.
        """
        self._container = container
        self._db_handler = db_handler
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler."""
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler.

        Args:
            config: Configuration dict (currently unused).

        Raises:
            RuntimeHostError: If HandlerDb is not initialized.
        """
        if not getattr(self._db_handler, "_initialized", False):
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="initialize",
            )
            raise RuntimeHostError(
                "HandlerDb must be initialized before HandlerLedgerQuery",
                context=ctx,
            )

        self._initialized = True
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={"handler": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._initialized = False
        logger.info("HandlerLedgerQuery shutdown complete")

    # =========================================================================
    # Public Query Methods (Typed Interface)
    # =========================================================================

    async def query_by_correlation_id(
        self,
        correlation_id: UUID,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
    ) -> list[ModelLedgerEntry]:
        """Query ledger entries by correlation ID.

        Args:
            correlation_id: The correlation ID to search for.
            limit: Maximum entries to return (default: 100, max: 10000).
            offset: Number of entries to skip for pagination.

        Returns:
            List of ModelLedgerEntry matching the correlation ID.
        """
        self._ensure_initialized("ledger.query.by_correlation_id")
        limit = self._normalize_limit(limit)

        # Execute query via HandlerDb
        rows = await self._execute_query(
            sql=_SQL_QUERY_BY_CORRELATION_ID,
            parameters=[str(correlation_id), limit, offset],
            operation="ledger.query.by_correlation_id",
            correlation_id=correlation_id,
        )

        return [self._row_to_entry(row) for row in rows]

    async def query_by_time_range(
        self,
        start: datetime,
        end: datetime,
        correlation_id: UUID | None = None,
        event_type: str | None = None,
        topic: str | None = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
    ) -> list[ModelLedgerEntry]:
        """Query ledger entries within a time range.

        Args:
            start: Start of time range (inclusive).
            end: End of time range (exclusive).
            correlation_id: Correlation ID for distributed tracing (auto-generated if None).
            event_type: Optional filter by event type.
            topic: Optional filter by Kafka topic.
            limit: Maximum entries to return (default: 100, max: 10000).
            offset: Number of entries to skip for pagination.

        Returns:
            List of ModelLedgerEntry within the time range.
        """
        self._ensure_initialized("ledger.query.by_time_range")
        limit = self._normalize_limit(limit)
        # Auto-generate correlation_id if not provided
        effective_correlation_id = (
            correlation_id if correlation_id is not None else uuid4()
        )

        # Build query model for SQL generation
        query_params = ModelLedgerQuery(
            start_time=start,
            end_time=end,
            event_type=event_type,
            topic=topic,
            limit=limit,
            offset=offset,
        )

        # Build dynamic SQL with optional filters
        sql, _count_sql, parameters = self._build_time_range_query(query_params)

        # Execute query via HandlerDb
        rows = await self._execute_query(
            sql=sql,
            parameters=parameters,
            operation="ledger.query.by_time_range",
            correlation_id=effective_correlation_id,
        )

        return [self._row_to_entry(row) for row in rows]

    async def query(
        self,
        query: ModelLedgerQuery,
        correlation_id: UUID,
    ) -> ModelLedgerQueryResult:
        """Execute a query using the ModelLedgerQuery parameters.

        Routes to the appropriate private method based on query parameters.

        Args:
            query: Query parameters model.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            ModelLedgerQueryResult with entries, total_count, and has_more.
        """
        self._ensure_initialized("ledger.query")

        # Route based on query parameters
        if query.correlation_id is not None:
            entries = await self.query_by_correlation_id(
                correlation_id=query.correlation_id,
                limit=query.limit,
                offset=query.offset,
            )
            total_count = await self._count_by_correlation_id(query.correlation_id)
        elif query.start_time is not None and query.end_time is not None:
            entries = await self.query_by_time_range(
                start=query.start_time,
                end=query.end_time,
                correlation_id=correlation_id,
                event_type=query.event_type,
                topic=query.topic,
                limit=query.limit,
                offset=query.offset,
            )
            total_count = await self._count_by_time_range(
                start=query.start_time,
                end=query.end_time,
                correlation_id=correlation_id,
                event_type=query.event_type,
                topic=query.topic,
            )
        else:
            # No specific query criteria - would return all events
            # This is likely an error or needs explicit "get all" operation
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="ledger.query",
            )
            raise RuntimeHostError(
                "Query must specify either correlation_id or time range (start_time + end_time)",
                context=ctx,
            )

        has_more = query.offset + len(entries) < total_count

        return ModelLedgerQueryResult(
            entries=entries,
            total_count=total_count,
            has_more=has_more,
            query=query,
        )

    # =========================================================================
    # Envelope-Based Interface (ProtocolHandler)
    # =========================================================================

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelLedgerQueryResult]:
        """Execute ledger query from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: "ledger.query"
                - payload: ModelLedgerQuery as dict
                - correlation_id: Optional correlation ID

        Returns:
            ModelHandlerOutput wrapping ModelLedgerQueryResult.
        """
        correlation_id_raw = envelope.get("correlation_id")
        correlation_id = (
            UUID(str(correlation_id_raw)) if correlation_id_raw else uuid4()
        )
        input_envelope_id = uuid4()

        payload_raw = envelope.get("payload")
        if not isinstance(payload_raw, dict):
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="ledger.query",
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        # Parse payload into typed model
        query = ModelLedgerQuery.model_validate(payload_raw)

        # Execute query
        result = await self.query(query, correlation_id=correlation_id)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_LEDGER_QUERY,
            result=result,
        )

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _ensure_initialized(self, operation: str) -> None:
        """Ensure handler is initialized."""
        if not self._initialized:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation=operation,
            )
            raise RuntimeHostError(
                "HandlerLedgerQuery not initialized. Call initialize() first.",
                context=ctx,
            )

    def _normalize_limit(self, limit: int) -> int:
        """Normalize limit to valid range."""
        if limit < 1:
            return _DEFAULT_LIMIT
        if limit > _MAX_LIMIT:
            return _MAX_LIMIT
        return limit

    async def _execute_query(
        self,
        sql: str,
        parameters: list[object],
        operation: str,
        correlation_id: UUID,
    ) -> list[dict[str, object]]:
        """Execute a query via HandlerDb and return rows."""
        envelope: dict[str, object] = {
            "operation": "db.query",
            "payload": {
                "sql": sql,
                "parameters": parameters,
            },
            "correlation_id": str(correlation_id),
        }

        db_result = await self._db_handler.execute(envelope)
        if db_result.result is None:
            return []
        return db_result.result.payload.rows

    async def _count_by_correlation_id(self, correlation_id: UUID) -> int:
        """Get total count for correlation_id query."""
        rows = await self._execute_query(
            sql=_SQL_COUNT_BY_CORRELATION_ID,
            parameters=[str(correlation_id)],
            operation="ledger.query.count",
            correlation_id=correlation_id,
        )
        if rows and rows[0].get("total") is not None:
            return int(str(rows[0]["total"]))
        return 0

    async def _count_by_time_range(
        self,
        start: datetime,
        end: datetime,
        correlation_id: UUID,
        event_type: str | None = None,
        topic: str | None = None,
    ) -> int:
        """Get total count for time_range query."""
        query_params = ModelLedgerQuery(
            start_time=start,
            end_time=end,
            event_type=event_type,
            topic=topic,
            limit=1,
            offset=0,
        )
        _, count_sql, parameters = self._build_time_range_query(
            query_params, count_only=True
        )

        rows = await self._execute_query(
            sql=count_sql,
            parameters=parameters,
            operation="ledger.query.count",
            correlation_id=correlation_id,
        )
        if rows and rows[0].get("total") is not None:
            return int(str(rows[0]["total"]))
        return 0

    def _build_time_range_query(
        self,
        query: ModelLedgerQuery,
        count_only: bool = False,
    ) -> tuple[str, str, list[object]]:
        """Build dynamic SQL for time range query with optional filters.

        Args:
            query: Query parameters including start_time, end_time, filters, pagination.
            count_only: If True, don't add limit/offset to parameters.

        Returns:
            Tuple of (query_sql, count_sql, parameters).
        """
        # Start with base parameters (start_time and end_time are required for this path)
        parameters: list[object] = [query.start_time, query.end_time]
        param_index = 3  # $1 and $2 are start/end

        # Build WHERE clause additions
        where_additions: list[str] = []

        if query.event_type is not None:
            where_additions.append(f"AND event_type = ${param_index}")
            parameters.append(query.event_type)
            param_index += 1

        if query.topic is not None:
            where_additions.append(f"AND topic = ${param_index}")
            parameters.append(query.topic)
            param_index += 1

        # Build final SQL
        where_clause = " ".join(where_additions)

        # Query SQL with ordering and pagination
        query_sql = (
            _SQL_QUERY_BY_TIME_RANGE_BASE
            + where_clause
            + f"""
ORDER BY COALESCE(event_timestamp, ledger_written_at) DESC
LIMIT ${param_index}
OFFSET ${param_index + 1}
"""
        )

        # Count SQL without ordering/pagination
        count_sql = _SQL_COUNT_BY_TIME_RANGE_BASE + where_clause

        if not count_only:
            parameters.extend([query.limit, query.offset])

        return query_sql, count_sql, parameters

    def _row_to_entry(self, row: dict[str, object]) -> ModelLedgerEntry:
        """Convert a database row to ModelLedgerEntry.

        The row comes from HandlerDb which returns dict[str, object].
        event_key and event_value are already base64-encoded via SQL encode().

        Raises:
            RuntimeHostError: If ledger_written_at is not a datetime (data corruption).
        """
        # Extract ledger_written_at which is guaranteed to exist
        ledger_written_at_raw = row["ledger_written_at"]
        if not isinstance(ledger_written_at_raw, datetime):
            # This should never happen for valid ledger entries - indicates data corruption
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="ledger.query.row_to_entry",
            )
            raise RuntimeHostError(
                f"Data integrity error: ledger_written_at must be datetime, got {type(ledger_written_at_raw).__name__}",
                context=ctx,
            )

        return ModelLedgerEntry(
            ledger_entry_id=UUID(str(row["ledger_entry_id"])),
            topic=str(row["topic"]),
            partition=int(str(row["partition"])),
            kafka_offset=int(str(row["kafka_offset"])),
            event_key=str(row["event_key"]) if row["event_key"] else None,
            event_value=str(row["event_value"]),
            onex_headers=row["onex_headers"]
            if isinstance(row["onex_headers"], dict)
            else {},
            envelope_id=UUID(str(row["envelope_id"])) if row["envelope_id"] else None,
            correlation_id=UUID(str(row["correlation_id"]))
            if row["correlation_id"]
            else None,
            event_type=str(row["event_type"]) if row["event_type"] else None,
            source=str(row["source"]) if row["source"] else None,
            event_timestamp=row["event_timestamp"]
            if isinstance(row["event_timestamp"], datetime)
            else None,
            ledger_written_at=ledger_written_at_raw,
        )


__all__ = ["HandlerLedgerQuery"]
