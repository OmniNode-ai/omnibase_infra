# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for ledger query operations with internal routing.

Query operations for the event ledger, supporting
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

Design Decision - Internally-Composed HandlerDb (OMN-14140):
    HandlerDb is composed INTERNALLY from `container` rather than accepted as a
    constructor argument. The contract-driven auto-wiring resolver
    (runtime/auto_wiring/handler_wiring.py) can only construct handlers whose
    required constructor parameters are drawn from a small known set
    (container, event_bus, ownership_query, ...) -- it has no way to resolve an
    arbitrary `db_handler: HandlerDb` parameter, so a two-arg constructor left
    this handler permanently unconstructable (quarantined, routed to the no-op
    skip dispatcher). Composing HandlerDb from `container` matches HandlerDb's
    own single-argument constructor and is the same shape already used by
    other auto-wired handlers (e.g. HandlerLedgerProjection, HandlerCheckpointWrite).

    The auto-wiring resolver never calls `initialize()` on constructed
    handlers, so the composed HandlerDb connects lazily on first real use via
    `_ensure_db_ready()` rather than requiring an external initialize() call.
"""

from __future__ import annotations

import asyncio
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
from omnibase_infra.handlers.handler_db import HandlerDb
from omnibase_infra.nodes.node_ledger_write_effect.models import (
    ModelLedgerEntry,
    ModelLedgerQuery,
    ModelLedgerQueryResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

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

    Query operations for ProtocolLedgerPersistence,
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
        >>> handler = HandlerLedgerQuery(container)
        >>> await handler.initialize({})
        >>> # Query by correlation_id
        >>> entries = await handler.query_by_correlation_id(corr_id, limit=50)
        >>> # Query by time range
        >>> entries = await handler.query_by_time_range(start, end, event_type="NodeRegistered")
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        db_dsn: str | None = None,
    ) -> None:
        """Initialize the ledger query handler.

        Args:
            container: ONEX dependency injection container. HandlerDb is
                composed internally from this container (OMN-14140) so the
                auto-wiring resolver can construct this handler with a
                single, always-resolvable constructor argument.
            db_dsn: Optional PostgreSQL DSN supplied by the runtime auto-wiring
                boundary. Handlers do not read environment directly; runtime
                composition owns that IO boundary.
        """
        self._container = container
        self._db_handler = HandlerDb(container)
        self._db_dsn = db_dsn.strip() if db_dsn else ""
        self._initialized: bool = False
        self._db_init_lock = asyncio.Lock()

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler."""
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler by connecting its composed HandlerDb.

        The contract-driven auto-wiring resolver never calls initialize() on
        constructed handlers, so this is an optional eager-connect path for
        callers that do invoke it explicitly (tests, hand-wired call sites).
        Public query methods connect lazily via `_ensure_db_ready()` regardless.

        Args:
            config: Optional configuration dict. A non-empty ``dsn`` value
                updates the runtime-supplied DSN before connecting.

        Raises:
            RuntimeHostError: If no PostgreSQL DSN is configured.
        """
        config_dsn = config.get("dsn")
        if isinstance(config_dsn, str) and config_dsn.strip():
            self._db_dsn = config_dsn.strip()
        await self._ensure_db_ready()
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={"handler": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        """Shutdown the handler and its internally-composed HandlerDb."""
        if self._initialized:
            await self._db_handler.shutdown()
        self._initialized = False
        logger.info("HandlerLedgerQuery shutdown complete")

    async def _ensure_db_ready(self) -> None:
        """Lazily connect the composed HandlerDb on first real use.

        The auto-wiring resolver constructs contract-routed handlers from
        `container` alone and never calls their `initialize()` method
        (OMN-14140), so this handler owns its HandlerDb connection lifecycle
        instead of relying on an external initialize() call. Guarded by a
        lock so concurrent first-dispatches connect exactly once.

        Raises:
            RuntimeHostError: If no PostgreSQL DSN was supplied by runtime
                composition or initialize({"dsn": ...}).
        """
        if self._initialized:
            return
        async with self._db_init_lock:
            if self._initialized:
                return
            dsn = self._db_dsn
            if not dsn:
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="ledger.query.connect",
                )
                raise RuntimeHostError(
                    "Missing PostgreSQL DSN for ledger persistence -- provide "
                    "db_dsn at construction or initialize({'dsn': ...})",
                    context=ctx,
                )
            await self._db_handler.initialize({"dsn": dsn})
            self._initialized = True

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
        await self._ensure_db_ready()
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
            correlation_id: Correlation ID for distributed tracing only (auto-generated if
                None). This value is NOT used as a query filter - use query_by_correlation_id()
                for that purpose.
            event_type: Optional filter by event type.
            topic: Optional filter by Kafka topic.
            limit: Maximum entries to return (default: 100, max: 10000).
            offset: Number of entries to skip for pagination.

        Returns:
            List of ModelLedgerEntry within the time range.
        """
        await self._ensure_db_ready()
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
        await self._ensure_db_ready()

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

    async def handle(
        self,
        envelope: object,
    ) -> ModelHandlerOutput[ModelLedgerQueryResult]:
        """Contract-typed auto-wiring entry point.

        ``node_ledger_write_effect``'s contract declares ``operation_match``
        routing with no ``event_model``, so the dispatch engine's auto-wiring
        (``handler_wiring._make_dispatch_callback``) invokes ``handle(envelope)``
        directly instead of ``execute()``. Without this method the callback binds
        ``_missing_handle``, which raises on every dispatched ledger-query
        command -- the same gap ``HandlerLedgerAppend.handle()`` closed for the
        append side (OMN-14134).

        The value actually delivered here is whatever
        ``MessageDispatchEngine._materialize_envelope_with_bindings`` produces
        for the live dispatch path -- a **dict** (``{"payload": ..., ...}``),
        not an attribute-bearing envelope object. ``_extract_envelope_field``
        handles both shapes so this method works for the real runtime dispatch
        path as well as for object-shaped test envelopes.

        Extracts the query payload from the auto-wired envelope, delegates to
        query(), and wraps the result identically to execute().
        """
        payload_raw = self._extract_envelope_field(envelope, "payload")
        if payload_raw is None:
            payload_raw = envelope
        query = (
            payload_raw
            if isinstance(payload_raw, ModelLedgerQuery)
            else ModelLedgerQuery.model_validate(payload_raw)
        )

        envelope_correlation_id = self._extract_envelope_field(
            envelope, "correlation_id"
        )
        correlation_id = self._safe_correlation_id(
            envelope_correlation_id or query.correlation_id
        )
        input_envelope_id = uuid4()

        result = await self.query(query, correlation_id=correlation_id)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_LEDGER_QUERY,
            result=result,
        )

    @staticmethod
    def _extract_envelope_field(envelope: object, key: str) -> object:
        """Return `key` from a dict-shaped or attribute-shaped envelope.

        The dispatch engine may deliver either a materialized dict (the real
        runtime dispatch path, via
        ``MessageDispatchEngine._materialize_envelope_with_bindings``) or a
        ``ModelEventEnvelope``-like object (the auto_wiring event-bus callback
        / test doubles) -- ``handle()`` must accept both shapes. Mirrors
        ``HandlerBuildLoopProjection._coerce_event_message``'s dict-vs-attribute
        handling.
        """
        if isinstance(envelope, dict):
            return envelope.get(key)
        return getattr(envelope, key, None)

    @staticmethod
    def _safe_correlation_id(raw: object) -> UUID:
        """Parse a correlation ID from envelope/payload-supplied raw input.

        Returns a fresh UUID if `raw` is missing or unparseable. Mirrors
        ``HandlerLedgerAppend._safe_correlation_id`` -- ``handle()`` has no
        envelope validation step to reject a malformed correlation_id before
        reaching this point, so this degrades to a fresh UUID rather than
        raising.
        """
        if not raw:
            return uuid4()
        if isinstance(raw, UUID):
            return raw
        try:
            return UUID(str(raw))
        except (ValueError, TypeError):
            return uuid4()

    # =========================================================================
    # Private Helpers
    # =========================================================================

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
