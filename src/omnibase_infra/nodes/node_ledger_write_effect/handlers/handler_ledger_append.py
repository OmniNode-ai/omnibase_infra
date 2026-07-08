# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for ledger append operations with idempotent write support.

This handler composes with HandlerDb for PostgreSQL operations, providing
a typed interface for appending events to the audit ledger with duplicate
detection via ON CONFLICT DO NOTHING.

Bytes Encoding:
    The ModelPayloadLedgerAppend contains base64-encoded event_key and event_value
    since bytes cannot safely cross intent boundaries. This handler decodes them
    to bytes before passing to PostgreSQL, which stores them as BYTEA.

Idempotency:
    Uses INSERT ... ON CONFLICT (topic, partition, kafka_offset) DO NOTHING RETURNING.
    If RETURNING returns no rows, the event was already in the ledger (duplicate).
    Duplicates are not errors - they enable idempotent replay.

Design Decision - Composition with HandlerDb:
    This handler delegates SQL execution to HandlerDb rather than using asyncpg
    directly. This provides:
    - Circuit breaker protection
    - Error classification (transient vs permanent)
    - Connection pool management
    - Consistent error handling

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
import base64
import binascii
import json
import logging
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
from omnibase_infra.nodes.node_ledger_write_effect.models import ModelLedgerAppendResult

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.nodes.node_registration_reducer.models import (
        ModelPayloadLedgerAppend,
    )

logger = logging.getLogger(__name__)

# Handler ID for ModelHandlerOutput
HANDLER_ID_LEDGER_APPEND: str = "ledger-append-handler"

# SQL for idempotent append with duplicate detection
# Uses RETURNING to detect whether insert succeeded (returns row) or
# ON CONFLICT was triggered (returns nothing)
_SQL_APPEND = """
INSERT INTO event_ledger (
    topic,
    partition,
    kafka_offset,
    event_key,
    event_value,
    onex_headers,
    envelope_id,
    correlation_id,
    event_type,
    source,
    event_timestamp
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
ON CONFLICT (topic, partition, kafka_offset) DO NOTHING
RETURNING ledger_entry_id
"""


class HandlerLedgerAppend:
    """Handler for appending events to the audit ledger with idempotent writes.

    The append operation for ProtocolLedgerPersistence,
    composing with HandlerDb for PostgreSQL operations. It provides:

    - Base64 decoding of event payloads to bytes
    - Idempotent INSERT via ON CONFLICT DO NOTHING
    - Duplicate detection via RETURNING clause
    - Type-safe input/output with Pydantic models

    Attributes:
        handler_type: EnumHandlerType.INFRA_HANDLER
        handler_category: EnumHandlerTypeCategory.EFFECT

    Example:
        >>> handler = HandlerLedgerAppend(container)
        >>> await handler.initialize({})
        >>> result = await handler.append(payload)
        >>> if result.duplicate:
        ...     logger.info("Event already in ledger")
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        db_dsn: str | None = None,
    ) -> None:
        """Initialize the ledger append handler.

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
        append() connects lazily via `_ensure_db_ready()` regardless.

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
        logger.info("HandlerLedgerAppend shutdown complete")

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
                    operation="ledger.append.connect",
                )
                raise RuntimeHostError(
                    "Missing PostgreSQL DSN for ledger persistence -- provide "
                    "db_dsn at construction or initialize({'dsn': ...})",
                    context=ctx,
                )
            await self._db_handler.initialize({"dsn": dsn})
            self._initialized = True

    async def append(
        self,
        payload: ModelPayloadLedgerAppend,
    ) -> ModelLedgerAppendResult:
        """Append an event to the audit ledger.

        Decodes base64 event data, executes idempotent INSERT, and detects
        duplicates via the RETURNING clause.

        Args:
            payload: Event payload containing Kafka position and event data.

        Returns:
            ModelLedgerAppendResult with success, ledger_entry_id, and duplicate flag.

        Raises:
            RuntimeHostError: If no PostgreSQL DSN is configured, or validation fails.
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
        """
        if payload.correlation_id is None:
            logger.warning(
                "Event appended to ledger without correlation_id — emitting source "
                "is violating the envelope contract; trace chain will be broken for "
                "topic=%s partition=%d offset=%d",
                payload.topic,
                payload.partition,
                payload.kafka_offset,
            )
        correlation_id = payload.correlation_id or uuid4()

        # Lazily connect the composed HandlerDb on first real use -- the
        # auto-wiring resolver never calls initialize() (OMN-14140).
        await self._ensure_db_ready()

        # Decode base64 event data to bytes
        event_key_bytes = (
            self._decode_base64(payload.event_key) if payload.event_key else None
        )
        event_value_bytes = self._decode_base64(payload.event_value)

        # Serialize onex_headers to JSON string for JSONB column
        onex_headers_json = json.dumps(payload.onex_headers)

        # Build parameters for INSERT
        # Order must match $1..$11 in _SQL_APPEND
        parameters: list[object] = [
            payload.topic,  # $1
            payload.partition,  # $2
            payload.kafka_offset,  # $3
            event_key_bytes,  # $4 (BYTEA, nullable)
            event_value_bytes,  # $5 (BYTEA)
            onex_headers_json,  # $6 (JSONB)
            str(payload.envelope_id)
            if payload.envelope_id
            else None,  # $7 (UUID, nullable)
            str(payload.correlation_id)
            if payload.correlation_id
            else None,  # $8 (UUID, nullable)
            payload.event_type,  # $9 (TEXT, nullable)
            payload.source,  # $10 (TEXT, nullable)
            payload.event_timestamp,  # $11 (TIMESTAMPTZ, nullable)
        ]

        # Build envelope for HandlerDb
        envelope: dict[str, object] = {
            "operation": "db.query",  # Use query because RETURNING produces rows
            "payload": {
                "sql": _SQL_APPEND,
                "parameters": parameters,
            },
            "correlation_id": str(correlation_id),
        }

        logger.debug(
            "Appending event to ledger",
            extra={
                "topic": payload.topic,
                "partition": payload.partition,
                "offset": payload.kafka_offset,
                "correlation_id": str(correlation_id),
            },
        )

        # Execute via HandlerDb
        db_result = await self._db_handler.execute(envelope)

        # Check if RETURNING produced a row (insert succeeded) or not (duplicate)
        # db_result.result is guaranteed non-None for successful db operations
        if db_result.result is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="ledger.append",
            )
            raise RuntimeHostError("Database operation returned no result", context=ctx)

        rows = db_result.result.payload.rows
        if rows and len(rows) > 0:
            # Insert succeeded - extract ledger_entry_id from RETURNING
            ledger_entry_id = UUID(str(rows[0]["ledger_entry_id"]))
            duplicate = False
            logger.debug(
                "Event appended to ledger",
                extra={
                    "ledger_entry_id": str(ledger_entry_id),
                    "topic": payload.topic,
                    "partition": payload.partition,
                    "offset": payload.kafka_offset,
                },
            )
        else:
            # ON CONFLICT DO NOTHING triggered - duplicate
            ledger_entry_id = None
            duplicate = True
            logger.debug(
                "Duplicate event detected (already in ledger)",
                extra={
                    "topic": payload.topic,
                    "partition": payload.partition,
                    "offset": payload.kafka_offset,
                },
            )

        return ModelLedgerAppendResult(
            success=True,
            ledger_entry_id=ledger_entry_id,
            duplicate=duplicate,
            topic=payload.topic,
            partition=payload.partition,
            kafka_offset=payload.kafka_offset,
        )

    def _decode_base64(self, encoded: str) -> bytes:
        """Decode base64 string to bytes.

        Args:
            encoded: Base64-encoded string.

        Returns:
            Decoded bytes.

        Raises:
            RuntimeHostError: If decoding fails.
        """
        try:
            return base64.b64decode(encoded, validate=True)
        except binascii.Error as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="ledger.append",
            )
            raise RuntimeHostError(
                f"Failed to decode base64 event data: {type(e).__name__}",
                context=ctx,
            ) from e

    async def handle(
        self,
        envelope: object,
    ) -> ModelHandlerOutput[ModelLedgerAppendResult]:
        """Contract-typed auto-wiring entry point.

        ``node_ledger_write_effect``'s contract declares ``operation_match``
        routing with no ``event_model``, so the dispatch engine's auto-wiring
        (``handler_wiring._make_dispatch_callback``) invokes ``handle(envelope)``
        directly instead of ``execute()``. Without this method the callback binds
        ``_missing_handle``, which raises on every dispatched ledger-append
        command (OMN-14134 — WI-14 keystone).

        The value actually delivered here is whatever
        ``MessageDispatchEngine._materialize_envelope_with_bindings`` produces
        for the live dispatch path -- a **dict** (``{"payload": ..., ...}``),
        not an attribute-bearing envelope object. ``_extract_envelope_field``
        handles both shapes so this method works for the real runtime dispatch
        path as well as for object-shaped test envelopes.

        Extracts the append payload from the auto-wired envelope, delegates to
        append(), and wraps the result identically to execute().
        """
        from omnibase_infra.nodes.node_registration_reducer.models import (
            ModelPayloadLedgerAppend,
        )

        payload_raw = self._extract_envelope_field(envelope, "payload")
        if payload_raw is None:
            payload_raw = envelope
        payload = (
            payload_raw
            if isinstance(payload_raw, ModelPayloadLedgerAppend)
            else ModelPayloadLedgerAppend.model_validate(payload_raw)
        )

        envelope_correlation_id = self._extract_envelope_field(
            envelope, "correlation_id"
        )
        correlation_id = self._safe_correlation_id(
            envelope_correlation_id or payload.correlation_id
        )
        input_envelope_id = uuid4()

        result = await self.append(payload)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_LEDGER_APPEND,
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

        Returns a fresh UUID if `raw` is missing or unparseable. Unlike
        ``execute()``, ``handle()`` has no envelope validation step to reject a
        malformed correlation_id before reaching this point — the audit ledger
        must never drop an event over a bad correlation_id, so this degrades to
        a fresh UUID rather than raising.
        """
        if not raw:
            return uuid4()
        if isinstance(raw, UUID):
            return raw
        try:
            return UUID(str(raw))
        except (ValueError, TypeError):
            return uuid4()

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelLedgerAppendResult]:
        """Execute ledger append from envelope (ProtocolHandler interface).

        The standard handler interface for contract-driven
        invocation. It extracts the payload from the envelope and delegates to
        the append() method.

        Args:
            envelope: Request envelope containing:
                - operation: "ledger.append"
                - payload: ModelPayloadLedgerAppend as dict
                - correlation_id: Optional correlation ID

        Returns:
            ModelHandlerOutput wrapping ModelLedgerAppendResult.
        """
        from omnibase_infra.nodes.node_registration_reducer.models import (
            ModelPayloadLedgerAppend,
        )

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
                operation="ledger.append",
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        # Parse payload into typed model
        payload = ModelPayloadLedgerAppend.model_validate(payload_raw)

        # Execute append
        result = await self.append(payload)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_LEDGER_APPEND,
            result=result,
        )


__all__ = ["HandlerLedgerAppend"]
