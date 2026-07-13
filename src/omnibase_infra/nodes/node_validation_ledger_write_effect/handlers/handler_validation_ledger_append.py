# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for validation ledger append operations with idempotent write support.

Mirrors ``HandlerLedgerAppend`` (node_ledger_write_effect) for the
validation_event_ledger table (OMN-14524). This handler composes with
HandlerDb for PostgreSQL operations, providing a typed interface for
appending cross-repo validation events to the ledger with duplicate
detection via ON CONFLICT DO NOTHING.

Bytes Encoding:
    ModelPayloadValidationLedgerAppend carries a base64-encoded
    ``envelope_bytes`` field since raw bytes cannot safely cross intent
    boundaries. This handler decodes it to bytes before passing to
    PostgreSQL, which stores it as BYTEA.

Idempotency:
    Uses INSERT ... ON CONFLICT (kafka_topic, kafka_partition, kafka_offset)
    DO NOTHING RETURNING. If RETURNING returns no rows, the event was
    already in the ledger (duplicate). Duplicates are not errors -- they
    enable idempotent replay.

Design Decision - Composition with HandlerDb:
    This handler delegates SQL execution to HandlerDb rather than using
    asyncpg directly, matching HandlerLedgerAppend's design: circuit breaker
    protection, error classification, connection pool management, and
    consistent error handling.

Design Decision - Internally-Composed HandlerDb (mirrors OMN-14140):
    HandlerDb is composed INTERNALLY from `container` rather than accepted
    as a constructor argument. The contract-driven auto-wiring resolver AND
    the OMN-14516 intent-routing derivation both construct effect handlers
    as ``handler_cls(container, dsn)`` -- a two-arg constructor accepting an
    arbitrary ``db_handler: HandlerDb`` parameter cannot be resolved by
    either path. This is precisely why ``PostgresValidationLedgerRepository``
    (which requires a pre-built ``asyncpg.Pool``) could never be wired: it
    has no ``(container, dsn)`` constructor shape.

    The auto-wiring resolver never calls `initialize()` on constructed
    handlers, so the composed HandlerDb connects lazily on first real use via
    `_ensure_db_ready()` rather than requiring an external initialize() call.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
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
from omnibase_infra.models.validation_ledger import ModelValidationLedgerAppendResult

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.models.validation_ledger import (
        ModelPayloadValidationLedgerAppend,
    )

logger = logging.getLogger(__name__)

# Handler ID for ModelHandlerOutput
HANDLER_ID_VALIDATION_LEDGER_APPEND: str = "validation-ledger-append-handler"

# SQL for idempotent append with duplicate detection. Mirrors
# PostgresValidationLedgerRepository._SQL_APPEND -- same table, same columns,
# same ON CONFLICT target -- this handler is a constructable alternate write
# surface for the same validation_event_ledger table, not a schema change.
_SQL_APPEND = """
INSERT INTO validation_event_ledger (
    run_id,
    repo_id,
    event_type,
    event_version,
    occurred_at,
    kafka_topic,
    kafka_partition,
    kafka_offset,
    envelope_bytes,
    envelope_hash
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (kafka_topic, kafka_partition, kafka_offset) DO NOTHING
RETURNING id
"""


class HandlerValidationLedgerAppend:
    """Handler for appending validation events to the validation ledger.

    The append operation for validation_event_ledger persistence, composing
    with HandlerDb for PostgreSQL operations. Mirrors HandlerLedgerAppend's
    shape exactly so the kernel's generic intent-routing derivation
    (OMN-14516) can construct it as ``HandlerValidationLedgerAppend(container,
    dsn)`` with no by-name allowlist entry.

    Attributes:
        handler_type: EnumHandlerType.INFRA_HANDLER
        handler_category: EnumHandlerTypeCategory.EFFECT

    Example:
        >>> handler = HandlerValidationLedgerAppend(container)
        >>> await handler.initialize({})
        >>> result = await handler.append(payload)
        >>> if result.duplicate:
        ...     logger.info("Validation event already in ledger")
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        db_dsn: str | None = None,
    ) -> None:
        """Initialize the validation ledger append handler.

        Args:
            container: ONEX dependency injection container. HandlerDb is
                composed internally from this container so the auto-wiring
                resolver and intent-routing derivation can construct this
                handler with a single, always-resolvable constructor shape.
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
        logger.info("HandlerValidationLedgerAppend shutdown complete")

    async def _ensure_db_ready(self) -> None:
        """Lazily connect the composed HandlerDb on first real use.

        The auto-wiring resolver constructs contract-routed handlers from
        `container` alone and never calls their `initialize()` method, so
        this handler owns its HandlerDb connection lifecycle instead of
        relying on an external initialize() call. Guarded by a lock so
        concurrent first-dispatches connect exactly once.

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
                    operation="validation_ledger.append.connect",
                )
                raise RuntimeHostError(
                    "Missing PostgreSQL DSN for validation ledger persistence -- "
                    "provide db_dsn at construction or initialize({'dsn': ...})",
                    context=ctx,
                )
            await self._db_handler.initialize({"dsn": dsn})
            self._initialized = True

    async def append(
        self,
        payload: ModelPayloadValidationLedgerAppend,
    ) -> ModelValidationLedgerAppendResult:
        """Append a validation event to the validation ledger.

        Decodes base64 envelope bytes, executes idempotent INSERT, and
        detects duplicates via the RETURNING clause.

        Args:
            payload: Validation event payload containing Kafka position and
                envelope data.

        Returns:
            ModelValidationLedgerAppendResult with success, ledger_entry_id,
            and duplicate flag.

        Raises:
            RuntimeHostError: If no PostgreSQL DSN is configured, or
                validation fails.
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
        """
        # Lazily connect the composed HandlerDb on first real use.
        await self._ensure_db_ready()

        envelope_bytes = self._decode_base64(payload.envelope_bytes)

        # Build parameters for INSERT. Order must match $1..$10 in _SQL_APPEND.
        parameters: list[object] = [
            payload.run_id,  # $1
            payload.repo_id,  # $2
            payload.event_type,  # $3
            payload.event_version,  # $4
            payload.occurred_at,  # $5
            payload.kafka_topic,  # $6
            payload.kafka_partition,  # $7
            payload.kafka_offset,  # $8
            envelope_bytes,  # $9 (BYTEA)
            payload.envelope_hash,  # $10
        ]

        envelope: dict[str, object] = {
            "operation": "db.query",  # Use query because RETURNING produces rows
            "payload": {
                "sql": _SQL_APPEND,
                "parameters": parameters,
            },
            "correlation_id": str(payload.correlation_id or uuid4()),
        }

        logger.debug(
            "Appending validation event to ledger",
            extra={
                "kafka_topic": payload.kafka_topic,
                "kafka_partition": payload.kafka_partition,
                "kafka_offset": payload.kafka_offset,
                "run_id": str(payload.run_id),
            },
        )

        # Execute via HandlerDb
        db_result = await self._db_handler.execute(envelope)

        if db_result.result is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=payload.correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="validation_ledger.append",
            )
            raise RuntimeHostError("Database operation returned no result", context=ctx)

        rows = db_result.result.payload.rows
        if rows and len(rows) > 0:
            ledger_entry_id = UUID(str(rows[0]["id"]))
            duplicate = False
            logger.debug(
                "Validation event appended to ledger",
                extra={
                    "ledger_entry_id": str(ledger_entry_id),
                    "kafka_topic": payload.kafka_topic,
                    "kafka_partition": payload.kafka_partition,
                    "kafka_offset": payload.kafka_offset,
                },
            )
        else:
            ledger_entry_id = None
            duplicate = True
            logger.debug(
                "Duplicate validation event detected (already in ledger)",
                extra={
                    "kafka_topic": payload.kafka_topic,
                    "kafka_partition": payload.kafka_partition,
                    "kafka_offset": payload.kafka_offset,
                },
            )

        return ModelValidationLedgerAppendResult(
            success=True,
            ledger_entry_id=ledger_entry_id,
            duplicate=duplicate,
            kafka_topic=payload.kafka_topic,
            kafka_partition=payload.kafka_partition,
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
                operation="validation_ledger.append",
            )
            raise RuntimeHostError(
                f"Failed to decode base64 envelope bytes: {type(e).__name__}",
                context=ctx,
            ) from e

    async def handle(
        self,
        envelope: object,
    ) -> ModelHandlerOutput[ModelValidationLedgerAppendResult]:
        """Contract-typed auto-wiring entry point.

        ``node_validation_ledger_write_effect``'s contract declares
        ``operation_match`` routing with no ``event_model``, so the dispatch
        engine's auto-wiring (``handler_wiring._make_dispatch_callback``)
        invokes ``handle(envelope)`` directly instead of ``execute()``.
        Without this method the callback binds ``_missing_handle``, which
        raises on every dispatched validation-ledger-append command (mirrors
        OMN-14134).

        The value actually delivered here is whatever
        ``MessageDispatchEngine._materialize_envelope_with_bindings`` produces
        for the live dispatch path -- a **dict**
        (``{"payload": ..., "correlation_id": ...}``), the exact shape
        ``IntentEffectDispatchBridge.execute()`` constructs -- not an
        attribute-bearing envelope object. ``_extract_envelope_field`` handles
        both shapes so this method works for the real intent-routed dispatch
        path as well as for object-shaped test envelopes.

        Extracts the append payload from the auto-wired envelope, delegates
        to append(), and wraps the result identically to execute().
        """
        from omnibase_infra.models.validation_ledger import (
            ModelPayloadValidationLedgerAppend,
        )

        payload_raw = self._extract_envelope_field(envelope, "payload")
        if payload_raw is None:
            payload_raw = envelope
        payload = (
            payload_raw
            if isinstance(payload_raw, ModelPayloadValidationLedgerAppend)
            else ModelPayloadValidationLedgerAppend.model_validate(payload_raw)
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
            handler_id=HANDLER_ID_VALIDATION_LEDGER_APPEND,
            result=result,
        )

    @staticmethod
    def _extract_envelope_field(envelope: object, key: str) -> object:
        """Return `key` from a dict-shaped or attribute-shaped envelope.

        The dispatch engine may deliver either a materialized dict (the real
        runtime dispatch path, via
        ``MessageDispatchEngine._materialize_envelope_with_bindings``, and the
        shape ``IntentEffectDispatchBridge.execute()`` constructs) or a
        ``ModelEventEnvelope``-like object (test doubles) -- ``handle()`` must
        accept both shapes.
        """
        if isinstance(envelope, dict):
            return envelope.get(key)
        return getattr(envelope, key, None)

    @staticmethod
    def _safe_correlation_id(raw: object) -> UUID:
        """Parse a correlation ID from envelope/payload-supplied raw input.

        Returns a fresh UUID if `raw` is missing or unparseable. Unlike
        ``execute()``, ``handle()`` has no envelope validation step to reject
        a malformed correlation_id before reaching this point -- the
        validation ledger must never drop an event over a bad correlation_id,
        so this degrades to a fresh UUID rather than raising.
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
    ) -> ModelHandlerOutput[ModelValidationLedgerAppendResult]:
        """Execute validation ledger append from envelope (ProtocolHandler interface).

        The standard handler interface for contract-driven invocation. It
        extracts the payload from the envelope and delegates to the append()
        method.

        Args:
            envelope: Request envelope containing:
                - operation: "validation_ledger.append"
                - payload: ModelPayloadValidationLedgerAppend as dict
                - correlation_id: Optional correlation ID

        Returns:
            ModelHandlerOutput wrapping ModelValidationLedgerAppendResult.
        """
        from omnibase_infra.models.validation_ledger import (
            ModelPayloadValidationLedgerAppend,
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
                operation="validation_ledger.append",
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        payload = ModelPayloadValidationLedgerAppend.model_validate(payload_raw)

        result = await self.append(payload)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VALIDATION_LEDGER_APPEND,
            result=result,
        )


__all__ = ["HandlerValidationLedgerAppend"]
