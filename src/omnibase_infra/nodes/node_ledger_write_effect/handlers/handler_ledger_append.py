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
"""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumResponseStatus,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.nodes.node_ledger_write_effect.models import ModelLedgerAppendResult

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.handlers.handler_db import HandlerDb
    from omnibase_infra.nodes.reducers.models import ModelPayloadLedgerAppend

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

    This handler implements the append operation for ProtocolLedgerPersistence,
    composing with HandlerDb for PostgreSQL operations. It provides:

    - Base64 decoding of event payloads to bytes
    - Idempotent INSERT via ON CONFLICT DO NOTHING
    - Duplicate detection via RETURNING clause
    - Type-safe input/output with Pydantic models

    Attributes:
        handler_type: EnumHandlerType.INFRA_HANDLER
        handler_category: EnumHandlerTypeCategory.EFFECT

    Example:
        >>> handler = HandlerLedgerAppend(container, db_handler)
        >>> await handler.initialize({})
        >>> result = await handler.append(payload)
        >>> if result.duplicate:
        ...     logger.info("Event already in ledger")
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        db_handler: HandlerDb,
    ) -> None:
        """Initialize the ledger append handler.

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

        The underlying HandlerDb must already be initialized before
        calling this method.

        Args:
            config: Configuration dict (currently unused).

        Raises:
            RuntimeHostError: If HandlerDb is not initialized.
        """
        # Verify db_handler is initialized
        if not getattr(self._db_handler, "_initialized", False):
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="initialize",
            )
            raise RuntimeHostError(
                "HandlerDb must be initialized before HandlerLedgerAppend",
                context=ctx,
            )

        self._initialized = True
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={"handler": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        """Shutdown the handler.

        Does not shutdown the underlying HandlerDb - that is managed separately.
        """
        self._initialized = False
        logger.info("HandlerLedgerAppend shutdown complete")

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
            RuntimeHostError: If handler not initialized or validation fails.
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
        """
        correlation_id = payload.correlation_id or uuid4()

        if not self._initialized:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="ledger.append",
            )
            raise RuntimeHostError(
                "HandlerLedgerAppend not initialized. Call initialize() first.",
                context=ctx,
            )

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
            return base64.b64decode(encoded)
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="ledger.append",
            )
            raise RuntimeHostError(
                f"Failed to decode base64 event data: {type(e).__name__}",
                context=ctx,
            ) from e

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelLedgerAppendResult]:
        """Execute ledger append from envelope (ProtocolHandler interface).

        This method provides the standard handler interface for contract-driven
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
        from omnibase_infra.nodes.reducers.models import ModelPayloadLedgerAppend

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
