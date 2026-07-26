# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerValidationLedgerAppend (OMN-14524).

Mirrors test_handler_ledger_append.py's coverage for the sibling
event_ledger write-effect handler. This handler is the previously-missing
constructable write surface for validation_event_ledger --
PostgresValidationLedgerRepository (asyncpg.Pool constructor) could never be
built by the contract-driven auto-wiring resolver or the OMN-14516
intent-routing derivation, both of which construct effect handlers as
``handler_cls(container, dsn)``.

Tests validate:
- Constructor composes HandlerDb from `container` alone (mirrors OMN-14140)
- Lazy DB connection guard (missing DSN raises RuntimeHostError)
- Duplicate detection via RETURNING clause (rows vs empty rows)
- Base64 decode errors raise RuntimeHostError with proper context
- handle() auto-wiring adapter accepts both dict- and object-shaped envelopes
- Protocol/property compliance (handler_type, handler_category)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumResponseStatus,
)
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.handler_db import HandlerDb
from omnibase_infra.handlers.models import ModelDbQueryPayload, ModelDbQueryResponse
from omnibase_infra.models.validation_ledger import (
    ModelPayloadValidationLedgerAppend,
)
from omnibase_infra.nodes.node_validation_ledger_write_effect.handlers.handler_validation_ledger_append import (
    HandlerValidationLedgerAppend,
)

# =============================================================================
# Fixtures
# =============================================================================


def make_mock_container() -> MagicMock:
    """Create a minimal mock ModelONEXContainer."""
    return MagicMock(spec=ModelONEXContainer)


def make_handler_with_mock_db(
    initialized: bool = True,
) -> tuple[HandlerValidationLedgerAppend, AsyncMock]:
    """Create a HandlerValidationLedgerAppend with its HandlerDb replaced by a mock.

    HandlerValidationLedgerAppend composes HandlerDb internally from
    `container` -- construction never takes a db_handler argument. Tests that
    need to control DB behavior replace `_db_handler` post-construction and
    set `_initialized` directly to bypass the real (env-DSN-driven) lazy
    connect in `_ensure_db_ready`.
    """
    container = make_mock_container()
    handler = HandlerValidationLedgerAppend(container)
    db_handler = AsyncMock()
    handler._db_handler = db_handler
    handler._initialized = initialized
    return handler, db_handler


def make_db_result(rows: list[dict[str, object]]) -> MagicMock:
    """Build a mock ModelHandlerOutput[ModelDbQueryResponse] with given rows."""
    correlation_id = uuid4()
    payload = ModelDbQueryPayload(rows=rows, row_count=len(rows))
    response = ModelDbQueryResponse(
        status=EnumResponseStatus.SUCCESS,
        payload=payload,
        correlation_id=correlation_id,
    )
    result_wrapper = MagicMock()
    result_wrapper.result = response
    return result_wrapper


def make_minimal_payload(
    **overrides: object,
) -> ModelPayloadValidationLedgerAppend:
    """Create a minimal valid ModelPayloadValidationLedgerAppend."""
    from datetime import UTC, datetime

    defaults: dict[str, object] = {
        "run_id": uuid4(),
        "repo_id": "omnibase_core",
        "event_type": "onex.evt.validation.cross-repo-run-started.v1",
        "event_version": "v1",
        "occurred_at": datetime.now(UTC),
        "kafka_topic": "onex.evt.validation.cross-repo-run-started.v1",
        "kafka_partition": 0,
        "kafka_offset": 42,
        "envelope_bytes": "SGVsbG8gV29ybGQ=",  # base64 "Hello World"
        "envelope_hash": "a" * 64,
    }
    defaults.update(overrides)
    return ModelPayloadValidationLedgerAppend(**defaults)  # type: ignore[arg-type]


# =============================================================================
# Constructor / Composition Tests
# =============================================================================


class TestHandlerValidationLedgerAppendComposition:
    """Tests that HandlerValidationLedgerAppend composes HandlerDb from `container` alone."""

    @pytest.mark.unit
    def test_constructor_takes_only_container(self) -> None:
        """HandlerValidationLedgerAppend(container) constructs without a db_handler arg."""
        container = make_mock_container()

        handler = HandlerValidationLedgerAppend(container)

        assert isinstance(handler._db_handler, HandlerDb)
        assert handler._initialized is False


# =============================================================================
# Initialization / Lazy DB Connection Tests
# =============================================================================


class TestHandlerValidationLedgerAppendInitialization:
    """Tests for HandlerValidationLedgerAppend's lazy HandlerDb connection lifecycle."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_initialize_raises_when_dsn_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """initialize() raises RuntimeHostError when no DSN is supplied."""
        monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        container = make_mock_container()
        handler = HandlerValidationLedgerAppend(container)

        with pytest.raises(RuntimeHostError, match="Missing PostgreSQL DSN"):
            await handler.initialize({})

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_initialize_connects_composed_db_handler_when_dsn_configured(
        self,
    ) -> None:
        """initialize() connects the composed HandlerDb using config DSN."""
        container = make_mock_container()
        handler = HandlerValidationLedgerAppend(container)
        handler._db_handler.initialize = AsyncMock()  # type: ignore[method-assign]

        await handler.initialize({"dsn": "postgresql://test-dsn"})

        handler._db_handler.initialize.assert_awaited_once_with(
            {"dsn": "postgresql://test-dsn"}
        )
        assert handler._initialized is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ensure_db_ready_is_idempotent(self) -> None:
        """Repeated _ensure_db_ready calls connect at most once."""
        container = make_mock_container()
        handler = HandlerValidationLedgerAppend(
            container, db_dsn="postgresql://test-dsn"
        )
        handler._db_handler.initialize = AsyncMock()  # type: ignore[method-assign]

        await handler._ensure_db_ready()
        await handler._ensure_db_ready()

        handler._db_handler.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_append_raises_when_dsn_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """append() raises RuntimeHostError when no DSN is supplied."""
        monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        container = make_mock_container()
        handler = HandlerValidationLedgerAppend(container)

        payload = make_minimal_payload()
        with pytest.raises(RuntimeHostError, match="Missing PostgreSQL DSN"):
            await handler.append(payload)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_shutdown_sets_initialized_false(self) -> None:
        """shutdown() marks the handler as not initialized and shuts down HandlerDb."""
        handler, db_handler = make_handler_with_mock_db(initialized=True)
        assert handler._initialized is True

        await handler.shutdown()

        assert handler._initialized is False
        db_handler.shutdown.assert_awaited_once()


# =============================================================================
# Duplicate Detection Tests
# =============================================================================


class TestHandlerValidationLedgerAppendDuplicateDetection:
    """Tests for duplicate detection via RETURNING clause."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_new_event_returns_ledger_entry_id(self) -> None:
        """When RETURNING produces a row, result has ledger_entry_id and duplicate=False."""
        handler, db_handler = make_handler_with_mock_db()
        ledger_entry_id = uuid4()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"id": str(ledger_entry_id)}])
        )

        payload = make_minimal_payload()
        result = await handler.append(payload)

        assert result.success is True
        assert result.duplicate is False
        assert result.ledger_entry_id == ledger_entry_id

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_duplicate_event_returns_no_entry_id(self) -> None:
        """When RETURNING produces no rows (ON CONFLICT), result has duplicate=True."""
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        payload = make_minimal_payload()
        result = await handler.append(payload)

        assert result.success is True
        assert result.duplicate is True
        assert result.ledger_entry_id is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_duplicate_preserves_kafka_position(self) -> None:
        """Duplicate result carries original kafka_topic/partition/offset for tracing."""
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        payload = make_minimal_payload(
            kafka_topic="onex.evt.validation.cross-repo-run-completed.v1",
            kafka_partition=3,
            kafka_offset=999,
        )
        result = await handler.append(payload)

        assert result.kafka_topic == "onex.evt.validation.cross-repo-run-completed.v1"
        assert result.kafka_partition == 3
        assert result.kafka_offset == 999


# =============================================================================
# Base64 Decode Error Tests
# =============================================================================


class TestHandlerValidationLedgerAppendBase64Errors:
    """Tests for base64 decode error handling."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalid_base64_raises_runtime_host_error(self) -> None:
        """Invalid base64 envelope_bytes raises RuntimeHostError."""
        handler, _db_handler = make_handler_with_mock_db()

        payload = make_minimal_payload(envelope_bytes="!!!not-valid-base64!!!")

        with pytest.raises(RuntimeHostError, match="Failed to decode base64"):
            await handler.append(payload)

    @pytest.mark.unit
    def test_decode_base64_valid_input(self) -> None:
        """_decode_base64 returns correct bytes for valid input."""
        handler, _db_handler = make_handler_with_mock_db()

        result = handler._decode_base64("SGVsbG8gV29ybGQ=")  # "Hello World"
        assert result == b"Hello World"

    @pytest.mark.unit
    def test_decode_base64_raises_binascii_error_path(self) -> None:
        """_decode_base64 raises RuntimeHostError wrapping binascii.Error."""
        import binascii

        handler, _db_handler = make_handler_with_mock_db()

        with pytest.raises(RuntimeHostError) as exc_info:
            handler._decode_base64("!!!not-valid!!!")

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, binascii.Error)


# =============================================================================
# Property Compliance Tests
# =============================================================================


class TestHandlerValidationLedgerAppendProperties:
    """Tests for handler classification properties."""

    @pytest.mark.unit
    def test_handler_type_is_infra_handler(self) -> None:
        """handler_type returns INFRA_HANDLER."""
        handler, _db_handler = make_handler_with_mock_db()

        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    @pytest.mark.unit
    def test_handler_category_is_effect(self) -> None:
        """handler_category returns EFFECT."""
        handler, _db_handler = make_handler_with_mock_db()

        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


# =============================================================================
# handle() Auto-Wiring Adapter Tests
# =============================================================================


class TestHandlerValidationLedgerAppendHandle:
    """Tests for the handle() auto-wiring entry point.

    node_validation_ledger_write_effect's contract declares operation_match
    routing with no event_model, so handler_wiring._make_dispatch_callback
    dispatches via handle(envelope) directly rather than execute(). This is
    also the exact shape IntentEffectDispatchBridge.execute() constructs when
    the kernel's intent-routing derivation invokes this handler.
    """

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_routes_to_append_and_returns_handler_output(self) -> None:
        """handle() extracts the payload, calls append(), and wraps the result."""
        handler, db_handler = make_handler_with_mock_db()
        ledger_entry_id = uuid4()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"id": str(ledger_entry_id)}])
        )

        payload = make_minimal_payload()
        envelope = MagicMock()
        envelope.payload = payload
        envelope.correlation_id = None

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "validation-ledger-append-handler"
        assert output.result is not None
        assert output.result.success is True
        assert output.result.duplicate is False
        assert output.result.ledger_entry_id == ledger_entry_id

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_accepts_dict_shaped_envelope(self) -> None:
        """handle() extracts payload/correlation_id from a dict envelope.

        This is the shape MessageDispatchEngine._materialize_envelope_with_bindings
        delivers on the real dispatch path AND the exact shape
        IntentEffectDispatchBridge.execute() constructs
        (``{"payload": ..., "correlation_id": ...}``) -- not an
        attribute-bearing object.
        """
        handler, db_handler = make_handler_with_mock_db()
        ledger_entry_id = uuid4()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(rows=[{"id": str(ledger_entry_id)}])
        )

        correlation_id = uuid4()
        envelope = {
            "payload": make_minimal_payload(),
            "correlation_id": correlation_id,
        }

        output = await handler.handle(envelope)

        assert output.result is not None
        assert output.result.success is True
        assert output.result.ledger_entry_id == ledger_entry_id
        assert output.correlation_id == correlation_id

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_coerces_dict_payload(self) -> None:
        """handle() validates a raw dict payload into ModelPayloadValidationLedgerAppend."""
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        payload_dict = make_minimal_payload().model_dump(mode="json")
        envelope = {"payload": payload_dict, "correlation_id": None}

        output = await handler.handle(envelope)

        assert output.result is not None
        assert output.result.duplicate is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_falls_back_to_fresh_uuid_when_no_correlation_id(
        self,
    ) -> None:
        """handle() never leaves correlation_id unset — it defaults to a fresh UUID."""
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        envelope = {
            "payload": make_minimal_payload(correlation_id=None),
            "correlation_id": None,
        }

        output = await handler.handle(envelope)

        assert output.correlation_id is not None

    @pytest.mark.unit
    def test_safe_correlation_id_parses_string_uuid(self) -> None:
        """_safe_correlation_id parses a string UUID into a UUID instance."""
        correlation_id = uuid4()
        assert (
            HandlerValidationLedgerAppend._safe_correlation_id(str(correlation_id))
            == correlation_id
        )

    @pytest.mark.unit
    def test_safe_correlation_id_falls_back_on_garbage(self) -> None:
        """_safe_correlation_id never raises — bad input yields a fresh UUID."""
        from uuid import UUID

        result = HandlerValidationLedgerAppend._safe_correlation_id("not-a-uuid")
        assert isinstance(result, UUID)


# =============================================================================
# DB Result Guard Tests
# =============================================================================


class TestHandlerValidationLedgerAppendDbResultGuard:
    """Tests for the guard on None db_result.result."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_none_db_result_raises_runtime_host_error(self) -> None:
        """If db_result.result is None, RuntimeHostError is raised."""
        handler, db_handler = make_handler_with_mock_db()

        none_wrapper = MagicMock()
        none_wrapper.result = None
        db_handler.execute = AsyncMock(return_value=none_wrapper)

        payload = make_minimal_payload()
        with pytest.raises(
            RuntimeHostError, match="Database operation returned no result"
        ):
            await handler.append(payload)


__all__ = [
    "TestHandlerValidationLedgerAppendComposition",
    "TestHandlerValidationLedgerAppendInitialization",
    "TestHandlerValidationLedgerAppendDuplicateDetection",
    "TestHandlerValidationLedgerAppendBase64Errors",
    "TestHandlerValidationLedgerAppendProperties",
    "TestHandlerValidationLedgerAppendHandle",
    "TestHandlerValidationLedgerAppendDbResultGuard",
]
