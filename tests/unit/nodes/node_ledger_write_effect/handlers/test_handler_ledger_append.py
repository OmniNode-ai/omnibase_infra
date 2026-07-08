# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerLedgerAppend.

Tests validate:
- Duplicate detection via RETURNING clause (rows vs empty rows)
- Base64 decode errors raise RuntimeHostError with proper context
- Lazy DB connection guard (missing DSN raises RuntimeHostError)
- Successful append returns ModelLedgerAppendResult with ledger_entry_id
- Protocol compliance via isinstance() check

Related Tickets:
    - OMN-1686: Add unit tests and minor fixes for NodeLedgerWriteEffect handlers
    - OMN-1647: Add PostgreSQL handlers for event ledger persistence
    - OMN-14140: HandlerDb composed internally from `container` (single-arg
      constructor) instead of accepted as a `db_handler` constructor argument,
      so the auto-wiring resolver can construct this handler.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.handler_db import HandlerDb
from omnibase_infra.handlers.models import ModelDbQueryPayload, ModelDbQueryResponse
from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_append import (
    HandlerLedgerAppend,
)
from omnibase_infra.nodes.node_registration_reducer.models.model_payload_ledger_append import (
    ModelPayloadLedgerAppend,
)

# =============================================================================
# Fixtures
# =============================================================================


def make_mock_container() -> MagicMock:
    """Create a minimal mock ModelONEXContainer."""
    return MagicMock(spec=ModelONEXContainer)


def make_handler_with_mock_db(
    initialized: bool = True,
) -> tuple[HandlerLedgerAppend, AsyncMock]:
    """Create a HandlerLedgerAppend with its composed HandlerDb replaced by a mock.

    HandlerLedgerAppend composes HandlerDb internally from `container`
    (OMN-14140) -- construction never takes a db_handler argument. Tests that
    need to control DB behavior replace `_db_handler` post-construction and
    set `_initialized` directly to bypass the real (env-DSN-driven) lazy
    connect in `_ensure_db_ready`.
    """
    container = make_mock_container()
    handler = HandlerLedgerAppend(container)
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


def make_minimal_payload(**overrides: object) -> ModelPayloadLedgerAppend:
    """Create a minimal valid ModelPayloadLedgerAppend."""
    defaults: dict[str, object] = {
        "topic": "test.events.v1",
        "partition": 0,
        "kafka_offset": 42,
        "event_value": "SGVsbG8gV29ybGQ=",  # base64 "Hello World"
    }
    defaults.update(overrides)
    return ModelPayloadLedgerAppend(**defaults)


# =============================================================================
# Constructor / Composition Tests (OMN-14140)
# =============================================================================


class TestHandlerLedgerAppendComposition:
    """Tests that HandlerLedgerAppend composes HandlerDb from `container` alone."""

    @pytest.mark.unit
    def test_constructor_takes_only_container(self) -> None:
        """HandlerLedgerAppend(container) constructs without a db_handler arg."""
        container = make_mock_container()

        handler = HandlerLedgerAppend(container)

        assert isinstance(handler._db_handler, HandlerDb)
        assert handler._initialized is False


# =============================================================================
# Initialization / Lazy DB Connection Tests
# =============================================================================


class TestHandlerLedgerAppendInitialization:
    """Tests for HandlerLedgerAppend's lazy HandlerDb connection lifecycle."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_initialize_raises_when_dsn_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """initialize() raises RuntimeHostError when no DSN is supplied.

        The auto-wiring resolver never calls initialize() on constructed
        handlers, so the composed HandlerDb only connects lazily -- and lazily
        connecting with no configured DSN must fail loudly, not silently no-op
        (OMN-14140). Handler-level code does not read environment directly.
        """
        monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        container = make_mock_container()
        handler = HandlerLedgerAppend(container)

        with pytest.raises(RuntimeHostError, match="Missing PostgreSQL DSN"):
            await handler.initialize({})

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_initialize_connects_composed_db_handler_when_dsn_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """initialize() connects the composed HandlerDb using config DSN."""
        container = make_mock_container()
        handler = HandlerLedgerAppend(container)
        handler._db_handler.initialize = AsyncMock()  # type: ignore[method-assign]

        await handler.initialize({"dsn": "postgresql://test-dsn"})

        handler._db_handler.initialize.assert_awaited_once_with(
            {"dsn": "postgresql://test-dsn"}
        )
        assert handler._initialized is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ensure_db_ready_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated _ensure_db_ready calls connect at most once."""
        container = make_mock_container()
        handler = HandlerLedgerAppend(container, db_dsn="postgresql://test-dsn")
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
        handler = HandlerLedgerAppend(container)

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


class TestHandlerLedgerAppendDuplicateDetection:
    """Tests for duplicate detection via RETURNING clause."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_new_event_returns_ledger_entry_id(self) -> None:
        """When RETURNING produces a row, result has ledger_entry_id and duplicate=False."""
        handler, db_handler = make_handler_with_mock_db()
        ledger_entry_id = uuid4()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(
                rows=[{"ledger_entry_id": str(ledger_entry_id)}]
            )
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
        # Empty rows = ON CONFLICT DO NOTHING was triggered
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        payload = make_minimal_payload()
        result = await handler.append(payload)

        assert result.success is True
        assert result.duplicate is True
        assert result.ledger_entry_id is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_duplicate_preserves_topic_partition_offset(self) -> None:
        """Duplicate result carries original topic/partition/offset for tracing."""
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        payload = make_minimal_payload(
            topic="prod.orders.v2", partition=3, kafka_offset=999
        )
        result = await handler.append(payload)

        assert result.topic == "prod.orders.v2"
        assert result.partition == 3
        assert result.kafka_offset == 999

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_new_event_preserves_topic_partition_offset(self) -> None:
        """Successful insert result carries original topic/partition/offset."""
        handler, db_handler = make_handler_with_mock_db()
        ledger_entry_id = uuid4()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(
                rows=[{"ledger_entry_id": str(ledger_entry_id)}]
            )
        )

        payload = make_minimal_payload(
            topic="dev.events.v1", partition=5, kafka_offset=1234
        )
        result = await handler.append(payload)

        assert result.topic == "dev.events.v1"
        assert result.partition == 5
        assert result.kafka_offset == 1234


# =============================================================================
# Base64 Decode Error Tests
# =============================================================================


class TestHandlerLedgerAppendBase64Errors:
    """Tests for base64 decode error handling."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalid_base64_raises_runtime_host_error(self) -> None:
        """Invalid base64 event_value raises RuntimeHostError."""
        handler, _db_handler = make_handler_with_mock_db()

        # Not valid base64 - contains invalid characters and wrong padding
        payload = make_minimal_payload(event_value="!!!not-valid-base64!!!")

        with pytest.raises(RuntimeHostError, match="Failed to decode base64"):
            await handler.append(payload)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalid_base64_event_key_raises_runtime_host_error(self) -> None:
        """Invalid base64 event_key raises RuntimeHostError."""
        handler, _db_handler = make_handler_with_mock_db()

        payload = make_minimal_payload(
            event_key="!!!not-valid-base64!!!",
            event_value="SGVsbG8=",  # "Hello" - valid
        )

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

        # Verify the cause is a binascii.Error
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, binascii.Error)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_none_event_key_skips_decode(self) -> None:
        """None event_key is not decoded - keyless events are supported."""
        handler, db_handler = make_handler_with_mock_db()
        ledger_entry_id = uuid4()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(
                rows=[{"ledger_entry_id": str(ledger_entry_id)}]
            )
        )

        # event_key=None - keyless event
        payload = make_minimal_payload(event_key=None)
        result = await handler.append(payload)

        assert result.success is True
        assert result.ledger_entry_id == ledger_entry_id


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestHandlerLedgerAppendProtocolCompliance:
    """Tests for ProtocolLedgerPersistence partial compliance.

    HandlerLedgerAppend implements the append() method of ProtocolLedgerPersistence.
    HandlerLedgerQuery implements the query methods. Together they form a full
    implementation. The isinstance() check requires all protocol methods on a single
    object, which does not apply to this split handler design.
    """

    @pytest.mark.unit
    def test_handler_has_append_method_matching_protocol(self) -> None:
        """HandlerLedgerAppend implements the append() method defined by ProtocolLedgerPersistence."""
        import inspect

        handler, _db_handler = make_handler_with_mock_db()

        # HandlerLedgerAppend implements the append() slice of the protocol
        assert hasattr(handler, "append")
        assert inspect.iscoroutinefunction(handler.append)

    @pytest.mark.unit
    def test_handler_does_not_implement_query_methods(self) -> None:
        """HandlerLedgerAppend correctly does not implement query methods - HandlerLedgerQuery owns those."""
        handler, _db_handler = make_handler_with_mock_db()

        # Query methods belong to HandlerLedgerQuery
        assert not hasattr(handler, "query_by_correlation_id")
        assert not hasattr(handler, "query_by_time_range")

    @pytest.mark.unit
    def test_handler_type_is_infra_handler(self) -> None:
        """handler_type returns INFRA_HANDLER."""
        from omnibase_infra.enums import EnumHandlerType

        handler, _db_handler = make_handler_with_mock_db()

        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    @pytest.mark.unit
    def test_handler_category_is_effect(self) -> None:
        """handler_category returns EFFECT."""
        from omnibase_infra.enums import EnumHandlerTypeCategory

        handler, _db_handler = make_handler_with_mock_db()

        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


# =============================================================================
# handle() Auto-Wiring Adapter Tests (OMN-14134)
# =============================================================================


class TestHandlerLedgerAppendHandle:
    """Tests for the handle() auto-wiring entry point.

    node_ledger_write_effect's contract declares operation_match routing with
    no event_model, so handler_wiring._make_dispatch_callback dispatches via
    handle(envelope) directly rather than execute(). Without handle(), the
    auto-wiring binds _missing_handle and every dispatched ledger-append
    command raises.
    """

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_routes_to_append_and_returns_handler_output(self) -> None:
        """handle() extracts the payload, calls append(), and wraps the result."""
        handler, db_handler = make_handler_with_mock_db()
        ledger_entry_id = uuid4()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(
                rows=[{"ledger_entry_id": str(ledger_entry_id)}]
            )
        )

        payload = make_minimal_payload()
        envelope = MagicMock()
        envelope.payload = payload
        envelope.correlation_id = None

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "ledger-append-handler"
        assert output.result is not None
        assert output.result.success is True
        assert output.result.duplicate is False
        assert output.result.ledger_entry_id == ledger_entry_id

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_coerces_dict_payload(self) -> None:
        """handle() validates a raw dict payload into ModelPayloadLedgerAppend."""
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        envelope = MagicMock()
        envelope.payload = {
            "topic": "test.events.v1",
            "partition": 0,
            "kafka_offset": 42,
            "event_value": "SGVsbG8gV29ybGQ=",
        }
        envelope.correlation_id = None

        output = await handler.handle(envelope)

        assert output.result is not None
        assert output.result.topic == "test.events.v1"
        assert output.result.duplicate is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_propagates_envelope_correlation_id(self) -> None:
        """handle() copies correlation_id from the envelope onto the output."""
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        correlation_id = uuid4()
        envelope = MagicMock()
        envelope.payload = make_minimal_payload()
        envelope.correlation_id = correlation_id

        output = await handler.handle(envelope)

        assert output.correlation_id == correlation_id

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_falls_back_to_fresh_uuid_when_no_correlation_id(
        self,
    ) -> None:
        """handle() never leaves correlation_id unset — it defaults to a fresh UUID."""
        handler, db_handler = make_handler_with_mock_db()
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        envelope = MagicMock()
        envelope.payload = make_minimal_payload(correlation_id=None)
        envelope.correlation_id = None

        output = await handler.handle(envelope)

        assert output.correlation_id is not None

    @pytest.mark.unit
    def test_safe_correlation_id_parses_string_uuid(self) -> None:
        """_safe_correlation_id parses a string UUID into a UUID instance."""
        correlation_id = uuid4()
        assert (
            HandlerLedgerAppend._safe_correlation_id(str(correlation_id))
            == correlation_id
        )

    @pytest.mark.unit
    def test_safe_correlation_id_falls_back_on_garbage(self) -> None:
        """_safe_correlation_id never raises — bad input yields a fresh UUID."""
        from uuid import UUID

        result = HandlerLedgerAppend._safe_correlation_id("not-a-uuid")
        assert isinstance(result, UUID)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_accepts_dict_shaped_envelope(self) -> None:
        """handle() extracts payload/correlation_id from a dict envelope.

        This is the shape MessageDispatchEngine._materialize_envelope_with_bindings
        actually delivers on the real dispatch path: a dict with a "payload" key,
        not an attribute-bearing object. A MagicMock-only test suite would miss
        this — getattr(dict, "payload", default) silently falls through to the
        default instead of doing a key lookup.
        """
        handler, db_handler = make_handler_with_mock_db()
        ledger_entry_id = uuid4()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(
                rows=[{"ledger_entry_id": str(ledger_entry_id)}]
            )
        )

        correlation_id = uuid4()
        envelope = {
            "payload": {
                "topic": "test.events.v1",
                "partition": 0,
                "kafka_offset": 42,
                "event_value": "SGVsbG8gV29ybGQ=",
            },
            "correlation_id": str(correlation_id),
            "partition_key": None,
        }

        output = await handler.handle(envelope)

        assert output.result is not None
        assert output.result.success is True
        assert output.result.ledger_entry_id == ledger_entry_id
        assert output.correlation_id == correlation_id

    @pytest.mark.unit
    def test_extract_envelope_field_reads_dict_key(self) -> None:
        """_extract_envelope_field does a key lookup for dict envelopes."""
        envelope = {"payload": {"topic": "x"}, "correlation_id": "abc"}

        assert HandlerLedgerAppend._extract_envelope_field(envelope, "payload") == {
            "topic": "x"
        }
        assert (
            HandlerLedgerAppend._extract_envelope_field(envelope, "correlation_id")
            == "abc"
        )
        assert HandlerLedgerAppend._extract_envelope_field(envelope, "missing") is None

    @pytest.mark.unit
    def test_extract_envelope_field_reads_object_attribute(self) -> None:
        """_extract_envelope_field falls back to attribute access for non-dict envelopes."""
        envelope = MagicMock()
        envelope.payload = {"topic": "x"}

        assert HandlerLedgerAppend._extract_envelope_field(envelope, "payload") == {
            "topic": "x"
        }


# =============================================================================
# DB Result Guard Tests
# =============================================================================


class TestHandlerLedgerAppendDbResultGuard:
    """Tests for the guard on None db_result.result."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_none_db_result_raises_runtime_host_error(self) -> None:
        """If db_result.result is None, RuntimeHostError is raised."""
        handler, db_handler = make_handler_with_mock_db()

        # result=None simulates an unexpected None from db handler
        none_wrapper = MagicMock()
        none_wrapper.result = None
        db_handler.execute = AsyncMock(return_value=none_wrapper)

        payload = make_minimal_payload()
        with pytest.raises(
            RuntimeHostError, match="Database operation returned no result"
        ):
            await handler.append(payload)


__all__ = [
    "TestHandlerLedgerAppendComposition",
    "TestHandlerLedgerAppendInitialization",
    "TestHandlerLedgerAppendDuplicateDetection",
    "TestHandlerLedgerAppendBase64Errors",
    "TestHandlerLedgerAppendProtocolCompliance",
    "TestHandlerLedgerAppendHandle",
    "TestHandlerLedgerAppendDbResultGuard",
]
