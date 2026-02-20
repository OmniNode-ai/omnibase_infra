# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerLedgerAppend.

Tests validate:
- Duplicate detection via RETURNING clause (rows vs empty rows)
- Base64 decode errors raise RuntimeHostError with proper context
- Initialization guard (not initialized raises RuntimeHostError)
- Successful append returns ModelLedgerAppendResult with ledger_entry_id
- Protocol compliance via isinstance() check

Related Tickets:
    - OMN-1686: Add unit tests and minor fixes for NodeLedgerWriteEffect handlers
    - OMN-1647: Add PostgreSQL handlers for event ledger persistence
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.models import ModelDbQueryPayload, ModelDbQueryResponse
from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_append import (
    HandlerLedgerAppend,
)
from omnibase_infra.nodes.node_ledger_write_effect.protocols.protocol_ledger_persistence import (
    ProtocolLedgerPersistence,
)
from omnibase_infra.nodes.reducers.models.model_payload_ledger_append import (
    ModelPayloadLedgerAppend,
)

# =============================================================================
# Fixtures
# =============================================================================


def make_mock_container() -> MagicMock:
    """Create a minimal mock ModelONEXContainer."""
    return MagicMock(spec=ModelONEXContainer)


def make_mock_db_handler(initialized: bool = True) -> AsyncMock:
    """Create a mock HandlerDb with _initialized attribute."""
    mock = AsyncMock()
    mock._initialized = initialized
    return mock


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
# Initialization Tests
# =============================================================================


class TestHandlerLedgerAppendInitialization:
    """Tests for HandlerLedgerAppend initialization lifecycle."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_initialize_succeeds_when_db_handler_ready(self) -> None:
        """initialize() completes when HandlerDb._initialized is True."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        handler = HandlerLedgerAppend(container, db_handler)
        await handler.initialize({})

        assert handler._initialized is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_initialize_raises_when_db_not_initialized(self) -> None:
        """initialize() raises RuntimeHostError if HandlerDb is not yet initialized."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=False)

        handler = HandlerLedgerAppend(container, db_handler)

        with pytest.raises(RuntimeHostError, match="HandlerDb must be initialized"):
            await handler.initialize({})

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_append_raises_when_not_initialized(self) -> None:
        """append() raises RuntimeHostError if handler not initialized."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        handler = HandlerLedgerAppend(container, db_handler)
        # Do NOT call initialize() - handler remains in uninitialized state

        payload = make_minimal_payload()
        with pytest.raises(RuntimeHostError, match="not initialized"):
            await handler.append(payload)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_shutdown_sets_initialized_false(self) -> None:
        """shutdown() marks handler as not initialized."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        handler = HandlerLedgerAppend(container, db_handler)
        await handler.initialize({})
        assert handler._initialized is True

        await handler.shutdown()
        assert handler._initialized is False


# =============================================================================
# Duplicate Detection Tests
# =============================================================================


class TestHandlerLedgerAppendDuplicateDetection:
    """Tests for duplicate detection via RETURNING clause."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_new_event_returns_ledger_entry_id(self) -> None:
        """When RETURNING produces a row, result has ledger_entry_id and duplicate=False."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        ledger_entry_id = uuid4()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(
                rows=[{"ledger_entry_id": str(ledger_entry_id)}]
            )
        )

        handler = HandlerLedgerAppend(container, db_handler)
        await handler.initialize({})

        payload = make_minimal_payload()
        result = await handler.append(payload)

        assert result.success is True
        assert result.duplicate is False
        assert result.ledger_entry_id == ledger_entry_id

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_duplicate_event_returns_no_entry_id(self) -> None:
        """When RETURNING produces no rows (ON CONFLICT), result has duplicate=True."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        # Empty rows = ON CONFLICT DO NOTHING was triggered
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        handler = HandlerLedgerAppend(container, db_handler)
        await handler.initialize({})

        payload = make_minimal_payload()
        result = await handler.append(payload)

        assert result.success is True
        assert result.duplicate is True
        assert result.ledger_entry_id is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_duplicate_preserves_topic_partition_offset(self) -> None:
        """Duplicate result carries original topic/partition/offset for tracing."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)
        db_handler.execute = AsyncMock(return_value=make_db_result(rows=[]))

        handler = HandlerLedgerAppend(container, db_handler)
        await handler.initialize({})

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
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)
        ledger_entry_id = uuid4()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(
                rows=[{"ledger_entry_id": str(ledger_entry_id)}]
            )
        )

        handler = HandlerLedgerAppend(container, db_handler)
        await handler.initialize({})

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
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        handler = HandlerLedgerAppend(container, db_handler)
        await handler.initialize({})

        # Not valid base64 - contains invalid characters and wrong padding
        payload = make_minimal_payload(event_value="!!!not-valid-base64!!!")

        with pytest.raises(RuntimeHostError, match="Failed to decode base64"):
            await handler.append(payload)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_invalid_base64_event_key_raises_runtime_host_error(self) -> None:
        """Invalid base64 event_key raises RuntimeHostError."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        handler = HandlerLedgerAppend(container, db_handler)
        await handler.initialize({})

        payload = make_minimal_payload(
            event_key="!!!not-valid-base64!!!",
            event_value="SGVsbG8=",  # "Hello" - valid
        )

        with pytest.raises(RuntimeHostError, match="Failed to decode base64"):
            await handler.append(payload)

    @pytest.mark.unit
    def test_decode_base64_valid_input(self) -> None:
        """_decode_base64 returns correct bytes for valid input."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        handler = HandlerLedgerAppend(container, db_handler)

        result = handler._decode_base64("SGVsbG8gV29ybGQ=")  # "Hello World"
        assert result == b"Hello World"

    @pytest.mark.unit
    def test_decode_base64_raises_binascii_error_path(self) -> None:
        """_decode_base64 raises RuntimeHostError wrapping binascii.Error."""
        import binascii

        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        handler = HandlerLedgerAppend(container, db_handler)

        with pytest.raises(RuntimeHostError) as exc_info:
            handler._decode_base64("!!!not-valid!!!")

        # Verify the cause is a binascii.Error
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, binascii.Error)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_none_event_key_skips_decode(self) -> None:
        """None event_key is not decoded - keyless events are supported."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)
        ledger_entry_id = uuid4()
        db_handler.execute = AsyncMock(
            return_value=make_db_result(
                rows=[{"ledger_entry_id": str(ledger_entry_id)}]
            )
        )

        handler = HandlerLedgerAppend(container, db_handler)
        await handler.initialize({})

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

        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        handler = HandlerLedgerAppend(container, db_handler)

        # HandlerLedgerAppend implements the append() slice of the protocol
        assert hasattr(handler, "append")
        assert inspect.iscoroutinefunction(handler.append)

    @pytest.mark.unit
    def test_handler_does_not_implement_query_methods(self) -> None:
        """HandlerLedgerAppend correctly does not implement query methods - HandlerLedgerQuery owns those."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        handler = HandlerLedgerAppend(container, db_handler)

        # Query methods belong to HandlerLedgerQuery
        assert not hasattr(handler, "query_by_correlation_id")
        assert not hasattr(handler, "query_by_time_range")

    @pytest.mark.unit
    def test_handler_type_is_infra_handler(self) -> None:
        """handler_type returns INFRA_HANDLER."""
        from omnibase_infra.enums import EnumHandlerType

        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        handler = HandlerLedgerAppend(container, db_handler)

        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    @pytest.mark.unit
    def test_handler_category_is_effect(self) -> None:
        """handler_category returns EFFECT."""
        from omnibase_infra.enums import EnumHandlerTypeCategory

        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        handler = HandlerLedgerAppend(container, db_handler)

        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


# =============================================================================
# DB Result Guard Tests
# =============================================================================


class TestHandlerLedgerAppendDbResultGuard:
    """Tests for the guard on None db_result.result."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_none_db_result_raises_runtime_host_error(self) -> None:
        """If db_result.result is None, RuntimeHostError is raised."""
        container = make_mock_container()
        db_handler = make_mock_db_handler(initialized=True)

        # result=None simulates an unexpected None from db handler
        none_wrapper = MagicMock()
        none_wrapper.result = None
        db_handler.execute = AsyncMock(return_value=none_wrapper)

        handler = HandlerLedgerAppend(container, db_handler)
        await handler.initialize({})

        payload = make_minimal_payload()
        with pytest.raises(
            RuntimeHostError, match="Database operation returned no result"
        ):
            await handler.append(payload)


__all__ = [
    "TestHandlerLedgerAppendInitialization",
    "TestHandlerLedgerAppendDuplicateDetection",
    "TestHandlerLedgerAppendBase64Errors",
    "TestHandlerLedgerAppendProtocolCompliance",
    "TestHandlerLedgerAppendDbResultGuard",
]
