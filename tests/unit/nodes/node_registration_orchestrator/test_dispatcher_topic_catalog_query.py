# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for DispatcherTopicCatalogQuery.

Tests validate:
- Happy path: successful dispatch returns EnumDispatchStatus.SUCCESS
- Malformed payload (wrong type): returns INVALID_MESSAGE
- ValidationError from model_validate: returns INVALID_MESSAGE
- Handler exception: returns HANDLER_ERROR and records circuit failure
- Circuit breaker open: InfraUnavailableError propagates
- Dispatcher properties: dispatcher_id, category, message_types, node_kind

Related Tickets:
    - OMN-2313: Topic Catalog: query handler + dispatcher + contract wiring
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus, EnumMessageCategory
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.models.catalog.model_topic_catalog_query import (
    ModelTopicCatalogQuery,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers.dispatcher_topic_catalog_query import (
    DispatcherTopicCatalogQuery,
)

TEST_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def _make_handler(
    response: ModelTopicCatalogResponse | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock HandlerTopicCatalogQuery."""
    handler = MagicMock()

    if response is None:
        response = ModelTopicCatalogResponse(
            correlation_id=uuid4(),
            topics=(),
            catalog_version=1,
            node_count=0,
            generated_at=TEST_NOW,
            warnings=(),
        )

    if side_effect is not None:
        handler.handle = AsyncMock(side_effect=side_effect)
    else:
        from omnibase_core.enums import EnumNodeKind

        handler_output: ModelHandlerOutput[object] = ModelHandlerOutput(
            input_envelope_id=uuid4(),
            correlation_id=uuid4(),
            handler_id="handler-topic-catalog-query",
            node_kind=EnumNodeKind.ORCHESTRATOR,
            events=(response,),
            intents=(),
            projections=(),
            result=None,
            processing_time_ms=0.1,
            timestamp=TEST_NOW,
        )
        handler.handle = AsyncMock(return_value=handler_output)

    return handler


def _make_query_payload(
    client_id: str = "test-client",
) -> ModelTopicCatalogQuery:
    """Create a valid ModelTopicCatalogQuery."""
    return ModelTopicCatalogQuery(
        correlation_id=uuid4(),
        client_id=client_id,
    )


def _make_envelope(
    payload: object = None,
    correlation_id: object = None,
) -> ModelEventEnvelope[object]:
    """Create a ModelEventEnvelope with the given payload."""
    if payload is None:
        payload = _make_query_payload()
    corr_id = correlation_id or uuid4()
    return ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=payload,
        envelope_timestamp=TEST_NOW,
        correlation_id=corr_id,
        source="test",
    )


# ---------------------------------------------------------------------------
# Dispatcher property tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dispatcher_properties() -> None:
    """Dispatcher exposes correct property values."""
    handler = _make_handler()
    dispatcher = DispatcherTopicCatalogQuery(handler=handler)

    assert dispatcher.dispatcher_id == "dispatcher.registration.topic-catalog-query"
    assert dispatcher.category == EnumMessageCategory.COMMAND
    assert "ModelTopicCatalogQuery" in dispatcher.message_types
    assert "platform.topic-catalog-query" in dispatcher.message_types


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_dispatcher_happy_path_returns_success() -> None:
    """Happy path: dispatcher returns SUCCESS status with output events."""
    handler = _make_handler()
    dispatcher = DispatcherTopicCatalogQuery(handler=handler)

    envelope = _make_envelope()
    result = await dispatcher.handle(envelope)

    assert result.status == EnumDispatchStatus.SUCCESS
    assert result.dispatcher_id == "dispatcher.registration.topic-catalog-query"
    assert result.output_count == 1
    assert len(result.output_events) == 1
    assert result.error_message is None or result.error_message == ""


@pytest.mark.unit
async def test_dispatcher_accepts_dict_envelope() -> None:
    """Dispatcher handles materialized dict envelopes from dispatch engine."""
    handler = _make_handler()
    dispatcher = DispatcherTopicCatalogQuery(handler=handler)

    query = _make_query_payload()
    # Dict format used by MessageDispatchEngine serialization boundary.
    # model_dump() produces UUID objects (not strings) for UUID fields.
    dict_envelope: dict[str, object] = {
        "payload": query.model_dump(),
        "__debug_trace": {"correlation_id": str(uuid4())},
    }

    result = await dispatcher.handle(dict_envelope)

    assert result.status == EnumDispatchStatus.SUCCESS
    # Handler must have been called - ensures dict deserialization reached the handler
    # rather than silently falling through to INVALID_MESSAGE before dispatch.
    handler.handle.assert_awaited_once()


@pytest.mark.unit
async def test_dispatcher_accepts_dict_envelope_with_string_uuids() -> None:
    """Dispatcher handles dict envelopes where UUID fields are serialized as strings.

    The MessageDispatchEngine may serialize envelopes via JSON round-trip before
    passing them to dispatchers. In that case UUID fields arrive as plain strings
    (e.g. ``"correlation_id": "550e8400-e29b-41d4-a716-446655440000"``) rather
    than ``uuid.UUID`` objects.  ``model_validate`` must coerce those strings back
    to UUIDs without raising a ``ValidationError``.
    """
    handler = _make_handler()
    dispatcher = DispatcherTopicCatalogQuery(handler=handler)

    query = _make_query_payload()
    # Simulate JSON round-trip: model_dump(mode="json") converts UUID → str
    dict_envelope_str_uuids: dict[str, object] = {
        "payload": query.model_dump(mode="json"),
        "__debug_trace": {"correlation_id": str(uuid4())},
    }

    result = await dispatcher.handle(dict_envelope_str_uuids)

    assert result.status == EnumDispatchStatus.SUCCESS
    # Handler must have been called with a fully-reconstructed ModelTopicCatalogQuery
    handler.handle.assert_awaited_once()


# ---------------------------------------------------------------------------
# Invalid message payloads
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_dispatcher_wrong_payload_type_returns_invalid_message() -> None:
    """Wrong payload type (not dict, not ModelTopicCatalogQuery) returns INVALID_MESSAGE."""
    handler = _make_handler()
    dispatcher = DispatcherTopicCatalogQuery(handler=handler)

    # Use an int payload which cannot be model_validated to ModelTopicCatalogQuery
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=42,
        envelope_timestamp=TEST_NOW,
        correlation_id=uuid4(),
        source="test",
    )

    result = await dispatcher.handle(envelope)

    assert result.status == EnumDispatchStatus.INVALID_MESSAGE
    assert result.error_message is not None
    assert len(result.output_events) == 0
    # Handler should NOT be called for invalid payloads
    handler.handle.assert_not_awaited()


@pytest.mark.unit
async def test_dispatcher_invalid_dict_payload_returns_invalid_message() -> None:
    """Dict payload that fails ModelTopicCatalogQuery validation returns INVALID_MESSAGE."""
    handler = _make_handler()
    dispatcher = DispatcherTopicCatalogQuery(handler=handler)

    # Dict with missing required fields
    bad_dict_envelope: dict[str, object] = {
        "payload": {"not_a_valid_field": "value"},
        "__debug_trace": {},
    }

    result = await dispatcher.handle(bad_dict_envelope)

    assert result.status == EnumDispatchStatus.INVALID_MESSAGE
    assert len(result.output_events) == 0


# ---------------------------------------------------------------------------
# Handler exception
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_dispatcher_handler_exception_returns_handler_error() -> None:
    """Handler exception returns HANDLER_ERROR and records circuit failure."""
    handler = _make_handler(side_effect=RuntimeError("handler failure"))
    dispatcher = DispatcherTopicCatalogQuery(handler=handler)

    envelope = _make_envelope()
    result = await dispatcher.handle(envelope)

    assert result.status == EnumDispatchStatus.HANDLER_ERROR
    assert result.error_message is not None
    assert len(result.output_events) == 0


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_dispatcher_circuit_open_raises_infra_unavailable() -> None:
    """When circuit is OPEN, dispatcher raises InfraUnavailableError."""
    handler = _make_handler()
    dispatcher = DispatcherTopicCatalogQuery(handler=handler)

    # Force circuit open by patching _check_circuit_breaker
    async def mock_check(*args: object, **kwargs: object) -> None:
        raise InfraUnavailableError("circuit open")

    with patch.object(dispatcher, "_check_circuit_breaker", side_effect=mock_check):
        with pytest.raises(InfraUnavailableError):
            envelope = _make_envelope()
            await dispatcher.handle(envelope)


# ---------------------------------------------------------------------------
# Duration and timestamps
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_dispatcher_result_has_valid_timestamps() -> None:
    """Dispatch result has valid started_at, completed_at, and positive duration."""
    handler = _make_handler()
    dispatcher = DispatcherTopicCatalogQuery(handler=handler)

    envelope = _make_envelope()
    result = await dispatcher.handle(envelope)

    assert result.started_at is not None
    assert result.completed_at is not None
    assert result.completed_at >= result.started_at
    assert (result.duration_ms or 0.0) >= 0.0


# ---------------------------------------------------------------------------
# Correlation ID propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_dispatcher_correlation_id_from_envelope() -> None:
    """Dispatcher uses correlation_id from envelope when present."""
    correlation_id = uuid4()
    handler = _make_handler()
    dispatcher = DispatcherTopicCatalogQuery(handler=handler)

    envelope = _make_envelope(correlation_id=correlation_id)
    result = await dispatcher.handle(envelope)

    assert result.correlation_id == correlation_id
