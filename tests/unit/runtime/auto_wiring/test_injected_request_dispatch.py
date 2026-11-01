# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime unwraps the envelope into a typed request for handlers (OMN-21536).

A handler whose ``handle()`` takes ``ModelInjectedRequest[ModelX]`` never sees the
event envelope: the auto-wiring dispatch callback reads the envelope's clock reading,
id and correlation id and hands them over on the request.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus, EnumMessageCategory
from omnibase_infra.models.catalog.model_topic_catalog_query import (
    ModelTopicCatalogQuery,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.dispatch.model_injected_request import (
    ModelInjectedRequest,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_topic_catalog_query import (
    HandlerTopicCatalogQuery,
)
from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProtocolHandleable,
    _make_dispatch_callback,
)
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_NODE_HEARTBEAT

ENVELOPE_ID = UUID("11111111-1111-4111-8111-111111111111")
CORRELATION_ID = UUID("22222222-2222-4222-8222-222222222222")
CLOCK_READING = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)


class _Event(BaseModel):
    value: str


class _RecordingHandler:
    def __init__(self) -> None:
        self.received: list[object] = []

    async def handle(
        self, request: ModelInjectedRequest[_Event]
    ) -> ModelHandlerOutput[object]:
        self.received.append(request)
        return ModelHandlerOutput(
            input_envelope_id=request.input_id,
            correlation_id=request.correlation_id or uuid4(),
            handler_id="handler-recording",
            node_kind=EnumNodeKind.ORCHESTRATOR,
            events=(),
            intents=(),
            projections=(),
            result=None,
            processing_time_ms=0.0,
            timestamp=request.now,
        )


async def _dispatch(
    handler: object, event_model: ModelHandlerRef, message: object
) -> ModelDispatchResult | None:
    callback = _make_dispatch_callback(cast("ProtocolHandleable", handler), event_model)
    return await callback(cast("ModelEventEnvelope[object]", message))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_materialized_dict_envelope_becomes_injected_request() -> None:
    handler = _RecordingHandler()
    event_model = ModelHandlerRef(name="_Event", module=__name__)

    await _dispatch(
        handler,
        event_model,
        {
            "payload": {"value": "beat"},
            "envelope_id": str(ENVELOPE_ID),
            "envelope_timestamp": CLOCK_READING.isoformat(),
            "correlation_id": str(CORRELATION_ID),
        },
    )

    assert len(handler.received) == 1
    request = handler.received[0]
    assert isinstance(request, ModelInjectedRequest)
    assert request.event == _Event(value="beat")
    assert request.now == CLOCK_READING
    assert request.input_id == ENVELOPE_ID
    assert request.correlation_id == CORRELATION_ID


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_envelope_object_becomes_injected_request() -> None:
    handler = _RecordingHandler()
    event_model = ModelHandlerRef(name="_Event", module=__name__)
    envelope: ModelEventEnvelope[_Event] = ModelEventEnvelope(
        envelope_id=ENVELOPE_ID,
        payload=_Event(value="beat"),
        envelope_timestamp=CLOCK_READING,
        correlation_id=CORRELATION_ID,
        source_tool="test",
    )

    await _dispatch(handler, event_model, envelope)

    request = handler.received[0]
    assert isinstance(request, ModelInjectedRequest)
    assert not isinstance(request, ModelEventEnvelope)
    assert request.now == CLOCK_READING
    assert request.input_id == ENVELOPE_ID
    assert request.correlation_id == CORRELATION_ID


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["envelope_timestamp", "envelope_id"])
async def test_envelope_without_clock_or_id_is_rejected(missing: str) -> None:
    handler = _RecordingHandler()
    event_model = ModelHandlerRef(name="_Event", module=__name__)

    envelope = {
        "payload": {"value": "beat"},
        "envelope_timestamp": CLOCK_READING.isoformat(),
        "envelope_id": str(ENVELOPE_ID),
    }
    del envelope[missing]
    with pytest.raises(ValueError):
        await _dispatch(handler, event_model, envelope)
    assert handler.received == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_production_handler_receives_request_through_the_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real registration handler is driven through the real dispatch callback."""
    correlation_id = uuid4()
    service = MagicMock()
    service.build_catalog = AsyncMock(
        return_value=ModelTopicCatalogResponse(
            correlation_id=correlation_id,
            topics=(),
            catalog_version=0,
            node_count=0,
            generated_at=CLOCK_READING,
            warnings=(),
        )
    )
    handler = HandlerTopicCatalogQuery(catalog_service=service)
    seen: list[ModelInjectedRequest[ModelTopicCatalogQuery]] = []
    outputs: list[ModelHandlerOutput[object]] = []
    real_handle = handler.handle

    async def _spy(request: ModelInjectedRequest[ModelTopicCatalogQuery]) -> object:
        seen.append(request)
        output = await real_handle(request)
        outputs.append(output)
        return output

    monkeypatch.setattr(handler, "handle", _spy)
    result = await _dispatch(
        handler,
        ModelHandlerRef(
            name="ModelTopicCatalogQuery",
            module="omnibase_infra.models.catalog.model_topic_catalog_query",
        ),
        {
            "payload": ModelTopicCatalogQuery(
                correlation_id=correlation_id, client_id="client"
            ).model_dump(mode="json"),
            "envelope_id": str(ENVELOPE_ID),
            "envelope_timestamp": CLOCK_READING.isoformat(),
            "correlation_id": str(correlation_id),
        },
    )

    assert isinstance(result, ModelDispatchResult)
    service.build_catalog.assert_awaited_once()
    assert result.correlation_id == correlation_id
    # The injected clock and input id reach the handler, not just the payload.
    assert seen[0].input_id == ENVELOPE_ID
    assert seen[0].now == CLOCK_READING
    assert outputs[0].input_envelope_id == ENVELOPE_ID
    assert outputs[0].timestamp == CLOCK_READING


@pytest.mark.unit
@pytest.mark.parametrize(
    "timestamp",
    [CLOCK_READING.isoformat(), CLOCK_READING.isoformat().replace("+00:00", "Z")],
)
def test_from_dispatch_parses_aware_iso_timestamps(timestamp: str) -> None:
    request = ModelInjectedRequest.from_dispatch(
        {"envelope_timestamp": timestamp, "envelope_id": str(ENVELOPE_ID)},
        _Event(value="x"),
    )
    assert request.now == CLOCK_READING
    assert request.input_id == ENVELOPE_ID


@pytest.mark.unit
def test_from_dispatch_keeps_naive_timestamp_for_handler_validation() -> None:
    request = ModelInjectedRequest.from_dispatch(
        {"envelope_timestamp": "2026-01-01T00:00:00", "envelope_id": ENVELOPE_ID},
        _Event(value="x"),
    )
    assert request.now.tzinfo is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "header",
    [
        {"envelope_id": "not-a-uuid"},
        {"envelope_id": ""},
        {"envelope_id": None},
        {"envelope_id": ENVELOPE_ID.bytes},
        {"correlation_id": "not-a-uuid"},
        {"correlation_id": ""},
        {"correlation_id": 123},
        {"correlation_id": CORRELATION_ID.bytes},
        {"envelope_timestamp": "not-a-time"},
        {"envelope_timestamp": "123"},
        {"envelope_timestamp": 123},
        {"envelope_timestamp": None},
        {"envelope_timestamp": {}},
    ],
)
def test_from_dispatch_rejects_malformed_headers(header: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ModelInjectedRequest.from_dispatch(
            {
                "envelope_timestamp": CLOCK_READING,
                "envelope_id": ENVELOPE_ID,
                **header,
            },
            _Event(value="x"),
        )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("header", "field"),
    [
        ({"envelope_id": None}, "envelope_id"),
        ({"envelope_timestamp": "123"}, "envelope_timestamp"),
        ({"correlation_id": 123}, "correlation_id"),
    ],
)
async def test_dispatch_metadata_failure_bypasses_payload_fallback(
    monkeypatch: pytest.MonkeyPatch, header: dict[str, object], field: str
) -> None:
    handler = _RecordingHandler()
    fallback = MagicMock(return_value=None)
    monkeypatch.setattr(
        handler_wiring, "_build_inference_intent_validation_failure_result", fallback
    )
    with pytest.raises(ValueError, match=f"Dispatch {field}"):
        await _dispatch(
            handler,
            ModelHandlerRef(name="_Event", module=__name__),
            {
                "payload": {"value": "valid-domain-event"},
                "envelope_timestamp": CLOCK_READING,
                "envelope_id": ENVELOPE_ID,
                **header,
            },
        )
    fallback.assert_not_called()
    assert handler.received == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pre_dispatch_guard_rejects_before_metadata_is_unwrapped() -> None:
    handler = _RecordingHandler()

    class _GuardRejectedError(Exception):
        pass

    def guard(_payload: object) -> None:
        raise _GuardRejectedError

    callback = _make_dispatch_callback(
        cast("ProtocolHandleable", handler),
        ModelHandlerRef(name="_Event", module=__name__),
        pre_dispatch_guard=guard,
    )
    with pytest.raises(_GuardRejectedError):
        await callback(
            cast(
                "ModelEventEnvelope[object]",
                {
                    "payload": {"value": "guarded"},
                    "envelope_timestamp": "not-a-timestamp",
                    "envelope_id": ENVELOPE_ID,
                },
            )
        )
    assert handler.received == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_materialization_preserves_injected_identity_on_replay() -> None:
    handler = _RecordingHandler()
    engine = MessageDispatchEngine()
    callback = _make_dispatch_callback(
        cast("ProtocolHandleable", handler),
        ModelHandlerRef(name="_Event", module=__name__),
    )
    engine.register_dispatcher("injected", callback, EnumMessageCategory.EVENT)
    engine.register_route(
        ModelDispatchRoute(
            route_id="injected-route",
            topic_pattern=SUFFIX_NODE_HEARTBEAT,
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="injected",
        )
    )
    engine.freeze()
    envelope = ModelEventEnvelope(
        payload=_Event(value="replay"),
        envelope_id=ENVELOPE_ID,
        envelope_timestamp=CLOCK_READING,
        correlation_id=CORRELATION_ID,
        source_tool="test",
    )
    for _ in range(2):
        result = await engine.dispatch(SUFFIX_NODE_HEARTBEAT, envelope)
        assert result.status == EnumDispatchStatus.SUCCESS
    assert len(handler.received) == 2
    assert handler.received[0] == handler.received[1]
    for request in handler.received:
        assert isinstance(request, ModelInjectedRequest)
        assert request.input_id == ENVELOPE_ID
        assert request.now == CLOCK_READING
        assert request.correlation_id == CORRELATION_ID


@pytest.mark.unit
@pytest.mark.asyncio
async def test_callback_does_not_introspect_per_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _RecordingHandler()
    callback = _make_dispatch_callback(
        cast("ProtocolHandleable", handler),
        ModelHandlerRef(name="_Event", module=__name__),
    )

    def unexpected_introspection(*args: object, **kwargs: object) -> None:
        pytest.fail("Handler signature introspected during dispatch")

    for name in (
        "_resolve_def_b_input_model_type",
        "_handler_accepts_event_envelope",
        "_handler_declares_typed_event_envelope",
    ):
        monkeypatch.setattr(handler_wiring, name, unexpected_introspection)
    for _ in range(2):
        await callback(
            ModelEventEnvelope(
                payload=_Event(value="beat"),
                envelope_id=ENVELOPE_ID,
                envelope_timestamp=CLOCK_READING,
                source_tool="test",
            )
        )
    assert len(handler.received) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_domain_payload_never_reaches_handler() -> None:
    handler = _RecordingHandler()
    with pytest.raises(ValueError):
        await _dispatch(
            handler,
            ModelHandlerRef(name="_Event", module=__name__),
            {
                "payload": {"wrong": "field"},
                "envelope_timestamp": CLOCK_READING,
                "envelope_id": ENVELOPE_ID,
            },
        )
    assert handler.received == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_model", [None, ModelHandlerRef(name="_Event", module=__name__)]
)
async def test_payload_handler_still_receives_validated_event(
    event_model: ModelHandlerRef | None,
) -> None:
    received: list[_Event] = []

    class PayloadHandler:
        def handle(self, event: _Event) -> _Event:
            received.append(event)
            return event

    callback = _make_dispatch_callback(
        cast("ProtocolHandleable", PayloadHandler()), event_model
    )
    result = await callback(
        {"payload": {"value": "payload"}, "__bindings": {}, "__debug_trace": {}}
    )
    assert received == [_Event(value="payload")]
    assert isinstance(result, ModelDispatchResult)
    assert result.output_events == [_Event(value="payload")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_envelope_handler_output_is_normalized() -> None:
    class EnvelopeHandler:
        async def handle(self, envelope: ModelEventEnvelope[_Event]) -> _Event:
            assert isinstance(envelope, ModelEventEnvelope)
            assert isinstance(envelope.payload, _Event)
            return envelope.payload

    result = await _dispatch(
        EnvelopeHandler(),
        ModelHandlerRef(name="_Event", module=__name__),
        {"payload": {"value": "enveloped"}, "__bindings": {}, "__debug_trace": {}},
    )
    assert isinstance(result, ModelDispatchResult)
    assert result.output_events == [_Event(value="enveloped")]
