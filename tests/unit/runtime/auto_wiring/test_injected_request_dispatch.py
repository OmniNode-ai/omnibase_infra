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
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
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
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProtocolHandleable,
    _make_dispatch_callback,
)
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef

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
async def test_envelope_without_clock_or_id_gets_fresh_ones() -> None:
    handler = _RecordingHandler()
    event_model = ModelHandlerRef(name="_Event", module=__name__)

    await _dispatch(
        handler,
        event_model,
        {"payload": {"value": "beat"}, "__debug_trace": {"topic": "t"}},
    )

    request = handler.received[0]
    assert isinstance(request, ModelInjectedRequest)
    assert request.now.tzinfo is not None
    assert isinstance(request.input_id, UUID)
    assert request.correlation_id is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_production_handler_receives_request_through_the_callback() -> None:
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
