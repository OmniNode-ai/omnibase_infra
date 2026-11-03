# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The registration orchestrator package reads time only through an injected clock.

Related Tickets:
    - OMN-21698: Registration orchestrator clock injection
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.catalog.model_topic_catalog_query import (
    ModelTopicCatalogQuery,
)
from omnibase_infra.models.dispatch.model_injected_request import (
    ModelInjectedRequest,
)
from omnibase_infra.models.dispatch.model_system_clock import ModelSystemClock
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers.dispatcher_topic_catalog_query import (
    DispatcherTopicCatalogQuery,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_topic_catalog_query import (
    HandlerTopicCatalogQuery,
)
from omnibase_infra.protocols.protocol_clock import ProtocolClock

FIXED_NOW = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)


class _FixedClock:
    """A clock whose wall reading is fixed and whose monotonic reading steps."""

    def __init__(self, step_seconds: float) -> None:
        self._monotonic = 100.0
        self._step = step_seconds
        self.wall_reads = 0

    def now(self) -> datetime:
        self.wall_reads += 1
        return FIXED_NOW + timedelta(seconds=self.wall_reads - 1)

    def monotonic(self) -> float:
        reading = self._monotonic
        self._monotonic += self._step
        return reading


def _query() -> ModelTopicCatalogQuery:
    return ModelTopicCatalogQuery(correlation_id=uuid4(), client_id="client")


@pytest.mark.unit
def test_system_clock_satisfies_the_clock_protocol() -> None:
    clock = ModelSystemClock()

    assert isinstance(clock, ProtocolClock)
    assert isinstance(_FixedClock(1.0), ProtocolClock)
    assert clock.now().tzinfo is not None
    assert clock.monotonic() <= clock.monotonic()
    assert clock == ModelSystemClock()


@pytest.mark.unit
async def test_handler_reads_durations_from_the_request_clock() -> None:
    handler = HandlerTopicCatalogQuery(catalog_service=None)
    request = ModelInjectedRequest(
        event=_query(),
        now=FIXED_NOW,
        input_id=uuid4(),
        clock=_FixedClock(step_seconds=0.25),
    )

    output = await handler.handle(request)

    assert output.processing_time_ms == pytest.approx(250.0)
    assert output.events[0].generated_at == FIXED_NOW


@pytest.mark.unit
async def test_dispatcher_hands_its_clock_to_the_handler_request() -> None:
    clock = _FixedClock(step_seconds=1.0)
    handler = MagicMock()
    handler.handle = AsyncMock(
        return_value=ModelHandlerOutput(
            input_envelope_id=uuid4(),
            correlation_id=uuid4(),
            handler_id="handler-topic-catalog-query",
            node_kind=EnumNodeKind.ORCHESTRATOR,
            events=(),
            intents=(),
            projections=(),
            result=None,
            processing_time_ms=0.0,
            timestamp=FIXED_NOW,
        )
    )
    dispatcher = DispatcherTopicCatalogQuery(handler=handler, clock=clock)
    envelope = ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=_query(),
        envelope_timestamp=FIXED_NOW,
        correlation_id=uuid4(),
        source_tool="test",
    )

    result = await dispatcher.handle(envelope)

    request = handler.handle.await_args.args[0]
    assert request.clock is clock
    assert request.now == FIXED_NOW + timedelta(seconds=1)
    assert result.started_at == FIXED_NOW
    assert result.completed_at == FIXED_NOW + timedelta(seconds=2)
