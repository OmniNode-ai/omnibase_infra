# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16418 AC2 — a record that fails handler dispatch must never be
re-published to its own source topic.

Root cause (recorded on OMN-16418, 2026-08-23): the replay storm on
``onex.evt.omniclaude.session-started.v1`` was ``node_dlq_replay_effect``
being auto-wired as a per-message trigger for a whole-topic batch drain
(OMN-16422, fixed) — NOT the boundary DLQ destination resolving to the
source topic. That destination-resolution question was still an open,
falsifiable AC on OMN-16418 and had no test pinning it: OMN-16422's own PR
(#2858) shipped six tests, all bounding the *drain loop*, none asserting
the *republish target*.

This test closes that gap directly against the boundary seam a handler
exception actually reaches (``_make_event_bus_callback``'s failure path,
``handler_wiring.py`` ~L5559-5572): on a deliberately-failing handler, the
DLQ publish must target a genuine ``onex.dlq.*`` topic derived via
``get_dlq_topic_for_original(topic)`` — never the original source topic
itself. Existing coverage in ``test_boundary_dlq_omn14507.py``
(``test_flag_on_routes_to_dlq_instead_of_vanishing``) asserts
``original_topic`` is recorded correctly but never asserts the *publish
destination* (``dlq_topic``) is distinct from it — that is the precise
assertion this file adds.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.topic_constants import get_dlq_topic_for_original
from omnibase_infra.runtime.auto_wiring.handler_wiring import _BOUNDARY_DLQ_ENV
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_event_bus_callback as _make_contract_scoped_event_bus_callback,
)

pytestmark = pytest.mark.unit


def _make_event_bus_callback(
    topic: str,
    dispatch_engine: object,
    result_applier: object | None = None,
    **kwargs: object,
) -> Callable[..., Awaitable[None]]:
    """Build the boundary under its required synthetic contract scope."""
    return _make_contract_scoped_event_bus_callback(
        topic,
        dispatch_engine,  # type: ignore[arg-type]
        result_applier=result_applier,  # type: ignore[arg-type]
        allowed_dispatcher_ids={"test-dispatcher"},
        **kwargs,  # type: ignore[arg-type]
    )


def _dlq_capable_event_bus() -> MagicMock:
    """A bus mock spec'd to EventBusKafka so only real attributes (including
    _publish_raw_to_dlq, from MixinKafkaDlq) can be set -- satisfies the
    transport-mock-lint gate (OMN-13026)."""
    bus = MagicMock(spec=EventBusKafka)
    bus._publish_raw_to_dlq = AsyncMock()
    return bus


def _raising_dispatch_engine(exc: Exception) -> MagicMock:
    engine = MagicMock()
    engine.dispatch_scoped = AsyncMock(side_effect=exc)
    return engine


def _envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object].model_construct(
        event_type="onex.cmd.test.v1",
        payload={},
        correlation_id=uuid4(),
    )


class TestBoundaryDlqDoesNotRepublishToSourceTopic:
    """AC2: a deliberately-failing handler's record must land on a real
    DLQ topic, never back on the topic it was consumed from."""

    @pytest.mark.asyncio
    async def test_failed_dispatch_dlq_target_is_not_the_source_topic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        source_topic = "onex.evt.omniclaude.session-started.v1"
        boom = RuntimeError("deliberately failing handler")
        dispatch_engine = _raising_dispatch_engine(boom)
        event_bus = _dlq_capable_event_bus()

        callback = _make_event_bus_callback(
            source_topic,
            dispatch_engine,
            event_bus=event_bus,
        )

        await callback(_envelope())  # must not raise

        event_bus._publish_raw_to_dlq.assert_awaited_once()
        call_kwargs = event_bus._publish_raw_to_dlq.call_args.kwargs

        # The precise OMN-16418 AC2 assertion: the resolved publish
        # destination is a genuine onex.dlq.* topic, and it is NOT the
        # source topic the failing record was consumed from -- the record
        # does not re-enter its own topic and therefore cannot self-feed a
        # republish storm on it.
        assert call_kwargs["dlq_topic"] != source_topic
        assert call_kwargs["dlq_topic"] == get_dlq_topic_for_original(source_topic)
        assert call_kwargs["dlq_topic"].startswith("onex.dlq.")
        # original_topic is still recorded accurately for attribution --
        # the record is *tagged* with its source, just never *published to* it.
        assert call_kwargs["original_topic"] == source_topic

    @pytest.mark.asyncio
    async def test_repeated_failures_never_republish_to_source_topic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even across the full bounded retry budget, every DLQ publish
        attempt targets the derived DLQ topic -- the source topic is never
        once used as a publish destination on this failure path."""
        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        source_topic = "onex.evt.omniclaude.session-started.v1"
        dispatch_engine = _raising_dispatch_engine(RuntimeError("always fails"))
        event_bus = _dlq_capable_event_bus()

        callback = _make_event_bus_callback(
            source_topic,
            dispatch_engine,
            event_bus=event_bus,
        )

        await callback(_envelope())

        for call in event_bus._publish_raw_to_dlq.call_args_list:
            assert call.kwargs["dlq_topic"] != source_topic
