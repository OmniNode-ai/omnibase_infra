# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection callback keeps envelope time transport-owned (OMN-18326).

A real ``ModelEventEnvelope`` is dispatched through the projection callback.
A payload that carries its own ``_envelope_timestamp`` must never reach the
handler in place of the authoritative envelope time, and must not survive when
the envelope records no usable time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.topic_constants import derive_event_type_alias_for_topic
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_projection_dispatch_callback,
)
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)

pytestmark = pytest.mark.integration

_TOPIC = "onex.evt.platform.node-heartbeat.v1"
_SPOOFED = datetime(2000, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _configured_projection_dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(
        monkeypatch, url="postgresql://user:pass@host:5432/omnidash_analytics"
    )


async def _await(awaitable: Awaitable[object]) -> object:
    return await awaitable


def _dispatch(envelope: Any) -> list[dict[str, object]]:
    received: list[dict[str, object]] = []

    class Handler:
        def handle(self, input_data: dict[str, object]) -> dict[str, int]:
            received.append(dict(input_data))
            return {"rows_upserted": 1}

    callback = _make_projection_dispatch_callback(
        Handler(),
        projection_database_target("live_events", schema="omninode_internal"),
        (_TOPIC,),
    )
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get",
        return_value="postgresql://user:pass@host:5432/omnidash_analytics",
    ):
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter",
            return_value=MagicMock(),
        ):
            asyncio.run(_await(callback(envelope)))
    return received


def test_envelope_time_wins_over_payload_supplied_time() -> None:
    recorded = datetime(2026, 9, 13, 17, 44, 8, tzinfo=UTC)
    envelope = ModelEventEnvelope[object](
        payload={"service_name": "svc-a", "_envelope_timestamp": _SPOOFED},
        envelope_id=uuid4(),
        envelope_timestamp=recorded,
        event_type=derive_event_type_alias_for_topic(_TOPIC),
    )

    received = _dispatch(envelope)

    assert len(received) == 1
    assert received[0]["_envelope_timestamp"] == recorded


def test_payload_supplied_time_dropped_when_envelope_has_none() -> None:
    envelope = MagicMock()
    envelope.event_type = derive_event_type_alias_for_topic(_TOPIC)
    envelope.topic = _TOPIC
    envelope.payload = {"service_name": "svc-a", "_envelope_timestamp": _SPOOFED}
    envelope.envelope_id = uuid4()
    envelope.envelope_timestamp = None

    received = _dispatch(envelope)

    assert len(received) == 1
    assert "_envelope_timestamp" not in received[0]
