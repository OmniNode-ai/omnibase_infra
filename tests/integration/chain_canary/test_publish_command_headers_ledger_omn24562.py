# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The canary's command record joins the probe's chain in the ledger (OMN-24562).

Drives the real publish function, hands the headers it stamped to the real
Kafka-header decoder of the event bus, and projects the resulting message through
the real ledger projection handler. The ledger row's ``correlation_id`` and
``envelope_id`` must be the canary's own, not freshly minted ones.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_chain_canary_effect.handlers import (
    handler_chain_canary as hc,
)
from omnibase_infra.nodes.node_ledger_projection_compute.handlers.handler_ledger_projection import (
    HandlerLedgerProjection,
)


class _Metadata:
    partition = 0
    offset = 7


class _RecordingProducer:
    sent: list[dict[str, Any]] = []

    def __init__(self, **_: object) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send_and_wait(self, topic: str, **kwargs: Any) -> _Metadata:
        _RecordingProducer.sent.append({"topic": topic, **kwargs})
        return _Metadata()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_published_command_projects_under_the_probe_correlation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingProducer.sent.clear()
    monkeypatch.setitem(
        sys.modules,
        "aiokafka",
        types.SimpleNamespace(AIOKafkaProducer=_RecordingProducer),
    )
    correlation_id = uuid4()
    envelope_id = uuid4()
    envelope: dict[str, object] = {
        "payload": {"correlation_id": str(correlation_id)},
        "envelope_id": str(envelope_id),
        "envelope_timestamp": "2026-11-25T03:09:39+00:00",
        "correlation_id": str(correlation_id),
        "source_tool": "chain-canary",
        "event_type": "omnimarket.delegate-skill",
    }
    topic = "onex.cmd.omnimarket.delegate-skill.v1"

    ack, error, _ = await hc._publish_to_bus_via_aiokafka(
        "broker:9092", topic, envelope, 5.0
    )
    assert error == ""
    assert ack is not None
    record = _RecordingProducer.sent[0]

    bus = EventBusKafka.__new__(EventBusKafka)
    message = ModelEventMessage(
        topic=topic,
        key=record["key"],
        value=record["value"],
        headers=bus._kafka_headers_to_model(record["headers"]),
        partition=0,
        offset="7",
    )
    intent = HandlerLedgerProjection(MagicMock()).project(message)
    payload = intent.payload

    assert UUID(str(payload.correlation_id)) == correlation_id
    assert UUID(str(payload.envelope_id)) == envelope_id
