# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The canary's command record carries the Kafka headers the ledger writer reads (OMN-24562).

``node_ledger_projection_compute`` takes ``correlation_id`` and ``envelope_id``
for ``public.event_ledger`` from the record's Kafka HEADERS, and the chain
writer assembles a chain by selecting ``event_ledger`` on that column. A record
published with no headers is recorded under a freshly minted correlation id, so
the ``delegate-skill`` command hop never joins the probe's chain and link 5
reads ``ledger_chain_incomplete``.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from uuid import UUID, uuid4

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.handlers import (
    handler_chain_canary as hc,
)


class _Metadata:
    partition = 0
    offset = 7


class _FakeProducer:
    instances: list[_FakeProducer] = []

    def __init__(self, **_: object) -> None:
        self.sent: list[dict[str, Any]] = []
        _FakeProducer.instances.append(self)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send_and_wait(self, topic: str, **kwargs: object) -> _Metadata:
        self.sent.append({"topic": topic, **kwargs})
        return _Metadata()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_command_record_carries_correlation_and_envelope_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeProducer.instances.clear()
    monkeypatch.setitem(
        sys.modules, "aiokafka", types.SimpleNamespace(AIOKafkaProducer=_FakeProducer)
    )
    correlation_id = str(uuid4())
    envelope_id = str(uuid4())
    envelope: dict[str, object] = {
        "payload": {"correlation_id": correlation_id},
        "envelope_id": envelope_id,
        "envelope_timestamp": "2026-11-25T03:09:39Z",
        "correlation_id": correlation_id,
        "source_tool": "chain-canary",
        "event_type": "omnimarket.delegate-skill",
    }

    ack, error, _ = await hc._publish_to_bus_via_aiokafka(
        "broker:9092", "onex.cmd.omnimarket.delegate-skill.v1", envelope, 5.0
    )

    assert error == ""
    assert ack is not None
    sent = _FakeProducer.instances[0].sent[0]
    headers: dict[str, bytes] = dict(sent["headers"])
    assert UUID(headers["correlation_id"].decode()) == UUID(correlation_id)
    assert UUID(headers["message_id"].decode()) == UUID(envelope_id)
    assert headers["source"] == b"chain-canary"
    assert headers["event_type"] == b"omnimarket.delegate-skill"
    assert "timestamp" in headers
    # The body is unchanged: headers are added, nothing is removed from it.
    assert json.loads(sent["value"])["envelope_id"] == envelope_id
