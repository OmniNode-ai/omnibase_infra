# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18385 -- the credential must be gone across the whole dead-letter chain.

The unit tests prove each piece: the model masks its own field, and the
publisher strips credential-named fields out of a body. Neither proves the
thing that actually matters, which is that a real attach command entering the
real dead-letter publisher comes out the other side, is parsed by the real
replay-side reader, and reaches the real eligibility decision with the
credential absent at every hop.

That chain is where the live defect lived. The producing side and the
consuming side are in different packages, were written by different tickets,
and agree only by convention: the publisher writes ``original_message.value``
and a redaction record, and the replay reader has to find both. A unit test on
either side passes happily while the two disagree about the envelope.

So this composes the real components end to end:

    ModelGatewayAttachRequest  (the real inbound model, secret-wrapped)
        -> the real boundary serialisation onto a Kafka record body
        -> MixinKafkaDlq._publish_raw_to_dlq   (the real publisher)
        -> the published envelope BYTES
        -> ModelDlqMessage.from_kafka_message  (the real replay-side reader)
        -> should_replay                       (the real eligibility decision)

Only the Kafka producer is a double, and only because a broker is not the
subject: the assertion is about the bytes handed to the producer, which are
byte-identical to what a live broker would receive. That is deliberate -- a
test that skips without a broker would not have gated this PR, and this defect
is exactly the kind that reaches production behind a skipped test.

Every token here is a sentinel. No real credential appears in this file.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

from omnibase_infra.event_bus.mixin_kafka_dlq import MixinKafkaDlq
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_request import (
    ModelGatewayAttachRequest,
)
from omnibase_infra.utils.util_dlq_credential_redaction import DLQ_REDACTION_MARKER

pytestmark = pytest.mark.integration

# onex-topic-allow: quotes the live topic the leaked records were read from
_DLQ_TOPIC = "onex.dlq.omnibase-infra.commands.v1"
# onex-topic-allow: quotes the live topic whose records were dead-lettered
_ORIGINAL_TOPIC = "onex.cmd.omnibase-infra.gateway-attach-request.v1"

_SENTINEL_TOKEN = "OMN18385-ROUNDTRIP-SENTINEL-000000000000000000000000"
_SENTINEL_EDGE = "OMN18385-ROUNDTRIP-EDGE-must-survive"


class _CapturingProducer:
    """Stands in for the broker. Records exactly the bytes it is handed."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_and_wait(
        self,
        topic: str,
        *,
        value: bytes,
        key: bytes | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> object:
        self.sent.append(
            {"topic": topic, "value": value, "key": key, "headers": headers}
        )
        return object()


class _DlqHost(MixinKafkaDlq):
    def __init__(self) -> None:
        self._config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            dead_letter_topic=_DLQ_TOPIC,
        )
        self._environment = "integration"
        self._group = "omn18385-roundtrip"
        self._producer = _CapturingProducer()  # type: ignore[assignment]
        self._producer_lock = asyncio.Lock()
        self._timeout_seconds = 5
        self._init_dlq()

    def _model_headers_to_kafka(self, headers: object) -> list[tuple[str, bytes]]:
        return []


class _InboundRecord:
    """The shape the boundary hands the raw publisher: bytes off the wire."""

    def __init__(self, value: bytes) -> None:
        self.key = b"edge-1"
        self.value = value
        self.offset = 137
        self.partition = 0
        self.headers = ()


def _inbound_attach_bytes(correlation_id: UUID) -> bytes:
    """A real attach command serialised the way the producer puts it on the bus.

    Built from the REAL model rather than a hand-written dict, so if the
    model's field names or shape drift, this test drifts with them.
    """
    request = ModelGatewayAttachRequest(
        access_token=SecretStr(_SENTINEL_TOKEN),
        edge_instance_id=_SENTINEL_EDGE,
    )
    # The producing edge holds the real credential and puts it on the wire;
    # that is the inbound fact this whole chain exists to handle safely.
    payload = request.model_dump(mode="json")
    payload["access_token"] = request.access_token.get_secret_value()
    return json.dumps(
        {
            "envelope_id": str(uuid4()),
            "correlation_id": str(correlation_id),
            "envelope_timestamp": "2026-09-14T22:03:15.114000+00:00",
            "payload": payload,
        }
    ).encode("utf-8")


async def _dead_letter(value: bytes, correlation_id: UUID) -> dict[str, Any]:
    host = _DlqHost()
    published = await host._publish_raw_to_dlq(
        original_topic=_ORIGINAL_TOPIC,
        raw_msg=_InboundRecord(value),
        error=ValueError("Keycloak introspection endpoint returned HTTP 401"),
        correlation_id=correlation_id,
        failure_type="handler_exception",
        consumer_group="omn18385-roundtrip",
    )
    assert published is True, "the publisher reported the write as not persisted"
    producer: _CapturingProducer = host._producer  # type: ignore[assignment]
    assert len(producer.sent) == 1, producer.sent
    return producer.sent[0]


@pytest.mark.asyncio
async def test_attach_credential_is_absent_at_every_hop_of_the_chain() -> None:
    """The whole point: the sentinel must not survive anywhere on the chain."""
    correlation_id = uuid4()
    inbound = _inbound_attach_bytes(correlation_id)
    # Control on the INPUT: the credential really is present going in, so a
    # pass below cannot be an artefact of the fixture never carrying one.
    assert _SENTINEL_TOKEN.encode("utf-8") in inbound

    sent = await _dead_letter(inbound, correlation_id)

    # Hop 1 -- the bytes handed to the broker.
    assert _SENTINEL_TOKEN.encode("utf-8") not in sent["value"]
    for _name, header_value in sent["headers"] or []:
        assert _SENTINEL_TOKEN.encode("utf-8") not in header_value
    assert _SENTINEL_TOKEN.encode("utf-8") not in (sent["key"] or b"")

    envelope = json.loads(sent["value"].decode("utf-8"))

    # Hop 2 -- the envelope the replay side will read.
    assert envelope["correlation_id"] == str(correlation_id)
    assert envelope["original_topic"] == _ORIGINAL_TOPIC
    assert envelope["redacted_fields"] == ["payload.access_token"]
    body = json.loads(envelope["original_message"]["value"])
    assert body["payload"]["access_token"] == DLQ_REDACTION_MARKER
    # Positive control: the record is still a usable forensic artefact.
    assert body["payload"]["edge_instance_id"] == _SENTINEL_EDGE
    assert body["correlation_id"] == str(correlation_id)

    # Hop 3 -- the real replay-side reader parses the real published envelope.
    # This is the hop a per-side unit test cannot cover: it proves the two
    # packages agree on the envelope, rather than each agreeing with itself.
    parsed = ModelDlqMessage.from_kafka_message(
        payload=envelope,
        dlq_offset=137,
        dlq_partition=0,
    )
    assert parsed.correlation_id == correlation_id
    assert parsed.original_topic == _ORIGINAL_TOPIC
    assert parsed.redacted_fields == ("payload.access_token",)
    assert _SENTINEL_TOKEN not in parsed.original_value

    # Hop 4 -- the real eligibility decision refuses to republish the body.
    eligible, reason = should_replay(
        parsed,
        ModelDlqReplayEngineConfig(
            bootstrap_servers="localhost:9092",
            dlq_topic=_DLQ_TOPIC,
        ),
    )
    assert eligible is False, (
        "a redacted body was judged replayable; republishing it would put a "
        "command carrying a redaction marker where its credential belongs "
        "back onto the original topic"
    )
    assert "payload.access_token" in reason


@pytest.mark.asyncio
async def test_a_command_with_no_credential_traverses_the_chain_unchanged() -> None:
    """Positive control for the whole chain, not just one hop.

    A redactor that passed the test above by blanking every body would fail
    here, and so would a replay guard that refused everything.
    """
    correlation_id = uuid4()
    inbound = json.dumps(
        {
            "correlation_id": str(correlation_id),
            "payload": {"edge_instance_id": _SENTINEL_EDGE, "max_tokens": 128},
        }
    ).encode("utf-8")

    sent = await _dead_letter(inbound, correlation_id)
    envelope = json.loads(sent["value"].decode("utf-8"))

    assert "redacted_fields" not in envelope
    assert envelope["original_message"]["value"] == inbound.decode("utf-8")

    parsed = ModelDlqMessage.from_kafka_message(
        payload=envelope, dlq_offset=1, dlq_partition=0
    )
    assert parsed.redacted_fields == ()
    eligible, reason = should_replay(
        parsed,
        ModelDlqReplayEngineConfig(
            bootstrap_servers="localhost:9092",
            dlq_topic=_DLQ_TOPIC,
        ),
    )
    assert eligible is True, reason
