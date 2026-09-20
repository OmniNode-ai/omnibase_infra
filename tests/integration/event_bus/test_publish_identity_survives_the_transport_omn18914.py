# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18914: identity crosses the transport boundary intact.

WHAT THIS COVERS THAT THE UNIT TEST DOES NOT

    The unit test asserts what the PRODUCER puts on the wire. This asserts
    that the CONSUMER reads back the same thing -- the property the defect
    actually violated. The producer was not merely emitting an odd header; it
    was emitting one the consumer then recorded as the record's identity, so
    the record landed in ``public.event_ledger`` under a correlation no other
    hop shared and the delegation chain's head hop became unjoinable.

    Both halves here are real production code on either side of the wire
    format: ``EventBusKafka.publish`` serializing through
    ``_model_headers_to_kafka``, and ``_kafka_headers_to_model`` parsing the
    captured record back. Nothing in between is reimplemented by the test, and
    the header list that crosses is the exact object handed to the producer.

    No broker is required, deliberately: the seam under test is the wire
    FORMAT contract between the two halves, and binding this proof to a live
    Redpanda would make it skippable on the runner that most needs it.

WHY THE CONTROL MATTERS

    ``test_an_identityless_body_still_round_trips`` passes on both sides of
    the fix. A consumer-side parser that minted on absence was always going to
    return SOME uuid; what changed is whether it is the same one the body
    states. A suite that had gone uniformly red would fail that case too.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelKafkaEventBusConfig

DELEGATE_SKILL_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"


@pytest.fixture
def mock_producer() -> AsyncMock:
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer._closed = False

    record_metadata = MagicMock()
    record_metadata.partition = 0
    record_metadata.offset = 11

    async def _send(*args: object, **kwargs: object) -> asyncio.Future[object]:
        future = asyncio.get_running_loop().create_future()
        future.set_result(record_metadata)
        return future

    producer.send = AsyncMock(side_effect=_send)
    return producer


@pytest.fixture
async def bus(mock_producer: AsyncMock) -> EventBusKafka:
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ):
        event_bus = EventBusKafka(
            config=ModelKafkaEventBusConfig(
                bootstrap_servers="localhost:9092",
                environment="integration",
            )
        )
        yield event_bus
        try:
            await event_bus.close()
        except Exception:  # noqa: BLE001 — boundary: best-effort cleanup
            pass


async def _publish_and_read_back(
    bus: EventBusKafka, mock_producer: AsyncMock, body: dict[str, object]
) -> tuple[object, dict[str, object]]:
    """Publish a body, then parse the captured record as the consumer does."""
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ):
        await bus.start()
        await bus.publish(DELEGATE_SKILL_TOPIC, None, json.dumps(body).encode("utf-8"))

    call = mock_producer.send.call_args
    consumed = bus._kafka_headers_to_model(call[1]["headers"])
    return consumed, json.loads(call[1]["value"])


@pytest.mark.asyncio
async def test_the_consumer_reads_back_the_correlation_the_body_states(
    bus: EventBusKafka, mock_producer: AsyncMock
) -> None:
    """The record's recorded identity IS the delegation's own.

    This is the defect at the level it did damage: the consumer side is what
    writes ``event_ledger.correlation_id``, and before this change it wrote a
    value that appeared nowhere in the message it was describing.
    """
    correlation = uuid4()
    envelope_id = uuid4()

    consumed, body = await _publish_and_read_back(
        bus,
        mock_producer,
        {
            "payload": {"prompt": "Reply with the single word: alive."},
            "envelope_id": str(envelope_id),
            "correlation_id": str(correlation),
        },
    )

    assert str(consumed.correlation_id) == body["correlation_id"]
    assert str(consumed.message_id) == body["envelope_id"]


@pytest.mark.asyncio
async def test_the_record_is_not_read_back_as_unknown(
    bus: EventBusKafka, mock_producer: AsyncMock
) -> None:
    """A headerless record parses as ``unknown``, and that is the fingerprint.

    On the .201 dev lane every gateway-produced row on this topic carried
    ``source=unknown`` and ``event_type=unknown``, which is how the defect was
    found. A record published through this path must never read that way.
    """
    consumed, _ = await _publish_and_read_back(
        bus,
        mock_producer,
        {"correlation_id": str(uuid4()), "payload": {"prompt": "alive"}},
    )

    assert consumed.source != "unknown"
    assert consumed.event_type == DELEGATE_SKILL_TOPIC


@pytest.mark.asyncio
async def test_an_identityless_body_still_round_trips(
    bus: EventBusKafka, mock_producer: AsyncMock
) -> None:
    """POSITIVE CONTROL — passes on both sides of the fix.

    A body stating no identity still publishes and still parses into a
    well-formed header model. What the fix changes is whose identity that is,
    not whether one exists.
    """
    consumed, _ = await _publish_and_read_back(
        bus, mock_producer, {"payload": {"prompt": "nothing to derive here"}}
    )

    assert consumed.correlation_id is not None
    assert consumed.message_id is not None


@pytest.mark.asyncio
async def test_the_head_records_no_causal_parent(
    bus: EventBusKafka, mock_producer: AsyncMock
) -> None:
    """NEGATIVE CONTROL — deriving identity must not invent an edge.

    ``parent_message_id`` absent is the checkable statement that a message is
    a chain HEAD, and the delegate-skill command is one. A derivation that
    also filled in a parent would turn a broken chain into a verifiable-
    looking one, which is the failure OMN-18116 guarded against.
    """
    consumed, _ = await _publish_and_read_back(
        bus,
        mock_producer,
        {"correlation_id": str(uuid4()), "envelope_id": str(uuid4())},
    )

    assert consumed.parent_message_id is None
