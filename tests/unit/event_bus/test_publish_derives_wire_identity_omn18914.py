# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18914: a headerless publish derives its wire identity from the body.

WHY THIS EXISTS

    ``EventBusKafka.publish()`` used to construct a fresh ``ModelEventHeaders``
    whenever a caller passed ``headers=None``. ``correlation_id`` and
    ``message_id`` both default to ``uuid4()``, so the wire identity was
    invented at publish time and bore no relation to the identity inside the
    message body.

    ``handler_ledger_projection`` fills ``public.event_ledger``'s identity
    columns from the HEADERS, never from the body, so a correlation-scoped read
    could not find such a message and the delegation chain's head hop went
    missing. Measured on the .201 dev lane: a live delegation whose body said
    ``da706588-8c8f-42c9-b80e-44ef2da77ab0`` reached the wire carrying header
    correlation ``3564c149-6d75-4dc0-b0dd-bd80f709e371``, and its
    ``ledger_chain`` held no head hop at all.

WHAT IS ASSERTED, AND AGAINST WHAT

    Every assertion reads the KAFKA HEADER LIST the producer was actually
    called with -- the output of ``_model_headers_to_kafka`` -- not a
    ``ModelEventHeaders`` object the test built. A change that derived identity
    into the model and dropped it before serialization would pass a
    model-level assertion and fail these.

    ``test_minted_identity_is_still_minted_for_an_identityless_body`` is the
    positive control: it passes on both sides of the fix, so a suite that has
    simply gone uniformly red is distinguishable from the defect.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelKafkaEventBusConfig

TEST_BOOTSTRAP_SERVERS = "localhost:9092"
TEST_ENVIRONMENT = "test"

# The real topic the defect was found on. Named rather than a placeholder so
# the ONEX topic-format gate this publish passes through is the one the live
# path passes through.
DELEGATE_SKILL_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"


@pytest.fixture
def mock_producer() -> AsyncMock:
    """A producer whose send() records its call and resolves like the real one."""
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer._closed = False

    record_metadata = MagicMock()
    record_metadata.partition = 0
    record_metadata.offset = 7

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
        config = ModelKafkaEventBusConfig(
            bootstrap_servers=TEST_BOOTSTRAP_SERVERS,
            environment=TEST_ENVIRONMENT,
        )
        event_bus = EventBusKafka(config=config)
        yield event_bus
        try:
            await event_bus.close()
        except Exception:  # noqa: BLE001 — boundary: best-effort cleanup
            pass


def _wire_header(mock_producer: AsyncMock, name: str) -> str | None:
    """Read one header off the producer call, decoded, or None if absent."""
    headers = mock_producer.send.call_args[1]["headers"]
    for key, value in headers:
        if key == name:
            return bytes(value).decode("utf-8")
    return None


async def _publish(
    bus: EventBusKafka, mock_producer: AsyncMock, body: object, *, topic: str
) -> None:
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ):
        await bus.start()
        payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        await bus.publish(topic, None, payload)


@pytest.mark.unit
class TestPublishDerivesWireIdentityFromBody:
    """AC1, AC2 and AC3 of OMN-18914."""

    @pytest.mark.asyncio
    async def test_body_correlation_reaches_the_wire(
        self, bus: EventBusKafka, mock_producer: AsyncMock
    ) -> None:
        """AC1 — the header correlation IS the body's, not a minted one.

        This is the defect, written as an assertion. On the parent commit the
        wire carries a uuid4 that appears nowhere in the body.
        """
        correlation = uuid4()
        await _publish(
            bus,
            mock_producer,
            {
                "prompt": "Reply with the single word: alive.",
                "task_type": "test",
                "correlation_id": str(correlation),
            },
            topic=DELEGATE_SKILL_TOPIC,
        )

        assert _wire_header(mock_producer, "correlation_id") == str(correlation)

    @pytest.mark.asyncio
    async def test_body_envelope_id_becomes_the_wire_message_id(
        self, bus: EventBusKafka, mock_producer: AsyncMock
    ) -> None:
        """AC1 — an envelope's own id is its message id, not a second identity.

        ``event_ledger.envelope_id`` is filled from the ``message_id`` header,
        and the chain's causal edges are recorded against that column, so an
        envelope publishing under a different id cannot be anybody's parent.
        """
        correlation = uuid4()
        envelope_id = uuid4()
        await _publish(
            bus,
            mock_producer,
            {
                "payload": {"prompt": "alive"},
                "envelope_id": str(envelope_id),
                "correlation_id": str(correlation),
                "event_type": "omnimarket.delegate-skill",
            },
            topic=DELEGATE_SKILL_TOPIC,
        )

        assert _wire_header(mock_producer, "message_id") == str(envelope_id)
        assert _wire_header(mock_producer, "correlation_id") == str(correlation)

    @pytest.mark.asyncio
    async def test_minted_identity_is_still_minted_for_an_identityless_body(
        self, bus: EventBusKafka, mock_producer: AsyncMock
    ) -> None:
        """AC2, POSITIVE CONTROL — passes on both sides of the fix.

        A body with nothing to derive from must still publish, with a minted
        identity. If this goes red the suite has broken, not the defect.
        """
        await _publish(
            bus,
            mock_producer,
            {"prompt": "no identity here"},
            topic=DELEGATE_SKILL_TOPIC,
        )

        minted = _wire_header(mock_producer, "correlation_id")
        assert minted is not None
        UUID(minted)  # raises if it is not a uuid

    @pytest.mark.asyncio
    async def test_an_identityless_publish_says_so_once_per_topic(
        self,
        bus: EventBusKafka,
        mock_producer: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """AC2 — minting is observable, and does not flood.

        A minted identity is a real finding about a producer. It must leave a
        trace naming the topic, and must not emit that trace on every message.
        """
        with caplog.at_level(
            logging.WARNING, logger="omnibase_infra.event_bus.event_bus_kafka"
        ):
            await _publish(
                bus, mock_producer, {"prompt": "one"}, topic=DELEGATE_SKILL_TOPIC
            )
            await _publish(
                bus, mock_producer, {"prompt": "two"}, topic=DELEGATE_SKILL_TOPIC
            )

        naming_the_topic = [
            record
            for record in caplog.records
            if DELEGATE_SKILL_TOPIC in record.getMessage()
            and "identity" in record.getMessage().lower()
        ]
        assert len(naming_the_topic) == 1

    @pytest.mark.asyncio
    async def test_a_malformed_body_correlation_does_not_reach_the_wire(
        self, bus: EventBusKafka, mock_producer: AsyncMock
    ) -> None:
        """AC3 — a body that lies is not trusted, and does not raise.

        Deriving identity must not become a way for a malformed payload to put
        a non-uuid into a uuid column, nor to fail a publish that would
        otherwise have succeeded.
        """
        await _publish(
            bus,
            mock_producer,
            {"prompt": "alive", "correlation_id": "not-a-uuid"},
            topic=DELEGATE_SKILL_TOPIC,
        )

        wire = _wire_header(mock_producer, "correlation_id")
        assert wire is not None
        assert wire != "not-a-uuid"
        UUID(wire)

    @pytest.mark.asyncio
    async def test_a_non_json_body_publishes_unchanged(
        self, bus: EventBusKafka, mock_producer: AsyncMock
    ) -> None:
        """AC3 — the bus carries bytes, and not every payload is a JSON object.

        Derivation is a best-effort read of a body that happens to be one. A
        binary or non-object payload must publish exactly as before.
        """
        await _publish(
            bus, mock_producer, b"\x00\x01not json at all", topic=DELEGATE_SKILL_TOPIC
        )

        assert mock_producer.send.call_args[1]["value"] == b"\x00\x01not json at all"
        minted = _wire_header(mock_producer, "correlation_id")
        assert minted is not None
        UUID(minted)

    @pytest.mark.asyncio
    async def test_caller_supplied_headers_are_never_overridden(
        self, bus: EventBusKafka, mock_producer: AsyncMock
    ) -> None:
        """The derivation is a fallback, not a policy.

        A caller that knows its own identity -- every hop downstream of the
        head -- keeps it, body or no body.
        """
        from datetime import UTC, datetime

        from omnibase_infra.event_bus.models import ModelEventHeaders

        caller_correlation = uuid4()
        headers = ModelEventHeaders(
            source="caller",
            event_type=DELEGATE_SKILL_TOPIC,
            correlation_id=caller_correlation,
            timestamp=datetime.now(UTC),
        )
        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            await bus.start()
            await bus.publish(
                DELEGATE_SKILL_TOPIC,
                None,
                json.dumps({"correlation_id": str(uuid4())}).encode("utf-8"),
                headers,
            )

        assert _wire_header(mock_producer, "correlation_id") == str(caller_correlation)
