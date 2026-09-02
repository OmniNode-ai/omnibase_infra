# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failure-path publish must not open the connection-wide circuit breaker.

OMN-17497. The Kafka bus keys its circuit breaker on the CONNECTION
(``kafka.<environment>``), so any failure charged to it is a verdict about the
whole broker connection and blocks every publisher on the instance while it is
open. On ``onex-dev`` that let one poison event on one low-value UI topic open
the breaker 108 times in two hours and take the gateway attach/heartbeat/detach
path down with it:

    HandlerRendererCapabilityProjection raises on every event
      -> no dlq_topics declared, so the wiring falls back to the platform
         quarantine sink onex.dlq.omnibase-infra.quarantine.v1
      -> that publish fails [Error 45] OutOfOrderSequenceNumber, 4/4 attempts
      -> the shared kafka.onex-dev breaker opens

Two independent cuts, either of which defuses the chain, are asserted here:

1. A publish to a DLQ/quarantine sink never records a circuit failure. It is
   already an error path; it does not get to vote on connection health.
2. The idempotent-producer fatal family (``OutOfOrderSequenceNumber`` and
   friends) discards the producer so the next attempt mints a fresh producer
   id, instead of burning four retries against permanently-broken local
   sequence state -- and is not counted as broker unavailability either.

Fixture pattern mirrors tests/unit/event_bus/test_circuit_breaker_topic_error.py
(the OMN-9553 exemption), so no real broker is needed.

Related Tickets:
    - OMN-17497: this ticket
    - OMN-16690: the access-class defect that produced the poison event
    - OMN-15957: the gateway session tests the flap made untrustworthy
    - OMN-9553 / OMN-16267: the two pre-existing breaker exemptions
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from aiokafka.errors import (
    InvalidProducerEpoch,
    KafkaError,
    OutOfOrderSequenceNumber,
    UnknownProducerId,
)

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.event_bus.event_bus_kafka import (
    _IDEMPOTENT_PRODUCER_FATAL_ERRORS,
    EventBusKafka,
)
from omnibase_infra.event_bus.models import ModelEventHeaders
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.topic_constants import is_dlq_topic

TEST_SERVERS: str = "localhost:9092"
TEST_ENV: str = "test"

QUARANTINE_TOPIC: str = "onex.dlq.omnibase-infra.quarantine.v1"
GATEWAY_TOPIC: str = "onex.evt.omnibase-infra.gateway-attached.v1"


def _headers() -> ModelEventHeaders:
    return ModelEventHeaders(
        correlation_id=uuid4(),
        source="test",
        event_type="test.event",
        timestamp=datetime.now(UTC),
    )


def _completed_send() -> asyncio.Future[SimpleNamespace]:
    future: asyncio.Future[SimpleNamespace] = asyncio.Future()
    future.set_result(SimpleNamespace(partition=0, offset=1))
    return future


@pytest.fixture
async def bus() -> AsyncGenerator[EventBusKafka, None]:
    """EventBusKafka with a mocked AIOKafkaProducer -- no real broker needed."""
    mock_producer = AsyncMock()
    mock_producer.start = AsyncMock()
    mock_producer.stop = AsyncMock()
    mock_producer._closed = False
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ):
        config = ModelKafkaEventBusConfig(
            bootstrap_servers=TEST_SERVERS,
            environment=TEST_ENV,
        )
        yield EventBusKafka(config=config)


@pytest.mark.unit
class TestDlqPublishDoesNotOpenSharedBreaker:
    """A DLQ/quarantine publish is an error path, not a health signal."""

    def test_quarantine_sink_is_recognised_as_a_dlq_topic(self) -> None:
        """The live quarantine sink must classify, or the exemption is dead code."""
        assert is_dlq_topic(QUARANTINE_TOPIC) is True
        assert is_dlq_topic(GATEWAY_TOPIC) is False

    def test_non_infra_producer_dlq_topics_classify(self) -> None:
        """OMN-17497: the producer segment is not an allowlist of one repo.

        Before this ticket ``DLQ_TOPIC_PATTERN`` hardcoded ``omnibase-infra``
        as the only producer, so every ``onex.dlq.omnimarket.*`` sink -- 15+
        of which are declared in live omnimarket contracts -- answered False
        and would still have been able to open the shared breaker.
        """
        assert is_dlq_topic("onex.dlq.omnimarket.adversarial-pipeline.v1") is True
        assert (
            is_dlq_topic("onex.dlq.omnimarket.renderer-capability-malformed.v1") is True
        )
        # Negative controls preserved.
        assert is_dlq_topic("onex.dlq.123invalid.v1") is False
        assert is_dlq_topic("onex.evt.platform.events.v1") is False
        assert is_dlq_topic("dlq.intents.v1") is False

    @pytest.mark.asyncio
    async def test_quarantine_publish_failure_does_not_charge_the_breaker(
        self, bus: EventBusKafka
    ) -> None:
        """The exact live chain: quarantine publish fails, breaker stays shut."""
        before = bus._circuit_breaker_failures
        bus._producer = AsyncMock()
        bus._producer.send = AsyncMock(side_effect=KafkaError("broker said no"))

        with pytest.raises(InfraConnectionError):
            await bus._publish_with_retry(
                topic=QUARANTINE_TOPIC,
                key=None,
                value=b"dlq-envelope",
                kafka_headers=[],
                headers=_headers(),
            )

        assert bus._circuit_breaker_failures == before, (
            "a failed publish to the quarantine sink must not be charged to the "
            "connection-wide circuit breaker"
        )
        assert not bus._circuit_breaker_open

    @pytest.mark.asyncio
    async def test_quarantine_publish_failure_at_threshold_keeps_breaker_closed(
        self, bus: EventBusKafka
    ) -> None:
        """One failure short of the threshold is where the live flap happened."""
        bus._circuit_breaker_failures = bus.circuit_breaker_threshold - 1
        before = bus._circuit_breaker_failures
        bus._producer = AsyncMock()
        bus._producer.send = AsyncMock(side_effect=KafkaError("broker said no"))

        with pytest.raises(InfraConnectionError):
            await bus._publish_with_retry(
                topic=QUARANTINE_TOPIC,
                key=None,
                value=b"dlq-envelope",
                kafka_headers=[],
                headers=_headers(),
            )

        assert bus._circuit_breaker_failures == before
        assert not bus._circuit_breaker_open, (
            "the quarantine sink must never be able to push the shared breaker "
            "over its threshold -- that is the OMN-17497 blast radius"
        )

    @pytest.mark.asyncio
    async def test_primary_path_failure_still_charges_the_breaker(
        self, bus: EventBusKafka
    ) -> None:
        """Control: the exemption is scoped to DLQ sinks, not a global disable.

        Without this the fix would read as "the breaker no longer works".
        """
        before = bus._circuit_breaker_failures
        bus._producer = AsyncMock()
        bus._producer.send = AsyncMock(side_effect=KafkaError("broker said no"))

        with pytest.raises(InfraConnectionError):
            await bus._publish_with_retry(
                topic=GATEWAY_TOPIC,
                key=None,
                value=b"gateway-envelope",
                kafka_headers=[],
                headers=_headers(),
            )

        assert bus._circuit_breaker_failures > before, (
            "a real publish failure on a primary topic must still be charged "
            "to the breaker"
        )


@pytest.mark.unit
class TestIdempotentProducerFatalStateResetsTheProducer:
    """OutOfOrderSequenceNumber is producer-local state, not broker health."""

    def test_family_members_are_kafka_error_subclasses(self) -> None:
        """Document the IS-A that makes the except-ordering load-bearing."""
        assert (
            OutOfOrderSequenceNumber,
            UnknownProducerId,
            InvalidProducerEpoch,
        ) == _IDEMPOTENT_PRODUCER_FATAL_ERRORS
        for exc_type in _IDEMPOTENT_PRODUCER_FATAL_ERRORS:
            assert issubclass(exc_type, KafkaError), (
                f"{exc_type.__name__} must remain a KafkaError subclass -- its "
                "handler MUST precede the generic `except KafkaError` block"
            )

    @pytest.mark.asyncio
    async def test_out_of_order_sequence_discards_the_producer(
        self, bus: EventBusKafka
    ) -> None:
        """A broken idempotent producer is dropped, not retried into the ground."""
        before = bus._circuit_breaker_failures
        broken = AsyncMock()
        broken.stop = AsyncMock()
        broken.send = AsyncMock(side_effect=OutOfOrderSequenceNumber())
        bus._producer = broken

        with pytest.raises(InfraConnectionError):
            await bus._publish_with_retry(
                topic=QUARANTINE_TOPIC,
                key=None,
                value=b"dlq-envelope",
                kafka_headers=[],
                headers=_headers(),
            )

        assert bus._producer is None, (
            "the producer whose sequence state the broker rejected must be "
            "discarded so the next attempt mints a fresh producer id"
        )
        broken.stop.assert_awaited()
        assert bus._circuit_breaker_failures == before, (
            "an idempotent-producer sequence rejection is not evidence that the "
            "broker connection is unavailable"
        )

    @pytest.mark.asyncio
    async def test_retry_after_reset_succeeds_on_a_fresh_producer(self) -> None:
        """The retry budget is spent on something that can actually work.

        Attempt 1 fails OutOfOrderSequenceNumber on the original producer;
        the branch discards it; ``_ensure_producer`` mints a replacement and
        attempt 2 lands. On the pre-OMN-17497 code all four attempts reused
        the same permanently-broken producer and the publish could not
        succeed at all.
        """
        broken = AsyncMock()
        broken.start = AsyncMock()
        broken.stop = AsyncMock()
        broken.send = AsyncMock(side_effect=OutOfOrderSequenceNumber())

        healthy = AsyncMock()
        healthy.start = AsyncMock()
        healthy.stop = AsyncMock()
        healthy.send = AsyncMock(return_value=_completed_send())

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            side_effect=[healthy],
        ):
            config = ModelKafkaEventBusConfig(
                bootstrap_servers=TEST_SERVERS,
                environment=TEST_ENV,
            )
            bus = EventBusKafka(config=config)
            bus._started = True
            bus._producer = broken
            before = bus._circuit_breaker_failures

            receipt = await bus._publish_with_retry(
                topic=QUARANTINE_TOPIC,
                key=None,
                value=b"dlq-envelope",
                kafka_headers=[],
                headers=_headers(),
            )

        assert receipt.topic == QUARANTINE_TOPIC
        assert bus._producer is healthy, (
            "attempt 2 must run on the freshly minted producer, not the one "
            "the broker refused"
        )
        healthy.send.assert_awaited_once()
        assert bus._circuit_breaker_failures == before
