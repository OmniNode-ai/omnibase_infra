# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An ACL refusal traverses the whole publish path attributably (OMN-18627).

The unit tests beside this one pin the branch. What they cannot show is the
property the defect actually turned on: the refusal has to survive the ENTIRE
`publish()` path -- retry loop, terminal raise, and the caller's own
`asyncio.wait_for` -- still wearing a type that says "never". Before this
change it did not. The retry ladder's exponential backoff outlasted the
caller's publish timeout, `wait_for` cancelled it, and what reached the caller
was `TimeoutError`: a permanent refusal wearing the one shape that means "try
again shortly".

A spool drain reading that signal re-queued the same unpublishable record every
cycle for 20 hours and held 126 records of four AUTHORIZED classes behind eight
of its own.

Mocked producer, live code path -- the same shape as
`test_kafka_invalid_partitions_retry_integration.py` beside it. What is
integration-level here is that the assertions are made at `publish()`, the
boundary a caller actually holds, rather than inside the retry helper.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiokafka.errors import KafkaError, TopicAuthorizationFailedError

from omnibase_infra.errors import EventTopicAuthorizationError, InfraConnectionError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

pytestmark = pytest.mark.integration

_UNGRANTED = "onex.evt.omniclaude.omn18627-ungranted.v1"
_HEALTHY = "onex.evt.omniclaude.omn18627-healthy.v1"


def _config(**overrides: object) -> ModelKafkaEventBusConfig:
    fields: dict[str, object] = {
        "bootstrap_servers": "localhost:19092",
        "environment": "test",
        "timeout_seconds": 10,
        "max_retry_attempts": 3,
        # 0.5 is load-bearing in the first test: the OLD four-attempt ladder
        # sleeps 0.5 + 1.0 + 2.0 before jitter, which is why a 2s caller
        # timeout cancelled it and delivered TimeoutError.
        "retry_backoff_base": 0.5,
    }
    fields.update(overrides)
    return ModelKafkaEventBusConfig(**fields)


def _producer() -> AsyncMock:
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer._closed = False
    return producer


def _accepted() -> asyncio.Future[MagicMock]:
    """A send() result the bus can await, as aiokafka returns."""
    future: asyncio.Future[MagicMock] = asyncio.get_running_loop().create_future()
    meta = MagicMock()
    meta.partition = 0
    meta.offset = 0
    future.set_result(meta)
    return future


@pytest.mark.asyncio
async def test_refusal_survives_a_caller_timeout_wrapper_as_itself() -> None:
    """Under the caller's own `wait_for`, the caller still sees "never".

    This is the regression in one assertion. With the old four-attempt ladder
    at this backoff base (0.5s: 0.5 + 1.0 + 2.0 = 3.5s of sleeps alone, before
    jitter), a 2-second caller timeout cancelled the ladder and delivered
    `TimeoutError`. The refusal now short-circuits, so it arrives as itself.
    """
    producer = _producer()
    producer.send = AsyncMock(side_effect=TopicAuthorizationFailedError(_UNGRANTED))

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        bus = EventBusKafka(config=_config())
        await bus.start()
        try:
            started = time.monotonic()
            with pytest.raises(EventTopicAuthorizationError) as excinfo:
                await asyncio.wait_for(bus.publish(_UNGRANTED, b"k", b"v"), timeout=2.0)
            elapsed = time.monotonic() - started
        finally:
            await bus.close()

    assert excinfo.value.topic == _UNGRANTED
    assert producer.send.call_count == 1
    # The bound, not a stopwatch assertion: the OLD path could not finish
    # inside the caller's 2s window at all, so any completion well under it
    # proves the backoff sleeps are gone.
    assert elapsed < 1.0, (
        f"the refusal took {elapsed:.2f}s; it must not spend the retry "
        "budget's backoff sleeps on a verdict that cannot change"
    )


@pytest.mark.asyncio
async def test_an_ungranted_topic_does_not_take_healthy_topics_down_with_it() -> None:
    """The breaker stays closed, so the next topic on the SAME bus publishes.

    The circuit breaker is shared across topics. Counting an ACL refusal as a
    connection failure would let one ungranted topic open it and refuse
    healthy topics -- exactly how a single poison event opened a shared
    breaker 108 times in two hours under OMN-17497, from a different cause.
    """
    producer = _producer()

    async def send(
        topic: str, *args: object, **kwargs: object
    ) -> asyncio.Future[MagicMock]:
        if topic == _UNGRANTED:
            raise TopicAuthorizationFailedError(topic)
        return _accepted()

    producer.send = AsyncMock(side_effect=send)

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        bus = EventBusKafka(config=_config(circuit_breaker_threshold=1))
        await bus.start()
        try:
            for _ in range(3):
                with pytest.raises(EventTopicAuthorizationError):
                    await bus.publish(_UNGRANTED, b"k", b"v")
            # A breaker threshold of 1 means a SINGLE counted failure would
            # already have opened it; three refusals is the margin.
            await bus.publish(_HEALTHY, b"k", b"v")
        finally:
            await bus.close()


@pytest.mark.asyncio
async def test_a_transient_broker_failure_still_retries_and_still_trips() -> None:
    """Positive control: only the ACL arm changed.

    Without it, the two tests above are also satisfied by a bus that stopped
    retrying and stopped counting anything -- which would convert every
    transient blip into a permanent failure and disable the breaker.
    """
    producer = _producer()
    producer.send = AsyncMock(side_effect=KafkaError("leader not available"))

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        bus = EventBusKafka(config=_config(retry_backoff_base=0.001))
        await bus.start()
        try:
            with pytest.raises(InfraConnectionError):
                await bus.publish(_HEALTHY, b"k", b"v")
            assert producer.send.call_count == 4
            assert bus._circuit_breaker_failures > 0
        finally:
            await bus.close()
