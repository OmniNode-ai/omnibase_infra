# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18084 — concurrent dispatchers must not stop each other's DLQ consumer.

WHY THREE DISPATCHERS SHARE ONE CONSUMER. ``service_kernel`` keys runtime
dependencies by handler NAME, so the three per-topic dispatcher entries
OMN-18013 split ``node_dlq_replay_effect``'s routing into all resolve to the
same ``dependencies["HandlerDlqReplay"]`` mapping — one ``DLQConsumer`` object
behind three ``HandlerDlqReplay`` instances.

THE RACE. ``_ensure_runtime_dependencies_started`` skips a dependency already
flagged ``_started`` and therefore omits it from the list its own ``finally``
stops, while the peer that DID start it stops it unconditionally. That is
check-then-use across an await boundary: between one dispatcher deciding not to
start the consumer and its drain generator actually executing, the peer's
``finally`` can run and clear ``_started``. The first line the victim's
generator then executes is the assertion that raises
``RuntimeError("Consumer not started")``.

WHY IT MATTERS BEYOND THIS NODE. That exception is what the auto-wiring boundary
answered by writing the record back onto the DLQ topic it came from — 193.8
records/s, ~151 GB/day, on a mount with 590 GB free shared with prod,
stability-test and judge. The loop-breaker stops the amplification; this stops
the failures being generated at all.

HOW THE INTERLEAVING IS MADE DETERMINISTIC. The double below subclasses the real
``DLQConsumer`` and overrides only the three methods that touch Kafka, so the
``_started`` flag it moves is the production one. Each drain waits on a gate the
test releases. That gate is a SCHEDULING hook, not a semantic change: it
controls *when* the generator body runs, which on the lane is whenever the event
loop gets round to the task ``asyncio.wait_for(anext(...))`` scheduled. The
``_started`` assertion still runs first thing in that body, exactly as it does
in production.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, cast

import pytest

from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    DLQProducer,
    DLQQuarantineProducer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_unparseable_dlq_record import (
    DlqDrainRecord,
)

if TYPE_CHECKING:
    from aiokafka import AIOKafkaConsumer

_DLQ_EVENTS_TOPIC = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: verbatim from the live amplification trace


class _SharedConsumerDouble(DLQConsumer):
    """The real ``DLQConsumer`` lifecycle with the Kafka client removed."""

    def __init__(self, config: ModelDlqReplayEngineConfig, *, gates: int = 4) -> None:
        super().__init__(config)
        self.drain_calls = 0
        self.not_started_errors = 0
        self.start_calls = 0
        self.stop_calls = 0
        # One entered/release pair per drain, indexed by drain order.
        self.drain_entered = [asyncio.Event() for _ in range(gates)]
        self.drain_release = [asyncio.Event() for _ in range(gates)]

    async def start(self) -> None:
        self.start_calls += 1
        # The production ``start`` assigns a live AIOKafkaConsumer and the drain
        # assertion reads that field, so the double has to move it too.
        self._consumer = cast("AIOKafkaConsumer", object())
        self._started = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self._started = False
        self._consumer = None

    async def commit_offsets(self, offsets: Mapping[tuple[str, int], int]) -> None:
        return None

    async def consume_messages(self) -> AsyncIterator[DlqDrainRecord]:
        index = self.drain_calls
        self.drain_calls += 1
        if index < len(self.drain_entered):
            self.drain_entered[index].set()
            await self.drain_release[index].wait()
        if not self._started or self._consumer is None:
            self.not_started_errors += 1
            raise RuntimeError("Consumer not started")
        return
        yield  # pragma: no cover - makes this an async generator


class _ProducerDouble:
    """Start/stop-shaped double for the two producers the handler also owns."""

    def __init__(self) -> None:
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False


def _consumer(gates: int = 4) -> _SharedConsumerDouble:
    config = ModelDlqReplayEngineConfig(
        bootstrap_servers="test-broker:9092",
        dlq_topic=_DLQ_EVENTS_TOPIC,
    )
    return _SharedConsumerDouble(config, gates=gates)


def _handler(
    consumer: _SharedConsumerDouble,
    producer: _ProducerDouble,
    quarantine: _ProducerDouble,
) -> HandlerDlqReplay:
    return HandlerDlqReplay(
        consumers={consumer.config.dlq_topic: consumer},
        producer=cast("DLQProducer", producer),
        quarantine_producer=cast("DLQQuarantineProducer", quarantine),
    )


@pytest.mark.unit
class TestSharedDlqConsumerLifecycle:
    """AC3 — ``Consumer not started`` cannot arise from concurrent start/stop."""

    @pytest.mark.asyncio
    async def test_a_peer_dispatcher_cannot_stop_a_consumer_a_peer_is_using(
        self,
    ) -> None:
        """RED at the parent commit: the second dispatcher raises.

        At ``a4393794`` the second handler skips starting (the flag is already
        set by the first), the first handler's ``finally`` clears the flag, and
        the second handler's drain raises ``RuntimeError("Consumer not
        started")`` — the exact error carried by all 2,000 sampled DLQ records.
        """
        consumer = _consumer()
        # The producers are shared too: one dependencies mapping, three handlers.
        producer = _ProducerDouble()
        quarantine = _ProducerDouble()

        first = asyncio.create_task(_handler(consumer, producer, quarantine).run())
        await asyncio.wait_for(consumer.drain_entered[0].wait(), timeout=5)

        second = asyncio.create_task(_handler(consumer, producer, quarantine).run())
        await asyncio.sleep(0)

        consumer.drain_release[0].set()
        await asyncio.wait_for(first, timeout=5)

        # The first dispatcher has now run its ``finally``. Let the second one's
        # drain body execute — on the lane this is simply the event loop getting
        # to it after the peer's teardown.
        consumer.drain_release[1].set()
        outcome = await asyncio.wait_for(
            asyncio.gather(second, return_exceptions=True), timeout=5
        )

        raised = [r for r in outcome if isinstance(r, BaseException)]
        assert raised == [], (
            "a dispatcher sharing the consumer failed while a peer tore it down: "
            f"{[f'{type(r).__name__}: {r}' for r in raised]}"
        )
        assert consumer.not_started_errors == 0, (
            "the shared DLQ consumer was stopped out from under a peer "
            f"{consumer.not_started_errors} time(s)"
        )

    @pytest.mark.asyncio
    async def test_three_dispatchers_over_one_consumer_all_complete(self) -> None:
        """The live shape: OMN-18013 created three per-topic dispatcher entries.

        Every one of them must be able to drain, in any interleaving, without a
        peer's teardown reaching into its run. Passes at the parent commit only
        because nothing forces an unfavourable interleaving here — it is the
        no-regression control for the serialisation the fix introduces, not a
        second demonstration of the defect.
        """
        consumer = _consumer()
        producer = _ProducerDouble()
        quarantine = _ProducerDouble()
        for release in consumer.drain_release:
            release.set()

        outcome = await asyncio.wait_for(
            asyncio.gather(
                *(_handler(consumer, producer, quarantine).run() for _ in range(3)),
                return_exceptions=True,
            ),
            timeout=10,
        )

        raised = [r for r in outcome if isinstance(r, BaseException)]
        assert raised == [], (
            "three concurrent dispatchers over one shared consumer produced: "
            f"{[f'{type(r).__name__}: {r}' for r in raised]}"
        )
        assert consumer.not_started_errors == 0

    @pytest.mark.asyncio
    async def test_every_started_consumer_is_stopped_exactly_once(self) -> None:
        """Serialising must not leak a started consumer, which would be worse.

        A run that returns without stopping what it started holds a broker
        connection and a group membership open; the group then stalls at
        ``max_poll_interval_ms`` rather than failing loudly.
        """
        consumer = _consumer()
        producer = _ProducerDouble()
        quarantine = _ProducerDouble()
        for release in consumer.drain_release:
            release.set()

        await asyncio.wait_for(
            asyncio.gather(
                *(_handler(consumer, producer, quarantine).run() for _ in range(3))
            ),
            timeout=10,
        )

        assert consumer.start_calls == consumer.stop_calls, (
            f"consumer started {consumer.start_calls} time(s) and stopped "
            f"{consumer.stop_calls} time(s) — a leaked or double teardown"
        )
        assert consumer._started is False
