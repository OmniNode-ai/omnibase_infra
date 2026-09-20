# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18852: a contract that declares a bound actually overlaps on the bus.

The two unit files each stop one layer short of the claim, and each stops on
the side that hides the defect:

* ``tests/unit/runtime/test_omn18852_wiring_declares_concurrency.py`` proves
  the contract key reaches ``declare_consume_concurrency`` in the right order
  -- against a recording in-memory bus, which has no consume loop, so a bound
  that is recorded and then never read would pass it;
* ``tests/unit/event_bus/test_omn18852_bounded_consume_concurrency.py`` proves
  ``EventBusKafka._consume_loop`` overlaps records -- against a bound set by
  the test itself, so a contract key that never arrives would pass it.

Serial execution passes both when the link between them is broken, which is
the exact shape of the live defect: nine delegations in 26 minutes on the
``.201`` dev lane, queue wait growing 3 s -> 445 s, zero overlapping pairs
across fifteen inferences.

This file closes that gap with one run of the real stack: a contract YAML on
disk, the real ``wire_from_manifest``, a real ``EventBusKafka``, the real
``subscribe``, and the real ``_consume_loop`` -- and it measures OVERLAP in
wall clock rather than completion, because "all N completed" is true on the
serial path.

Only the aiokafka transport is substituted. Nothing else here is a stand-in:
substituting the loop, the bus or the wiring engine would remove the seam
under test.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_ref import ModelHandlerRef
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing import (
    ModelHandlerRouting,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

SUBSCRIBE_TOPIC = "onex.cmd.omnibase-infra.omn18852-inference-request.v1"
PUBLISH_TOPIC = "onex.evt.omnibase-infra.omn18852-inference-response.v1"
PARTITION = 0
BASE_OFFSET = 4000
RECORD_COUNT = 4
DECLARED_BOUND = 4

#: Long enough that four serial handlers (>=0.60 s) cannot be mistaken for
#: four concurrent ones (~0.15 s) on a loaded CI box, short enough to keep the
#: file quick.
HANDLER_SECONDS = 0.15

_CONTRACT_BASE = f"""
name: node_omn18852_integration_effect
node_type: EFFECT_GENERIC
contract_version: 1.0.0
description: OMN-18852 end-to-end declared-concurrency fixture
event_bus:
  subscribe_topics:
    - {SUBSCRIBE_TOPIC}
  publish_topics:
    - {PUBLISH_TOPIC}
"""


class _Interval:
    """One handler invocation's wall-clock window."""

    __slots__ = ("finished", "offset", "started")

    def __init__(self, offset: int, started: float) -> None:
        self.offset = offset
        self.started = started
        self.finished = float("inf")

    def overlaps(self, other: _Interval) -> bool:
        return self.started < other.finished and other.started < self.finished


def _overlapping_pairs(intervals: list[_Interval]) -> int:
    """Pairs whose wall-clock windows intersect -- the live-lane measurement."""
    return sum(
        1
        for i, left in enumerate(intervals)
        for right in intervals[i + 1 :]
        if left.overlaps(right)
    )


class _FakeRecord:
    """The attribute surface ``_consume_loop`` reads off a Kafka record."""

    __slots__ = ("headers", "key", "offset", "partition", "topic", "value")

    def __init__(self, offset: int) -> None:
        self.topic = SUBSCRIBE_TOPIC
        self.partition = PARTITION
        self.offset = offset
        self.key = f"omn18852-{offset}".encode()
        self.value = b'{"payload": "ok"}'
        self.headers: tuple[tuple[str, bytes], ...] = ()


class _FakeConsumer:
    """Async stand-in for ``AIOKafkaConsumer``: the ONE substituted layer.

    Hands the loop every record in a single ``getmany`` batch, because a
    batch the loop has already fetched is precisely where the serial path
    serialises.
    """

    def __init__(self, offsets: list[int]) -> None:
        self._records = [_FakeRecord(offset) for offset in offsets]
        self.on_drained: Any = None
        self.stopped = False

    async def start(self) -> None:
        """Subscribe's real path starts the consumer it built."""

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[TopicPartition, list[_FakeRecord]]:
        if not self._records:
            if self.on_drained is not None:
                self.on_drained()
            return {}
        batch = self._records
        self._records = []
        return {TopicPartition(SUBSCRIBE_TOPIC, PARTITION): batch}

    def assignment(self) -> set[TopicPartition]:
        return set()

    def seek(self, topic_partition: TopicPartition, offset: int) -> None:
        raise AssertionError(
            f"no rewind is expected in this run; got seek to {offset} on "
            f"{topic_partition}"
        )

    async def stop(self) -> None:
        self.stopped = True


class _Handler:
    async def handle(self, request: object) -> object:
        return request


def _contract(tmp_path: Path, *, declaration: str) -> ModelDiscoveredContract:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_CONTRACT_BASE + declaration, encoding="utf-8")
    return ModelDiscoveredContract(
        name="node_omn18852_integration_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_path,
        entry_point_name="node_omn18852_integration_effect",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(SUBSCRIBE_TOPIC,),
            publish_topics=(PUBLISH_TOPIC,),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="_Handler", module=__name__),
                    event_model=None,
                ),
            ),
        ),
    )


async def _run_declared(
    tmp_path: Path, declaration: str
) -> tuple[list[_Interval], float]:
    """Wire a contract through the real stack and drive its real consume loop.

    Returns the handler windows the run produced and its wall-clock duration.
    """
    intervals: list[_Interval] = []

    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    producer._closed = False

    consumer = _FakeConsumer([BASE_OFFSET + i for i in range(RECORD_COUNT)])

    async def _dispatch(
        _callback: Any,
        _subscription_id: str,
        _event_message: Any,
        _topic: str,
        _group_id: str,
        _correlation_id: Any,
        *,
        record_coordinate: Any = None,
    ) -> bool:
        assert record_coordinate is not None, (
            "the loop must pass the record coordinate through to dispatch"
        )
        interval = _Interval(int(record_coordinate[1]), time.monotonic())
        intervals.append(interval)
        try:
            await asyncio.sleep(HANDLER_SECONDS)
            return True
        finally:
            interval.finished = time.monotonic()

    config = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:9092",
        environment="dev",
        dead_letter_topic="dlq-events",
    )

    with (
        patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=producer,
        ),
        patch.object(EventBusKafka, "_build_consumer", return_value=consumer),
    ):
        bus = EventBusKafka(config=config)
        await bus.start()

        # The consume loop subscribe spawns exits as soon as the fake batch is
        # drained; flipping the shutdown flag is how the fixture ends the run.
        consumer.on_drained = lambda: setattr(bus, "_shutdown", True)

        contract = _contract(tmp_path, declaration=declaration)
        engine = MessageDispatchEngine()
        started = time.monotonic()
        with (
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring."
                "_import_handler_class",
                return_value=_Handler,
            ),
            patch.object(bus, "_dispatch_to_subscriber", side_effect=_dispatch),
        ):
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(contract,)),
                engine,
                event_bus=bus,
                environment="local",
            )
            # subscribe() started the real loop as a task; let it run the
            # batch out rather than driving the loop by hand.
            await asyncio.wait_for(
                _drain(expected=RECORD_COUNT, intervals=intervals),
                timeout=30.0,
            )
        elapsed = time.monotonic() - started

        await bus.close()

    return intervals, elapsed


async def _drain(*, expected: int, intervals: list[_Interval]) -> None:
    """Wait until every record's handler window has opened and closed."""
    while len(intervals) < expected or any(
        interval.finished == float("inf") for interval in intervals
    ):
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_declared_contract_overlaps_records_on_a_real_bus(
    tmp_path: Path,
) -> None:
    """The claim, end to end: declaring the key makes the lane concurrent.

    Asserted as OVERLAP, not completion. Four records that each complete are
    what the serial lane already did.
    """
    intervals, elapsed = await _run_declared(
        tmp_path,
        f"consume_concurrency:\n  max_in_flight_records: {DECLARED_BOUND}\n",
    )

    assert len(intervals) == RECORD_COUNT, (
        f"every record must reach a handler; got {len(intervals)}"
    )
    assert _overlapping_pairs(intervals) > 0, (
        "a contract declaring a bound of "
        f"{DECLARED_BOUND} produced zero overlapping handler windows -- the "
        "declaration reached the bus but the lane still ran serial, which is "
        "the OMN-18852 defect with a configuration key on top of it"
    )
    serial_floor = RECORD_COUNT * HANDLER_SECONDS
    assert elapsed < serial_floor, (
        f"the run took {elapsed:.3f}s, at or past the {serial_floor:.3f}s a "
        "fully serial lane would take"
    )


@pytest.mark.asyncio
async def test_an_undeclared_contract_stays_serial_on_a_real_bus(
    tmp_path: Path,
) -> None:
    """The default is byte-for-byte today's behaviour, proven not assumed.

    The companion to the test above: if wiring made every lane concurrent,
    the first test would pass while every existing node silently lost its
    partition ordering.
    """
    intervals, _elapsed = await _run_declared(tmp_path, "")

    assert len(intervals) == RECORD_COUNT
    assert _overlapping_pairs(intervals) == 0, (
        "a contract that declares nothing must keep the inline serial path; "
        f"got {_overlapping_pairs(intervals)} overlapping handler windows"
    )
    assert [interval.offset for interval in intervals] == [
        BASE_OFFSET + i for i in range(RECORD_COUNT)
    ], "the serial path must preserve partition order"
