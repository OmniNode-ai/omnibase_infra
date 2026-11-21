# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26362: a withheld projection record must never stop the event loop.

Measured on lab host h053 at 2026-11-21T16:24:30Z with every omnimarket
projection contract wired into one process (real ``EventBusKafka``, real
auto-wiring, unmaterialized Postgres): right after a projection withhold the
loop stopped running tasks and the loop-lag sampler read unresponsive, gap
growing past 55 s until the process was killed. The loop thread was inside
``_consume_loop_concurrent._drain``::

    while in_flight:
        await asyncio.gather(*list(in_flight), return_exceptions=True)

``asyncio.gather`` over children that are all done finishes eagerly, and
awaiting a finished future does not yield. A dispatch task that finished in
the same loop iteration as the driver's wake-up, ahead of it, is still in
``in_flight``: its ``in_flight.discard`` done-callback sits in the ready queue
behind the driver's own step. ``_drain`` then never yields, the callback never
runs, and the loop spins on one core forever -- the 92% CPU, silent log and
``event_loop_unresponsive`` the dev lane showed after
``dlq_unpersisted_rewind_requested`` on delegate-skill-completed offset 148150.

These tests drive the REAL ``_consume_loop`` and the REAL projection withhold
arm (``_dispatch_to_subscriber``) over the recorded coordinate. Each scenario
runs on its own loop in its own thread, so a spinning loop is observed from
outside (it cannot report itself) and stopped with an asynchronous exception.
The callback is released a varying number of loop hops after each poll, so the
tests do not depend on one exact ordering of ready callbacks.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import threading
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

import omnibase_infra.event_bus.event_bus_kafka as event_bus_kafka_module
from omnibase_infra.enums import EnumDlqFailureClass
from omnibase_infra.errors import ProjectionNotMaterializedError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_runtime_loop_lag_sampler import (
    RuntimeLoopLagSampler,
)

pytestmark = pytest.mark.unit

FIXTURE = json.loads(
    (
        Path(__file__).parent
        / "fixtures"
        / "unmaterialized_projection_bounded_offset_148150.json"
    ).read_text(encoding="utf-8")
)
TOPIC: str = FIXTURE["topic"]
GROUP: str = FIXTURE["consumer_group"]
PARTITION: int = FIXTURE["partition"]
OFFSET: int = FIXTURE["offset"]
IN_FLIGHT: int = FIXTURE["consume_concurrency"]
#: health_probe_acceptor.DEFAULT_WAIT_BUDGET_SECONDS: past this /health reads
#: ``event_loop_unresponsive``.
UNRESPONSIVE_SECONDS = 4.0
#: AC3: no handler may hold the loop for longer than this.
MAX_LOOP_HOLD_SECONDS = 1.0
#: Wall clock one scenario may take before it is judged stalled.
SCENARIO_BUDGET_SECONDS = 8.0
#: Loop hops between a poll and the release of the waiting callback.
HOPS = range(8)
BACKOFF_SECONDS = 0.02


class _RedeliveringConsumer:
    """Serves the fixture record, and serves it again after every ``seek``."""

    def __init__(self, harness: _Scenario) -> None:
        self._harness = harness
        self._position = OFFSET
        self.seeks: list[int] = []

    async def getmany(
        self, *_partitions: TopicPartition, timeout_ms: int = 0, **_kw: Any
    ) -> dict[TopicPartition, list[Any]]:
        self._harness.release_waiting_callbacks()
        if self._position > OFFSET:
            if self._harness.dlq_classes:
                # Quarantined and released: the scenario is over.
                self._harness.bus._shutdown = True
            await asyncio.sleep(0)
            return {}
        self._position = OFFSET + 1
        return {TopicPartition(TOPIC, PARTITION): [_record()]}

    def seek(self, _tp: TopicPartition, offset: int) -> None:
        self.seeks.append(offset)
        self._position = offset

    async def commit(self, _offsets: object) -> None:
        return None

    def assignment(self) -> set[TopicPartition]:
        return {TopicPartition(TOPIC, PARTITION)}

    async def stop(self) -> None:
        return None


def _record() -> MagicMock:
    msg = MagicMock()
    msg.topic = TOPIC
    msg.partition = PARTITION
    msg.offset = OFFSET
    msg.key = None
    msg.value = json.dumps({"payload": FIXTURE["payload"]}).encode()
    msg.headers = ()
    return msg


class _Scenario:
    """One consume-loop run on its own loop, watched from another thread."""

    def __init__(self, hops: int) -> None:
        self.hops = hops
        self.bus: EventBusKafka
        self.loop: asyncio.AbstractEventLoop | None = None
        self.sampler = RuntimeLoopLagSampler(sample_interval_seconds=0.01)
        self.waiters: list[asyncio.Event] = []
        self.refusals = 0
        self.dlq_classes: list[object] = []
        self.consumer: _RedeliveringConsumer | None = None
        self.finished = threading.Event()
        self.completed = False

    def release_waiting_callbacks(self) -> None:
        """Release the callbacks waiting now, ``hops`` loop hops later."""
        waiting, self.waiters = self.waiters, []
        loop = asyncio.get_running_loop()

        def hop(remaining: int) -> None:
            if remaining == 0:
                for gate in waiting:
                    gate.set()
            else:
                loop.call_soon(hop, remaining - 1)

        loop.call_soon(hop, self.hops)

    async def _callback(self, _message: object) -> None:
        gate = asyncio.Event()
        self.waiters.append(gate)
        await gate.wait()
        self.refusals += 1
        raise ProjectionNotMaterializedError(
            f"projection handler {FIXTURE['projection_type']} consumed an event "
            f"from {TOPIC} and wrote no row because its write path failed "
            "(fixture: projection not materialized)",
            projection_type=FIXTURE["projection_type"],
        )

    async def _publish_to_dlq(self, **kwargs: Any) -> bool:
        self.dlq_classes.append(kwargs.get("failure_class"))
        return True

    async def _run(self) -> None:
        self.loop = asyncio.get_running_loop()
        producer = AsyncMock()
        producer._closed = False
        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=producer,
        ):
            self.bus = EventBusKafka(
                config=ModelKafkaEventBusConfig(
                    bootstrap_servers="localhost:9092",
                    environment="dev",
                    dead_letter_topic="dlq-events",
                )
            )
            await self.bus.start()
            await self.sampler.start()
            self.consumer = _RedeliveringConsumer(self)
            consumers = cast("dict[tuple[str, str], object]", self.bus._group_consumers)
            consumers[(TOPIC, GROUP)] = self.consumer
            self.bus.declare_consume_concurrency(
                topic=TOPIC, group_id=GROUP, max_in_flight_records=IN_FLIGHT
            )
            self.bus._subscribers[TOPIC].append(
                (GROUP, "projection-savings", self._callback)
            )
            with (
                patch.object(
                    self.bus, "_publish_to_dlq", side_effect=self._publish_to_dlq
                ),
                patch.object(
                    event_bus_kafka_module,
                    "DLQ_UNPERSISTED_REWIND_BACKOFF_SECONDS",
                    BACKOFF_SECONDS,
                ),
            ):
                await self.bus._consume_loop(TOPIC, GROUP, uuid4())
            await self.bus.close()
        await self.sampler.stop()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
            self.completed = True
        finally:
            self.finished.set()

    def run(self) -> tuple[bool, float]:
        """Run the scenario; return (finished in budget, worst loop gap)."""
        thread = threading.Thread(target=self._thread_main, daemon=True)
        thread.start()
        worst = 0.0
        deadline = time.monotonic() + SCENARIO_BUDGET_SECONDS
        while not self.finished.wait(0.02) and time.monotonic() < deadline:
            worst = max(worst, self._gap())
        worst = max(worst, self._gap())
        finished = self.finished.is_set()
        if not finished and thread.ident is not None:
            # A spinning loop never yields; stop it so the session survives.
            # Cleanup paths re-enter the same drain, so keep stopping it.
            for _ in range(50):
                if not thread.is_alive():
                    break
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(thread.ident), ctypes.py_object(SystemExit)
                )
                thread.join(0.1)
        return finished, worst

    def _gap(self) -> float:
        expected = self.sampler._expected_wake
        if expected is None or self.loop is None:
            return 0.0
        return max(0.0, self.loop.time() - expected)


def _run_all() -> list[tuple[_Scenario, bool, float]]:
    results = []
    for hops in HOPS:
        scenario = _Scenario(hops)
        finished, worst = scenario.run()
        results.append((scenario, finished, worst))
        if not finished:
            break
    return results


@pytest.fixture(scope="module")
def scenarios() -> list[tuple[_Scenario, bool, float]]:
    return _run_all()


def test_unmaterialized_projection_bounded_ends_in_dlq_while_loop_runs(
    scenarios: list[tuple[_Scenario, bool, float]],
) -> None:
    """AC2: typed refusal, bounded awaited redelivery, then the DLQ."""
    bound = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:9092"
    ).projection_withhold_max_redeliveries
    for scenario, finished, _worst in scenarios:
        assert finished, (
            f"hops={scenario.hops}: the consume loop did not finish within "
            f"{SCENARIO_BUDGET_SECONDS}s after {scenario.refusals} refusals of "
            f"{TOPIC} offset {OFFSET}; the loop stopped running tasks"
        )
        assert scenario.completed, f"hops={scenario.hops}: the consume loop raised"
        assert scenario.refusals == bound + 1
        assert scenario.dlq_classes == [EnumDlqFailureClass.PROJECTION_WEDGE_EXHAUSTED]
        assert scenario.consumer is not None
        assert scenario.consumer.seeks == [OFFSET] * bound


def test_unmaterialized_projection_bounded_sampler_never_unresponsive(
    scenarios: list[tuple[_Scenario, bool, float]],
) -> None:
    """AC3: the loop-lag sampler never reads unresponsive during redelivery."""
    for scenario, _finished, worst in scenarios:
        assert worst < UNRESPONSIVE_SECONDS, (
            f"hops={scenario.hops}: loop-lag sampler read {worst:.1f}s, "
            f"unresponsive (budget {UNRESPONSIVE_SECONDS}s)"
        )
        assert worst < MAX_LOOP_HOLD_SECONDS, (
            f"hops={scenario.hops}: the loop was held {worst:.2f}s "
            f"(limit {MAX_LOOP_HOLD_SECONDS}s)"
        )
