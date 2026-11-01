# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19241 -- the kernel's ONE dependency mapping, driven by concurrent runs.

The unit tests in ``tests/unit/nodes/node_dlq_replay_effect`` prove the lease
with doubles standing in for the engine classes. This file proves it on the
shape the runtime actually builds: ``_build_runtime_handler_dependencies``
supplies one consumer per declared topic and ONE replay producer and ONE
quarantine producer, and every per-topic dispatcher constructs its
``HandlerDlqReplay`` from that same mapping. Only aiokafka is replaced; the real
``DLQConsumer``, ``DLQProducer`` and ``DLQQuarantineProducer`` run their own
``start``/``stop``/``_started`` logic.

The interleaving is the one measured on the .201 dev lane 2026-09-23: one run
drains the commands DLQ, quarantining record after record, while a peer run on
the events DLQ finishes. At the parent commit the peer's teardown stopped the
shared quarantine producer and the commands run's next quarantine raised
``Quarantine producer not started``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.nodes.node_dlq_replay_effect import engine_dlq_replay
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

pytestmark = pytest.mark.integration

_EVENTS = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: a declared subscribe topic of this node
_COMMANDS = "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow: a declared subscribe topic of this node
_SOURCE = "onex.evt.omnimarket.prod-promotion-gate-evaluated.v1"  # onex-topic-allow: verbatim from the live storm
_RECORDS = 40
_TIMEOUT = 10.0
_LIFECYCLE_IO_SECONDS = 0.01
_FETCH_SECONDS = 0.002


class _Broker:
    """What the fake clients read from and write to, per test."""

    def __init__(self) -> None:
        self.records: dict[str, list[bytes]] = {}
        self.gates: dict[str, asyncio.Event] = {}
        self.entered: dict[str, asyncio.Event] = {}
        self.published: list[str] = []
        self.committed: dict[str, int] = {}


def _dead_letter(offset: int) -> bytes:
    return json.dumps(
        {
            "original_topic": _SOURCE,
            "original_message": {
                "key": "k",
                "value": json.dumps({"n": offset}),
                "offset": offset,
                "partition": 0,
            },
            "failure_reason": "handler dispatch failed",
            "failure_timestamp": "2026-09-23T10:32:40Z",
            "correlation_id": str(uuid4()),
            "retry_count": 9,  # past max_replay_count: must be quarantined
            "error_type": "InfraConnectionError",
        }
    ).encode()


def _fake_kafka(broker: _Broker) -> tuple[type, type]:
    class FakeProducer:
        def __init__(self, **_: Any) -> None:
            pass

        # A real start/stop is network I/O, so the loop runs other tasks while
        # it is in flight; a fake that never suspends hides every window.
        async def start(self) -> None:
            await asyncio.sleep(_LIFECYCLE_IO_SECONDS)

        async def stop(self) -> None:
            await asyncio.sleep(_LIFECYCLE_IO_SECONDS)

        async def send_and_wait(self, topic: str, **_: Any) -> object:
            broker.published.append(topic)
            return object()

    class FakeConsumer:
        def __init__(self, topic: str, **_: Any) -> None:
            self._topic = topic
            self._records: Iterator[tuple[int, bytes]] = iter(())

        async def start(self) -> None:
            await asyncio.sleep(_LIFECYCLE_IO_SECONDS)
            # A restarted consumer resumes from its group's committed offset.
            start = broker.committed.get(self._topic, 0)
            records = broker.records.get(self._topic, [])
            self._records = iter(list(enumerate(records))[start:])

        async def stop(self) -> None:
            return None

        async def commit(self, offsets: dict[Any, int] | None = None) -> None:
            for partition, next_offset in (offsets or {}).items():
                broker.committed[partition.topic] = next_offset

        def __aiter__(self) -> FakeConsumer:
            return self

        async def __anext__(self) -> SimpleNamespace:
            entered = broker.entered.get(self._topic)
            if entered is not None and not entered.is_set():
                entered.set()
                await broker.gates[self._topic].wait()
            # A real fetch takes time too, so the drain is still running when
            # a peer's teardown lands.
            await asyncio.sleep(_FETCH_SECONDS)
            try:
                offset, value = next(self._records)
            except StopIteration:
                raise StopAsyncIteration from None
            return SimpleNamespace(value=value, partition=0, offset=offset)

    return FakeProducer, FakeConsumer


@pytest.mark.asyncio
async def test_kernel_wired_dispatchers_never_stop_a_producer_a_peer_is_using(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _Broker()
    broker.records[_COMMANDS] = [_dead_letter(i) for i in range(_RECORDS)]
    for topic in (_EVENTS, _COMMANDS):
        broker.gates[topic] = asyncio.Event()
        broker.entered[topic] = asyncio.Event()
    fake_producer, fake_consumer = _fake_kafka(broker)
    monkeypatch.setattr(engine_dlq_replay, "AIOKafkaProducer", fake_producer)
    monkeypatch.setattr(engine_dlq_replay, "AIOKafkaConsumer", fake_consumer)

    dependencies = _build_runtime_handler_dependencies(None, "localhost:9092")
    assert dependencies is not None
    mapping = dependencies["HandlerDlqReplay"]
    # OMN-19085: the kernel also wires a backlog probe, which would read the
    # (fake) broker's committed offsets before each run. This test is about
    # producer leases across two concurrent drains, so both runs must drain;
    # the probe has its own tests in
    # tests/unit/nodes/node_dlq_replay_effect/test_omn19085_*.
    assert "backlog_probe" in mapping
    mapping = {key: value for key, value in mapping.items() if key != "backlog_probe"}
    events_run = HandlerDlqReplay(**mapping)
    commands_run = HandlerDlqReplay(**mapping)
    # Dispatchers rotate independently; on the lane the two were draining
    # different topics at once. Start this one on the commands DLQ.
    commands_run._start_index = list(mapping["consumers"]).index(_COMMANDS)

    first = asyncio.create_task(events_run.run())
    await asyncio.wait_for(broker.entered[_EVENTS].wait(), _TIMEOUT)
    second = asyncio.create_task(commands_run.run())
    await asyncio.wait_for(broker.entered[_COMMANDS].wait(), _TIMEOUT)

    # The commands run starts quarantining; the events run then ends its
    # drain underneath it.
    broker.gates[_COMMANDS].set()
    await asyncio.sleep(0)
    broker.gates[_EVENTS].set()
    results = await asyncio.wait_for(asyncio.gather(first, second), _TIMEOUT)

    failed = [
        r.message
        for result in results
        for r in result.results
        if r.status == EnumReplayStatus.FAILED
    ]
    assert failed == [], f"{len(failed)} quarantine(s) failed: {failed[:3]}"
    assert sum(result.quarantined for result in results) == _RECORDS
    assert len(broker.published) == _RECORDS
    assert not mapping["producer"]._started
    assert not mapping["quarantine_producer"]._started
