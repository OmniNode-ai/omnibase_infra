# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""K>=10 multi-cell regression for the terminal-consumer SUBSCRIPTION-flip wedge.

Ticket: OMN-13128 (OMN-13118 strike-5 follow-up).

What this test models that the #1970/#1971 repros did NOT
--------------------------------------------------------
``test_terminal_consumer_battery_load_wedge.py`` (OMN-13118 / #1970) models the
FAILED-topic *assign-cap* stall, and ``test_..._seek_reset_race`` (#1971) models
the lazy ``seek_to_end`` *offset* race. Both prior fixes went RED->GREEN offline
yet the LIVE K>=10 multi-cell battery on the stability lane WEDGED every time
(HALT_K10_WEDGE_PERSISTS, strikes 3 + 4). The reason the offline repros kept
lying: their fakes collapsed the COMPLETED and FAILED waits onto behaviour that a
SINGLE consumer could satisfy, so they never exercised the actual live fault —
the runtime waited for each trial's terminal across BOTH topics
(``node-generation-completed.v1`` + ``node-generation-failed.v1``) using ONE
ephemeral ``group_id=None`` consumer assigned the partitions of both topics, and a
single aiokafka consumer cannot deliver from two topics at once: its delivery
window flips between them and the COMPLETED record is torn down before it is read
(the live ``Updating subscribed topics`` flip the HALT log captured).

This test models that fault HONESTLY: the fake consumer's delivery is faithful
ONLY when its assignment is confined to a SINGLE topic. The moment ONE consumer is
assigned partitions spanning BOTH terminal topics, the fake flips its readable
topic every poll and DROPS any record sitting on the non-current topic — exactly
the live subscription-flip that loses the already-published COMPLETED terminal.

  - RED (single-consumer-both-topics, the pre-OMN-13128 shape): a consumer
    assigned both topics flips, the COMPLETED terminal is dropped, the trial never
    correlates, and the K>=10 x 2-cell x 2-arm matrix produces degenerate rows /
    times out — the live wedge.
  - GREEN (OMN-13128, one INDEPENDENT consumer per terminal topic): each consumer
    holds exactly one topic, never flips, and the COMPLETED consumer surfaces the
    correlated record on its own subscription while the partition-less FAILED
    consumer sleeps out its timeout — every cell across both arms correlates.

RED genuineness: revert ONLY the source change in
``RuntimePatternBBroker._dispatch_and_wait_with_direct_kafka_consumer`` (back to a
single consumer assigned both topics) and this test fails — the single consumer's
flip drops every COMPLETED terminal. The fix is what makes it pass; the assertion
is the live battery bar (matrix advances through both arms with non-degenerate
rows), not a unit-isolated probe.

NOT the acceptance gate: per the task standing orders, the consume-leg wedge is
declared cleared ONLY by the LIVE K>=10 multi-cell reprobe on the stability lane,
never by this test. A green run here is necessary but not sufficient.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, ClassVar
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker

pytestmark = [pytest.mark.integration]

_COMPLETED_TOPIC = "onex.evt.omnimarket.node-generation-completed.v1"
_FAILED_TOPIC = "onex.evt.omnimarket.node-generation-failed.v1"
_COMMAND_TOPIC = "onex.cmd.omnimarket.node-generation-requested.v1"

# Battery shape: K trials over >=2 cells x >=2 arms.
_TRIALS_PER_CELL = 10
_CELLS = ("bool_flag", "address_norm")
_ARMS = ("off", "golden_exemplar")


def _generation_route() -> ModelRuntimeLocalIngressRoute:
    """Route declaring BOTH terminal topics (completed + failed), like the runner."""
    return ModelRuntimeLocalIngressRoute(
        node_name="node_generation_consumer",
        contract_name="generation_consumer",
        command_topic=_COMMAND_TOPIC,
        event_type="omnimarket.node-generation-requested",
        terminal_event=_COMPLETED_TOPIC,
        contract_path="/tmp/node_generation_consumer/contract.yaml",  # noqa: S108
        package_name="omnimarket",
        terminal_events=(_COMPLETED_TOPIC, _FAILED_TOPIC),
    )


class _PartitionLog:
    """Append-only partition log with a high-water-mark (Redpanda-like)."""

    def __init__(self) -> None:
        self._records: list[tuple[str, bytes]] = []
        self._lock = threading.Lock()

    @property
    def hwm(self) -> int:
        with self._lock:
            return len(self._records)

    def append(self, correlation_id: str, value: bytes) -> None:
        with self._lock:
            self._records.append((correlation_id, value))

    def at(self, offset: int) -> tuple[str, bytes] | None:
        with self._lock:
            if 0 <= offset < len(self._records):
                return self._records[offset]
            return None


class _Cluster:
    """Shared broker state across all ephemeral consumers in a battery run.

    COMPLETED is produced to every trial (it advertises a partition); FAILED is
    never produced to in a pass-only battery (no partition advertised — the
    Redpanda auto-create-on-produce steady state).
    """

    def __init__(self) -> None:
        self.logs: dict[str, _PartitionLog] = {
            _COMPLETED_TOPIC: _PartitionLog(),
            _FAILED_TOPIC: _PartitionLog(),
        }
        self.topics_with_partitions: set[str] = {_COMPLETED_TOPIC}


class _FakeFlipConsumer:
    """``AIOKafkaConsumer`` stand-in that models the live subscription flip.

    Faithful single-topic delivery, faithless multi-topic delivery
    --------------------------------------------------------------
    Delivery is correct ONLY when the consumer's assignment is confined to a
    SINGLE topic: ``getone`` returns the next record on that topic at/after the
    pinned offset. When ONE consumer is assigned partitions spanning MULTIPLE
    topics (the pre-OMN-13128 single-consumer-both-topics shape), the fake models
    the live aiokafka fault: it can only hold ONE topic's delivery subscription at
    a time and FLIPS which topic it reads each poll, so a record sitting on the
    topic that is not currently subscribed is never surfaced — exactly the
    COMPLETED terminal the live wedge dropped.

    The cluster's COMPLETED log advances every trial; the FAILED topic stays at
    HWM 0 and advertises no partition (partition-less, #1970).
    """

    cluster: ClassVar[_Cluster | None] = None
    refresh_latency_seconds: ClassVar[float] = 0.0

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.kwargs = kwargs
        self._client = _FakeClient(self)
        self._assigned: list[Any] = []
        self._cursors: dict[str, int] = {}
        self._flip_index = 0
        self.started = False
        self.stopped = False

    @property
    def _cluster(self) -> _Cluster:
        cluster = type(self).cluster
        assert cluster is not None, "test must set _FakeFlipConsumer.cluster"
        return cluster

    @property
    def _assigned_topics(self) -> list[str]:
        seen: list[str] = []
        for tp in self._assigned:
            topic = getattr(tp, "topic", None)
            if isinstance(topic, str) and topic not in seen:
                seen.append(topic)
        return seen

    async def start(self) -> None:
        self.started = True

    def partitions_for_topic(self, topic: str) -> set[int]:
        if topic not in self._client.tracked:
            return set()
        if topic not in self._cluster.topics_with_partitions:
            return set()
        return {0}

    def assign(self, partitions: Any) -> None:
        self._assigned = list(partitions)

    def assignment(self) -> set[Any]:
        return set(self._assigned)

    async def end_offsets(self, partitions: Any) -> dict[Any, int]:
        # Synchronous HWM round-trip (#1971): the next-message offset captured NOW
        # per assigned partition, without moving the consumer position.
        offsets: dict[Any, int] = {}
        for tp in partitions:
            topic = getattr(tp, "topic", "")
            offsets[tp] = self._cluster.logs[topic].hwm if topic else 0
        return offsets

    def seek(self, partition: Any, offset: int) -> None:
        topic = getattr(partition, "topic", "")
        if topic:
            self._cursors[topic] = offset

    async def seek_to_end(self, *partitions: object) -> None:
        for tp in partitions:
            topic = getattr(tp, "topic", "")
            if topic:
                self._cursors[topic] = self._cluster.logs[topic].hwm

    async def getone(self) -> SimpleNamespace:
        while True:
            topics = self._assigned_topics
            if not topics:
                await asyncio.sleep(0.005)
                continue
            if len(topics) == 1:
                # FAITHFUL single-topic delivery.
                record = self._read(topics[0])
                if record is not None:
                    return record
            else:
                # FAITHLESS multi-topic delivery: the live subscription flip. Only
                # ONE topic is readable per poll; a record on the other topic is
                # NOT surfaced (its delivery window was torn down by the flip).
                topic = topics[self._flip_index % len(topics)]
                self._flip_index += 1
                record = self._read(topic)
                if record is not None:
                    return record
            await asyncio.sleep(0.005)

    def _read(self, topic: str) -> SimpleNamespace | None:
        cursor = self._cursors.get(topic, 0)
        record = self._cluster.logs[topic].at(cursor)
        if record is None:
            return None
        self._cursors[topic] = cursor + 1
        _cid, value = record
        return SimpleNamespace(topic=topic, value=value)

    async def stop(self) -> None:
        self.stopped = True


class _FakeClient:
    def __init__(self, consumer: _FakeFlipConsumer) -> None:
        self._consumer = consumer
        self.tracked: set[str] = set()

    async def _refresh(self) -> bool:
        latency = type(self._consumer).refresh_latency_seconds
        if latency:
            await asyncio.sleep(latency)
        return True

    def set_topics(self, topics: list[str]) -> Any:
        self.tracked = set(topics)
        return self._refresh()

    def force_metadata_update(self) -> Any:
        return self._refresh()


class _FlipBattyTransport:
    """Bus that publishes the generation command and appends a COMPLETED terminal.

    Models the live FAST generation (dispatch loop freed, OMN-13010): the moment
    the broker publishes the worker command, the correlated COMPLETED terminal
    lands on its partition log (in the open->wait gap). The FAILED topic is never
    written (every generation passes).
    """

    config = SimpleNamespace(
        session_timeout_ms=45000,
        heartbeat_interval_ms=15000,
        max_poll_interval_ms=1800000,
        reconnect_backoff_ms=2000,
    )
    _bootstrap_servers = "omn-13128-flip-test-broker"

    def __init__(self, cluster: _Cluster) -> None:
        self._cluster = cluster

    def _build_auth_kwargs(self) -> dict[str, object]:
        return {}

    async def publish(
        self,
        topic: str,
        _key: bytes | None,
        value: bytes,
        _headers: object | None = None,
    ) -> None:
        if topic != _COMMAND_TOPIC:
            return
        command_envelope = ModelEventEnvelope[object].model_validate_json(value)
        correlation_id = str(command_envelope.correlation_id)
        terminal = ModelEventEnvelope[object](
            payload={
                "attempt_count": 1,
                "contract_passed": True,
                "model_id": "Qwen3.6-35B-A3B",
                "provider": "local",
            },
            correlation_id=command_envelope.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=_COMPLETED_TOPIC,
            source_tool="node_generation_consumer",
        )
        self._cluster.logs[_COMPLETED_TOPIC].append(
            correlation_id, terminal.model_dump_json().encode("utf-8")
        )

    def subscribe(self, *_args: object, **_kwargs: object) -> object:
        pytest.fail("Kafka-backed terminal waits must use the direct consumer path")


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_battery_correlates_with_independent_terminal_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K>=10 x 2 cells x 2 arms each correlate; no consumer holds two subscriptions.

    GREEN only when each terminal topic has its OWN consumer (OMN-13128): the
    COMPLETED consumer surfaces the correlated record on its single, non-flipping
    subscription while the partition-less FAILED consumer sleeps out its timeout.
    RED (single consumer assigned both topics): the flip drops every COMPLETED
    terminal and the matrix produces degenerate rows.
    """
    cluster = _Cluster()
    _FakeFlipConsumer.cluster = cluster
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeFlipConsumer,
    )
    # Scale the live 2s partition-less grace down so the FAILED open (serial,
    # pre-publish, per trial) returns its empty assignment promptly. This is the
    # #1970 grace window, unchanged in behaviour — only its duration is scaled so
    # the K>=10 x 2-cell x 2-arm matrix is fast under test. (At the live 2s the
    # battery still passes; it is just slow.)
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker."
        "_DIRECT_TERMINAL_PARTITIONLESS_GRACE_SECONDS",
        0.05,
    )

    broker = RuntimePatternBBroker(
        _FlipBattyTransport(cluster),
        command_topic="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        routes={"generation": _generation_route()},
    )

    # The runner's per-arm generation timeout (scaled down from the live 120s). A
    # correctly-positioned independent consumer correlates in milliseconds; a
    # flipping single consumer burns this whole window per trial and the matrix
    # blows past the @timeout, so RED reproduces in bounded time.
    timeout_seconds = 1

    correlated: list[tuple[str, str, int]] = []
    degenerate: list[tuple[str, str, int]] = []
    arms_with_real_rows: set[str] = set()
    cells_with_real_rows: set[str] = set()

    for cell in _CELLS:
        for trial in range(1, _TRIALS_PER_CELL + 1):
            for arm in _ARMS:
                command = ModelDispatchBusCommand(
                    command_name="generation",
                    requester="context-roi-runner",
                    payload={"cell": cell, "arm": arm, "trial": trial},
                    response_topic="onex.evt.pattern-b.dispatch-completed.v1",
                    timeout_seconds=timeout_seconds,
                    correlation_id=uuid4(),
                )
                _route, result = await broker.dispatch_request(command)
                if (
                    result.status == "completed"
                    and result.correlation_id == command.correlation_id
                ):
                    correlated.append((cell, arm, trial))
                    arms_with_real_rows.add(arm)
                    cells_with_real_rows.add(cell)
                else:
                    degenerate.append((cell, arm, trial))

    total = len(_CELLS) * _TRIALS_PER_CELL * len(_ARMS)

    # The COMPLETED terminal IS published every trial (HWM climbs) — confirms this
    # is a consume-leg correlation failure, not a missing publish (the exact live
    # signature: COMPLETED HWM climbs, FAILED HWM stays 0).
    assert cluster.logs[_COMPLETED_TOPIC].hwm == total, (
        "every trial must have published its correlated COMPLETED terminal "
        f"(HWM should climb to {total}, got {cluster.logs[_COMPLETED_TOPIC].hwm})"
    )
    assert cluster.logs[_FAILED_TOPIC].hwm == 0, (
        "the FAILED topic must stay at HWM 0 in a pass-only battery"
    )

    assert not degenerate, (
        f"{len(degenerate)}/{total} trials produced a degenerate row — a consumer "
        "that holds both terminal subscriptions flips and drops the already-"
        f"published COMPLETED terminal. First few: {degenerate[:5]}"
    )
    assert len(correlated) == total, (
        f"expected all {total} trials correlated, got {len(correlated)}"
    )
    assert cells_with_real_rows == set(_CELLS), (
        f"matrix did not advance through both cells with real rows: "
        f"{cells_with_real_rows} != {set(_CELLS)}"
    )
    assert arms_with_real_rows == set(_ARMS), (
        f"matrix did not advance through both arms with real rows: "
        f"{arms_with_real_rows} != {set(_ARMS)}"
    )
