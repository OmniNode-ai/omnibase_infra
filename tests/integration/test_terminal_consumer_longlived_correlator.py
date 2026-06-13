# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Long-lived correlator full-battery regression for the consume-leg wedge.

Ticket: OMN-13118 (Tier B canonical redesign; epic OMN-12525).

Why this REPLACES the per-trial-substrate repros
-------------------------------------------------
``test_terminal_consumer_battery_load_wedge.py`` (#1970),
``test_terminal_consumer_seek_reset_race.py`` (#1971), and
``test_terminal_consumer_subscription_flip_wedge.py`` (#1972) each modeled a
facet of the PER-TRIAL ephemeral consumer (a brand-new thread+loop+consumer+
client per (trial x terminal-topic)) and each went RED->GREEN offline while the
LIVE K>=10 battery wedged every time (HALT_K10_WEDGE_PERSISTS, strikes 3-5). The
five offset/subscription patches all tuned that ephemeral substrate; the design
re-examination (CONSUME_LEG_REDESIGN_DESIGN.md, B1_M1_M2_PROBE_RESULT.md)
concluded the EPHEMERALITY itself is the defect class and replaced it with ONE
long-lived correlator (one consumer subscribed once to both terminal topics, a
single poll loop, demux by correlation_id). With that substrate there is no
per-trial open/assign/pin/teardown to race, so those repros model machinery that
no longer exists.

What this test models HONESTLY (the live fault the prior fakes masked)
----------------------------------------------------------------------
The decisive live fact (HALT doc): the COMPLETED terminal IS published every
trial (its HWM climbs) but the per-trial COMPLETED consumer never surfaced it —
the COMPLETED leg logged NOTHING while only the empty-set FAILED leg timed out at
the 30s assign cap. The prior fakes could not reproduce that because each built a
fresh in-memory consumer per trial whose ``getone`` reliably returned the record.

This test drives the REAL long-lived correlator
(``make_terminal_event_consumer`` -> ``TerminalEventConsumer.open()/wait()`` ->
``LongLivedTerminalCorrelator``) the way ``HandlerContextRoiRunner._run_trial``
does — open BOTH terminal topics, register the cid before publish, race the wait
— across K=10 x 2 cells x 2 arms. The fake ``AIOKafkaConsumer`` is built ONCE
(the correlator never rebuilds it); it models a shared cluster where the
COMPLETED topic has a partition and the FAILED topic (never produced to) does
not. The terminal lands in the open->wait gap, and the single always-running poll
loop demuxes it to the waiting future.

The structural assertion that distinguishes the redesign from the old substrate:
exactly ONE consumer is constructed for the WHOLE battery (not 2 per trial).
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from omnibase_infra.runtime import service_pattern_b_broker
from omnibase_infra.runtime.service_terminal_event_consumer import (
    make_terminal_event_consumer,
)

pytestmark = [pytest.mark.integration]

_COMPLETED_TOPIC = "onex.evt.omnimarket.node-generation-completed.v1"
_FAILED_TOPIC = "onex.evt.omnimarket.node-generation-failed.v1"
_HANDLER = "HandlerContextRoiRunner"

_TRIALS_PER_CELL = 10
_CELLS = ("bool_flag", "address_norm")
_ARMS = ("off", "golden_exemplar")


class _PartitionLog:
    """Append-only partition log for one terminal topic with a high-water-mark."""

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

    def read_at(self, offset: int) -> tuple[str, bytes] | None:
        with self._lock:
            if offset < len(self._records):
                return self._records[offset]
            return None


class _Cluster:
    """Shared broker state for the single long-lived consumer.

    The COMPLETED topic has a partition (produced to every trial). The FAILED
    topic does NOT (never written) — Redpanda auto-create-on-produce, so the
    correlator assigns it the empty set (a valid steady state).
    """

    def __init__(self) -> None:
        self.logs: dict[str, _PartitionLog] = {
            _COMPLETED_TOPIC: _PartitionLog(),
            _FAILED_TOPIC: _PartitionLog(),
        }
        self.topics_with_partitions: set[str] = {_COMPLETED_TOPIC}
        self.consumers_built = 0


class _FakeClient:
    def __init__(self, consumer: _FakeConsumer) -> None:
        self._consumer = consumer
        self.tracked: set[str] = set()

    async def _refresh(self) -> bool:
        return True

    def set_topics(self, topics: list[str]) -> Any:
        self.tracked = set(topics)
        return self._refresh()

    def force_metadata_update(self) -> Any:
        return self._refresh()


class _FakeConsumer:
    """Single long-lived ``AIOKafkaConsumer`` stand-in for the correlator.

    The correlator builds exactly ONE of these for the whole batch and runs a
    single poll loop over it. ``assign`` accumulates partitions across both
    terminal topics; ``getone`` reads the next record across all assigned
    partitions in append order, advancing a per-partition cursor. The FAILED
    topic contributes no partition (never produced to).
    """

    cluster: ClassVar[_Cluster | None] = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._client = _FakeClient(self)
        self._assigned: set[Any] = set()
        self._cursors: dict[Any, int] = {}
        assert type(self).cluster is not None
        type(self).cluster.consumers_built += 1

    @property
    def _c(self) -> _Cluster:
        assert type(self).cluster is not None
        return type(self).cluster

    async def start(self) -> None:
        return None

    def partitions_for_topic(self, topic: str) -> set[int]:
        self._client.tracked.add(topic)
        if topic not in self._c.topics_with_partitions:
            return set()
        return {0}

    def assign(self, partitions: Any) -> None:
        self._assigned = set(partitions)

    def assignment(self) -> set[Any]:
        return set(self._assigned)

    async def seek_to_end(self, *assignment: object) -> None:
        return None

    async def end_offsets(self, partitions: Any) -> dict[Any, int]:
        out: dict[Any, int] = {}
        for tp in partitions:
            out[tp] = self._c.logs[tp.topic].hwm
        return out

    def seek(self, partition: Any, offset: int) -> None:
        self._cursors[partition] = offset

    async def getone(self) -> SimpleNamespace:
        import asyncio

        while True:
            for tp in sorted(self._assigned, key=lambda p: p.topic):
                cursor = self._cursors.get(tp, 0)
                record = self._c.logs[tp.topic].read_at(cursor)
                if record is not None:
                    self._cursors[tp] = cursor + 1
                    _cid, value = record
                    return SimpleNamespace(topic=tp.topic, value=value)
            await asyncio.sleep(0.002)

    async def stop(self) -> None:
        return None


class _KafkaLikeBus:
    config = SimpleNamespace(
        session_timeout_ms=45000,
        heartbeat_interval_ms=15000,
        max_poll_interval_ms=1800000,
        reconnect_backoff_ms=2000,
    )
    _bootstrap_servers = "omn-13118-tierb-test-broker"

    def _build_auth_kwargs(self) -> dict[str, object]:
        return {}


def _terminal_envelope_json(correlation_id: str) -> bytes:
    return json.dumps(
        {
            "correlation_id": correlation_id,
            "payload": {
                "attempt_count": 1,
                "contract_passed": True,
                "model_id": "Qwen3.6-35B-A3B",
                "provider": "local",
                "prompt_tokens": 128,
                "completion_tokens": 64,
            },
        }
    ).encode("utf-8")


def _run_one_trial(
    *,
    consumer_factory: Any,
    cluster: _Cluster,
    correlation_id: str,
    timeout_seconds: float,
) -> dict[str, object] | None:
    """Drive ONE trial as ``HandlerContextRoiRunner._run_trial`` does.

    open both terminals -> register cid -> publish (terminal lands in the gap)
    -> wait on COMPLETED. The single long-lived poll loop demuxes the gap-landed
    terminal to the registered future.
    """
    completed = consumer_factory.open(_COMPLETED_TOPIC)
    failed = consumer_factory.open(_FAILED_TOPIC)
    completed.register(correlation_id)
    failed.register(correlation_id)

    # FAST generation: correlated COMPLETED terminal lands in the open->wait gap.
    cluster.logs[_COMPLETED_TOPIC].append(
        correlation_id, _terminal_envelope_json(correlation_id)
    )

    try:
        return completed.wait(correlation_id, timeout_seconds)
    finally:
        completed.close()
        failed.close()


@pytest.mark.timeout(180)
def test_longlived_correlator_does_not_wedge_under_full_battery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K>=10 x 2 cells x 2 arms must each correlate via ONE long-lived consumer.

    RED on the per-trial substrate (the old ``TerminalConsumerSession`` rebuilt a
    consumer per topic per trial whose COMPLETED delivery raced its own
    bootstrap/teardown). GREEN on the long-lived correlator: one consumer,
    subscribed once, single poll loop demuxes every gap-landed terminal.
    """
    cluster = _Cluster()
    _FakeConsumer.cluster = cluster
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeConsumer,
    )
    monkeypatch.setattr(
        service_pattern_b_broker,
        "_DIRECT_TERMINAL_PARTITIONLESS_GRACE_SECONDS",
        0.05,
        raising=False,
    )

    bus = _KafkaLikeBus()
    consumer = make_terminal_event_consumer(event_bus=bus, handler_name=_HANDLER)
    timeout_seconds = 5.0

    correlated: list[tuple[str, str, int]] = []
    degenerate: list[tuple[str, str, int]] = []
    arms_with_real_rows: set[str] = set()
    cells_with_real_rows: set[str] = set()

    try:
        run_order = 0
        for cell in _CELLS:
            for trial in range(1, _TRIALS_PER_CELL + 1):
                for arm in _ARMS:
                    run_order += 1
                    cid = f"omn-13118-tierb-{cell}-{arm}-t{trial}"
                    result = _run_one_trial(
                        consumer_factory=consumer,
                        cluster=cluster,
                        correlation_id=cid,
                        timeout_seconds=timeout_seconds,
                    )
                    if result is not None and str(result.get("correlation_id")) == cid:
                        correlated.append((cell, arm, trial))
                        arms_with_real_rows.add(arm)
                        cells_with_real_rows.add(cell)
                    else:
                        degenerate.append((cell, arm, trial))

        total = len(_CELLS) * _TRIALS_PER_CELL * len(_ARMS)

        # Structural proof of the redesign: exactly ONE consumer for the WHOLE
        # battery, not 2 per trial. This is the property the five prior fixes
        # could not establish (they multiplied #1-7 per trial-topic).
        assert cluster.consumers_built == 1, (
            f"the long-lived correlator must build exactly ONE consumer for the "
            f"whole battery; built {cluster.consumers_built} "
            f"(per-trial ephemeral churn is the defect class being removed)"
        )

        # The COMPLETED-topic HWM CLIMBED (every trial published its terminal) —
        # the wedge is a consume-leg correlation failure, not a missing publish.
        assert cluster.logs[_COMPLETED_TOPIC].hwm == total, (
            f"every trial's COMPLETED terminal must be published (HWM should be "
            f"{total}, got {cluster.logs[_COMPLETED_TOPIC].hwm})"
        )
        assert cluster.logs[_FAILED_TOPIC].hwm == 0, (
            "the FAILED topic must stay at HWM 0 (nothing lands there)"
        )

        assert not degenerate, (
            f"{len(degenerate)}/{total} trials produced a degenerate row — the "
            "correlator failed to demux the gap-landed COMPLETED terminal to the "
            f"registered future. First few: {degenerate[:5]}"
        )
        assert len(correlated) == total, (
            f"expected all {total} trials correlated, got {len(correlated)}"
        )
        assert cells_with_real_rows == set(_CELLS), (
            f"matrix did not advance through both cells: {cells_with_real_rows}"
        )
        assert arms_with_real_rows == set(_ARMS), (
            f"matrix did not advance through both arms: {arms_with_real_rows}"
        )
    finally:
        consumer.close()


@pytest.mark.timeout(60)
def test_correlator_reads_terminal_published_in_open_wait_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Focused repro: a terminal published in the register->wait gap is demuxed.

    The minimal shape of the wedge: open + register the cid BEFORE publish, the
    correlated terminal lands in the gap, then wait reads it off the single poll
    loop. The session.close() is a no-op and must NOT tear the shared correlator
    down (the teardown-races-fetch class the redesign eliminates).
    """
    cluster = _Cluster()
    _FakeConsumer.cluster = cluster
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeConsumer,
    )

    bus = _KafkaLikeBus()
    consumer = make_terminal_event_consumer(event_bus=bus, handler_name=_HANDLER)
    cid = "omn-13118-tierb-single-gap-0001"

    try:
        session = consumer.open(_COMPLETED_TOPIC)
        session.register(cid)
        cluster.logs[_COMPLETED_TOPIC].append(cid, _terminal_envelope_json(cid))

        result = session.wait(cid, timeout_seconds=5.0)

        assert result is not None, (
            "the COMPLETED terminal published in the register->wait gap was not "
            "demuxed to the registered future by the long-lived poll loop"
        )
        assert str(result.get("correlation_id")) == cid
        # The shared consumer survives the per-session close (no per-trial churn).
        assert cluster.consumers_built == 1
        session.close()

        # A second trial reuses the SAME consumer (no rebuild).
        cid2 = "omn-13118-tierb-single-gap-0002"
        session2 = consumer.open(_COMPLETED_TOPIC)
        session2.register(cid2)
        cluster.logs[_COMPLETED_TOPIC].append(cid2, _terminal_envelope_json(cid2))
        result2 = session2.wait(cid2, timeout_seconds=5.0)
        assert result2 is not None
        assert cluster.consumers_built == 1, (
            "a second trial must reuse the long-lived consumer, not rebuild it"
        )
    finally:
        consumer.close()


@pytest.mark.timeout(60)
def test_correlator_single_call_form_correlates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy ``(topic, cid, timeout)`` callable form still correlates.

    Open + register + wait in one step against the shared correlator. The
    terminal arrives DURING the wait (a correlated terminal published at-or-after
    the pinned end offset), and the always-running poll loop demuxes it to the
    registered future. (The single-call form pins the read position at call time,
    so a record published in the wait window is delivered; a record produced
    strictly BEFORE the call is past the pinned end — the documented race-prone
    legacy behavior the two-phase open/register path avoids.)
    """
    import threading as _threading

    cluster = _Cluster()
    _FakeConsumer.cluster = cluster
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeConsumer,
    )

    bus = _KafkaLikeBus()
    consumer = make_terminal_event_consumer(event_bus=bus, handler_name=_HANDLER)
    cid = "omn-13118-tierb-single-call-0001"
    try:
        # Publish the correlated terminal shortly AFTER the single-call wait
        # begins, so it lands at-or-after the pinned end offset (the in-window
        # delivery the poll loop demuxes).
        def _publish_during_wait() -> None:
            time.sleep(0.2)
            cluster.logs[_COMPLETED_TOPIC].append(cid, _terminal_envelope_json(cid))

        _threading.Thread(target=_publish_during_wait, daemon=True).start()
        result = consumer(_COMPLETED_TOPIC, cid, 5.0)
        assert result is not None
        assert str(result.get("correlation_id")) == cid
    finally:
        consumer.close()
