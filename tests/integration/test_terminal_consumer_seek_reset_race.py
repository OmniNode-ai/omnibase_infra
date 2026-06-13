# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lazy-``seek_to_end`` reset race regression for the consume-leg wedge (OMN-13118).

Ticket: OMN-13118 (strike-three residual after PR #1969 set_topics no-op fix and
PR #1970 partition-less assign-cap fix; the wedge PERSISTS on REBUILD-4
``ab9d606d1``).

Why a NEW test instead of the two existing consume-leg regressions
------------------------------------------------------------------
``test_terminal_consumer_concurrent_assign_race.py`` (OMN-13012/#1969) models the
``set_topics`` metadata no-op. ``test_terminal_consumer_battery_load_wedge.py``
(OMN-13118/#1970) models the partition-less FAILED-open assign-cap burn. BOTH are
GREEN on the wedging image — and both miss the live defect, because both fakes
make ``seek_to_end`` SYNCHRONOUS and EXACT: they snapshot the partition HWM the
instant ``seek_to_end`` is called and read from that cursor. The real
``AIOKafkaConsumer.seek_to_end`` does NOT do that.

The live defect (per HALT_K10_WEDGE_PERSISTS.md + runner-logs-tail.txt)
-----------------------------------------------------------------------
Per ~30s cycle on the wedged battery:
  1. the FAILED-topic wait times out (empty message) — EXPECTED, nothing is ever
     produced to FAILED, so its consumer correctly returns None; and
  2. the COMPLETED terminal IS published every trial (decisive: the ``cid`` that
     "wait failed" in cycle N is the ``correlation_id`` published to COMPLETED in
     cycle N-1), yet
  3. the matrix re-fires the SAME cell forever — the COMPLETED ``wait`` returns
     None even though the correlated terminal was published at an offset
     at-or-after the consumer's open() position.

Root mechanism modeled here (the real aiokafka 0.13.0 semantics)
----------------------------------------------------------------
``open_direct_terminal_consumer`` does ``start -> assign -> seek_to_end`` and
then the caller publishes; ``poll`` then ``getone``s. But
``AIOKafkaConsumer.seek_to_end`` is LAZY: it calls
``Fetcher.request_offset_reset(partitions, LATEST)``, which only marks the
partition "awaiting reset" and registers a ``wait_for_position()`` future. The
actual LATEST offset is resolved by a ``ListOffsets`` round-trip that happens on
the FIRST fetch/poll — AFTER the caller has published. With generation completing
in ~1s (dispatch loop freed, OMN-13010), the correlated COMPLETED terminal lands
in the gap between ``assign`` and that lazy LATEST resolution, so the resolved
position is the HWM AFTER the record. ``getone`` then starts reading past the
already-published terminal and blocks forever — the COMPLETED ``wait`` returns
None and the trial never correlates. This is the OMN-13012 probe3 "seek past the
terminal" race resurfacing: ``seek_to_end`` does not actually pin a pre-publish
offset, it requests a position that resolves LATER, after the publish.

This fake therefore models ``seek_to_end`` FAITHFULLY as a deferred LATEST reset
(NOT an immediate HWM snapshot): the resolved read offset is whatever the
partition HWM is at the time of the FIRST ``getone``, which by then includes the
freshly-published terminal — skipping it.

The defining assertions (RED before the fix, GREEN after)
---------------------------------------------------------
Driving the REAL ``TerminalEventConsumer.open()/wait()`` the way
``HandlerContextRoiRunner._run_trial`` does — open BOTH terminals per trial,
publish the correlated COMPLETED terminal in the publish->wait gap, race the wait
— across K>=10 trials over >=2 cells x >=2 arms:
  - every trial's COMPLETED terminal is published (HWM climbs to total), and
  - every cell across both arms correlates its COMPLETED terminal (no degenerate
    rows, matrix advances through both arms), and
  - HWMs settle (FAILED stays 0).

RED on the lazy-reset consume leg (the resolved position lands past the terminal,
every cell degenerate); GREEN once the consume leg pins the EXACT pre-publish end
offset synchronously (``end_offsets`` + ``seek``) instead of a deferred LATEST
reset.
"""

from __future__ import annotations

import asyncio
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

# Battery shape: K trials over >=2 cells x >=2 arms (mirrors the STRONG K>=10
# multi-cell reprobe that wedged on REBUILD-4 without launching the full 160).
_TRIALS_PER_CELL = 10
_CELLS = ("bool_flag", "address_norm")
_ARMS = ("off", "golden_exemplar")


class _BrokerLog:
    """Append-only partition log for one terminal topic with a high-water-mark.

    Models a Redpanda partition. The COMPLETED topic's HWM advances every trial
    (generation succeeds and publishes its correlated terminal); the FAILED
    topic's HWM stays 0 (nothing ever lands there).
    """

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

    def read_from(self, offset: int) -> list[tuple[str, bytes]]:
        with self._lock:
            return list(self._records[offset:])


class _Cluster:
    """Cluster-wide shared state across all ephemeral consumers (load model).

    The COMPLETED topic has a partition (produced to every trial). The FAILED
    topic does NOT (never written in this battery), modeling Redpanda's
    auto-create-on-produce — a partition-less reply topic.
    """

    def __init__(self) -> None:
        self.logs: dict[str, _BrokerLog] = {
            _COMPLETED_TOPIC: _BrokerLog(),
            _FAILED_TOPIC: _BrokerLog(),
        }
        self.topics_with_partitions: set[str] = {_COMPLETED_TOPIC}


class _FakeClient:
    """``AIOKafkaClient`` topic-tracking stand-in with awaitable metadata ops."""

    def __init__(self, consumer: _FakeConsumer) -> None:
        self._consumer = consumer
        self.tracked: set[str] = set()

    async def _refresh(self) -> bool:
        await asyncio.sleep(0)
        return True

    def set_topics(self, topics: list[str]) -> Any:
        self.tracked = set(topics)
        return self._refresh()

    def force_metadata_update(self) -> Any:
        return self._refresh()


class _FakeConsumer:
    """Faithful ``AIOKafkaConsumer`` stand-in modeling LAZY ``seek_to_end``.

    Drives ``open_direct_terminal_consumer`` / ``poll_direct_terminal_consumer``
    UNCHANGED: ``start`` -> ``partitions_for_topic`` -> ``assign`` ->
    ``seek_to_end`` -> ``getone`` (or ``end_offsets`` + ``seek`` once the consume
    leg is fixed).

    The critical fidelity — and the difference from the two prior consume-leg
    fakes — is that ``seek_to_end`` is a DEFERRED LATEST reset, NOT an immediate
    HWM snapshot. Calling ``seek_to_end`` only marks the partition "awaiting
    reset". The read offset is resolved on the FIRST ``getone`` to the partition
    HWM AT THAT MOMENT (a real broker ``ListOffsets`` round-trip that happens
    after the caller has already published). So a record produced in the
    open()->getone() gap is SKIPPED — exactly the live wedge.

    A consume leg that instead synchronously captures the end offset via
    ``end_offsets`` and pins it via ``seek`` resolves the read position at open()
    time (before the publish), so the gap record is delivered. ``end_offsets``
    here returns the HWM at the instant it is called (a real synchronous broker
    round-trip), and ``seek`` pins that offset deterministically.
    """

    cluster: ClassVar[_Cluster | None] = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.kwargs = kwargs
        self._client = _FakeClient(self)
        self._topic: str | None = None
        self._assigned_partitions: set[Any] = set()
        # None => the partition is "awaiting (lazy) reset"; the read offset is
        # resolved to the live HWM on the first getone. An int => a pinned offset
        # (set synchronously by seek()).
        self._cursor: int | None = None
        self.started = False
        self.stopped = False

    @property
    def _cluster(self) -> _Cluster:
        cluster = type(self).cluster
        assert cluster is not None, "test must set _FakeConsumer.cluster"
        return cluster

    async def start(self) -> None:
        self.started = True

    def partitions_for_topic(self, topic: str) -> set[int]:
        self._topic = topic
        if topic not in self._client.tracked:
            return set()
        if topic not in self._cluster.topics_with_partitions:
            return set()
        return {0}

    def assign(self, partitions: Any) -> None:
        self._assigned_partitions = set(partitions)
        # A fresh assignment with auto_offset_reset="latest" leaves the position
        # UNRESOLVED until seek_to_end / end_offsets+seek positions it.
        self._cursor = None

    def assignment(self) -> set[Any]:
        return set(self._assigned_partitions)

    async def seek_to_end(self, *assignment: object) -> None:
        # LAZY: request a LATEST reset but DO NOT resolve the offset now. The
        # real Fetcher.request_offset_reset only marks the partition "awaiting
        # reset"; the LATEST offset is fetched on the first poll. Leaving the
        # cursor unresolved here models that deferral faithfully.
        return None

    async def end_offsets(self, partitions: Any) -> dict[Any, int]:
        # Synchronous broker round-trip: return the CURRENT HWM (the offset of
        # the next message). Does NOT change the consumer position. This is the
        # offset the fixed consume leg pins via seek() at open() time.
        assert self._topic is not None
        hwm = self._cluster.logs[self._topic].hwm
        return dict.fromkeys(partitions, hwm)

    def seek(self, partition: Any, offset: int) -> None:
        # Deterministically pin the read offset NOW (synchronous, no broker
        # round-trip): the position the fixed consume leg resumes from.
        self._cursor = offset

    async def getone(self) -> SimpleNamespace:
        assert self._topic is not None
        log = self._cluster.logs[self._topic]
        while True:
            if self._cursor is None:
                # Deferred LATEST reset resolves NOW — to the live HWM, which by
                # this point includes the terminal published in the open->getone
                # gap. That record is therefore skipped (the live wedge).
                self._cursor = log.hwm
            records = log.read_from(self._cursor)
            if records:
                _cid, value = records[0]
                self._cursor += 1
                return SimpleNamespace(topic=self._topic, value=value)
            await asyncio.sleep(0.002)

    async def stop(self) -> None:
        self.stopped = True


class _KafkaLikeBus:
    config = SimpleNamespace(
        session_timeout_ms=45000,
        heartbeat_interval_ms=15000,
        max_poll_interval_ms=1800000,
        reconnect_backoff_ms=2000,
    )
    _bootstrap_servers = "omn-13118-seek-race-test-broker"

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


def _open_session_or_none(consumer_factory: Any, topic: str) -> Any:
    """Open a terminal session, returning None if open raises.

    Mirrors ``HandlerContextRoiRunner._open_terminal_session``: a failed open is
    caught and the runner falls back to the legacy single-call path for that
    topic.
    """
    try:
        return consumer_factory.open(topic)
    except Exception:  # noqa: BLE001 — mirrors handler's catch-all open fallback
        return None


def _run_one_trial(
    *,
    consumer_factory: Any,
    cluster: _Cluster,
    correlation_id: str,
    timeout_seconds: float,
) -> dict[str, object] | None:
    """Drive ONE trial as ``HandlerContextRoiRunner._run_trial`` does.

    Ordering (subscribe-before-publish, OMN-13012/13038):
      1. open the COMPLETED terminal session to current end;
      2. open the FAILED terminal session to current end (partition-less, returns
         promptly post-#1970);
      3. "publish" the generation command — FAST generation appends the
         correlated COMPLETED terminal IMMEDIATELY, in the open->wait gap;
      4. wait on the COMPLETED session.

    The terminal landing in step 3 is the whole point: a consume leg that only
    resolves its read offset on the first poll (lazy ``seek_to_end``) lands PAST
    this record and never reads it.
    """
    completed_session = _open_session_or_none(consumer_factory, _COMPLETED_TOPIC)
    failed_session = _open_session_or_none(consumer_factory, _FAILED_TOPIC)

    # FAST generation publishes the correlated COMPLETED terminal in the gap
    # between open() and the wait()'s first poll (mirrors ~1s gen vs the per-trial
    # consumer open). The HWM has now advanced past the consumer's open position.
    cluster.logs[_COMPLETED_TOPIC].append(
        correlation_id, _terminal_envelope_json(correlation_id)
    )

    try:
        if completed_session is None:
            return None
        return completed_session.wait(correlation_id, timeout_seconds)
    finally:
        if failed_session is not None:
            failed_session.close()


@pytest.mark.timeout(180)
def test_full_battery_consume_leg_reads_terminal_published_in_open_wait_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K>=10 x 2 cells x 2 arms: the correlated COMPLETED terminal published in
    the open->wait gap must be read by every trial; the matrix must not wedge.

    RED on the lazy-``seek_to_end`` consume leg: the read offset resolves on the
    first poll to the HWM that ALREADY includes the gap-published terminal, so
    every cell is degenerate and the matrix never advances. GREEN once the consume
    leg pins the exact pre-publish end offset synchronously.
    """
    cluster = _Cluster()
    _FakeConsumer.cluster = cluster
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeConsumer,
    )
    # Keep the partition-less grace tiny so the FAILED open returns promptly
    # (post-#1970 behavior) and the test stays fast; the seek-race is independent
    # of this grace.
    monkeypatch.setattr(
        service_pattern_b_broker,
        "_DIRECT_TERMINAL_PARTITIONLESS_GRACE_SECONDS",
        0.05,
        raising=False,
    )

    bus = _KafkaLikeBus()
    consumer = make_terminal_event_consumer(event_bus=bus, handler_name=_HANDLER)

    # The runner's per-arm timeout (generation_timeout_seconds in the battery),
    # scaled down so a wedged battery (every trial burns its full window then
    # returns None) still completes inside the test timeout while remaining
    # unambiguously RED. A correctly-positioned consumer correlates in
    # milliseconds; a wedged one burns this whole window per degenerate trial.
    timeout_seconds = 1.0

    correlated: list[tuple[str, str, int]] = []
    degenerate: list[tuple[str, str, int]] = []
    arms_with_real_rows: set[str] = set()
    cells_with_real_rows: set[str] = set()

    run_order = 0
    for cell in _CELLS:
        for trial in range(1, _TRIALS_PER_CELL + 1):
            for arm in _ARMS:
                run_order += 1
                cid = f"omn-13118-{cell}-{arm}-t{trial}"
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

    # The COMPLETED-topic HWM CLIMBED (every trial published its terminal) — the
    # wedge is NOT a missing-publish problem but a consume-leg positioning failure.
    assert cluster.logs[_COMPLETED_TOPIC].hwm == total, (
        "every trial's correlated COMPLETED terminal must have been published "
        f"(HWM should climb to {total}, got {cluster.logs[_COMPLETED_TOPIC].hwm})"
    )
    # HWMs settle: nothing ever lands on FAILED.
    assert cluster.logs[_FAILED_TOPIC].hwm == 0, (
        "the FAILED topic must stay at HWM 0 (nothing lands there)"
    )

    # Every cell across both arms correlates its COMPLETED terminal: no degenerate
    # rows, matrix advances through both cells AND both arms.
    assert not degenerate, (
        f"{len(degenerate)}/{total} trials produced a degenerate row — the "
        "per-trial consume leg failed to read the COMPLETED terminal published "
        "in the open->wait gap (lazy seek_to_end resolved the read offset PAST "
        f"the record). First few degenerate: {degenerate[:5]}"
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


@pytest.mark.timeout(60)
def test_single_trial_reads_terminal_published_immediately_after_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Focused unit-style repro: a terminal published in the open->wait gap is read.

    The minimal shape of the wedge: open COMPLETED to current end, publish the
    correlated terminal, then wait. The fixed consume leg pins the pre-publish end
    offset so the wait reads the record; the lazy-reset leg resolves past it.
    """
    cluster = _Cluster()
    _FakeConsumer.cluster = cluster
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeConsumer,
    )

    bus = _KafkaLikeBus()
    consumer = make_terminal_event_consumer(event_bus=bus, handler_name=_HANDLER)
    cid = "omn-13118-single-gap-0001"

    session = consumer.open(_COMPLETED_TOPIC)
    # Terminal lands in the open->wait gap (HWM advances past the open position).
    cluster.logs[_COMPLETED_TOPIC].append(cid, _terminal_envelope_json(cid))

    result = session.wait(cid, timeout_seconds=5.0)

    assert result is not None, (
        "the COMPLETED terminal published in the open->wait gap was not read — "
        "the consume leg resolved its read offset to the post-publish HWM "
        "(lazy seek_to_end) instead of pinning the pre-publish end offset"
    )
    assert str(result.get("correlation_id")) == cid


@pytest.mark.timeout(60)
def test_open_pins_end_offset_before_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consume leg must resolve its read position at open() time, not on poll.

    Asserts the structural fix: after ``open`` returns, the assigned partition's
    read offset is already pinned (an int, captured synchronously) rather than
    left as a deferred LATEST reset that resolves on the first poll. This is the
    invariant that makes a terminal published in the open->wait gap readable.
    """
    cluster = _Cluster()
    _FakeConsumer.cluster = cluster
    captured: dict[str, _FakeConsumer] = {}

    real_init = _FakeConsumer.__init__

    def _tracking_init(self: _FakeConsumer, *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)
        # Record the consumer the moment partitions_for_topic binds its topic so
        # the test can inspect the pinned cursor after open().
        original = self.partitions_for_topic

        def _track(topic: str) -> set[int]:
            result = original(topic)
            if topic == _COMPLETED_TOPIC:
                captured["completed"] = self
            return result

        self.partitions_for_topic = _track  # type: ignore[method-assign]

    monkeypatch.setattr(_FakeConsumer, "__init__", _tracking_init)
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeConsumer,
    )

    bus = _KafkaLikeBus()
    consumer = make_terminal_event_consumer(event_bus=bus, handler_name=_HANDLER)

    # Seed one pre-existing record so a pinned end offset is a non-zero, exact
    # value the fix must capture synchronously.
    cluster.logs[_COMPLETED_TOPIC].append(
        "preexisting", _terminal_envelope_json("preexisting")
    )
    pre_publish_hwm = cluster.logs[_COMPLETED_TOPIC].hwm

    session = consumer.open(_COMPLETED_TOPIC)
    try:
        # Give the worker loop a beat to finish open().
        deadline = time.monotonic() + 5.0
        while "completed" not in captured and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "completed" in captured, "COMPLETED consumer was never constructed"
        fake = captured["completed"]
        assert fake._assigned_partitions, "COMPLETED partition was never assigned"
        assert fake._cursor == pre_publish_hwm, (
            "open() did not pin the pre-publish end offset synchronously — the "
            f"read cursor is {fake._cursor!r}, expected {pre_publish_hwm} "
            "(an unresolved cursor means a lazy LATEST reset that races the "
            "publish)"
        )
    finally:
        session.close()
