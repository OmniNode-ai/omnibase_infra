# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Full-battery (K>=10 x multi-cell) regression for the consume-leg wedge (OMN-13118).

Ticket: OMN-13118 (post-OMN-13012/OMN-13038 residual; load-dependent).

Why a NEW test instead of the existing assign-race test
-------------------------------------------------------
``test_terminal_consumer_concurrent_assign_race.py`` (OMN-13012, PR #1969)
reproduced the FIRST-iteration metadata no-op and is GREEN on the deployed
image. The K>=2 / single-task reprobe also passed on that image. Yet the FULL
Exp 1 battery (K=10 x 8 tasks) WEDGES on the same image
(``docs/evidence/2026-06-12-weekend-pass/experiments/exp1/
HALT_RUNNER_CONSUME_LEG_REGRESSION_RELAUNCH.md`` +
``HALT_VERIFIER_SIGNOFF.md``, verifier-confirmed HALT_CONFIRMED).

The gate gap: the narrow probes open at most a handful of ephemeral
``group_id=None`` terminal consumers. The battery opens TWO per trial
(completed + failed, OMN-13038) across 160 trials = 320 ephemeral consumer
start/assign/seek/stop cycles. The wedge is therefore load-dependent and only
surfaces once the consume leg has accumulated enough per-trial ephemeral
consumer churn to slow broker metadata. This test reproduces THAT path: it
drives the REAL ``TerminalEventConsumer.open()/wait()`` (the object the runtime
injects as ``event_consumer``) the way ``HandlerContextRoiRunner._run_trial``
does — open BOTH terminal topics per trial, publish, race the wait — across
K>=10 trials over >=2 cells and >=2 arms.

The live wedge signature (the model this test must reproduce RED)
-----------------------------------------------------------------
Per the verifier sign-off, on the wedged battery:
  - ``node-generation-completed.v1`` HWM CLIMBS (generation succeeds; the
    correlated COMPLETED terminal IS published every trial), but
  - ``node-generation-failed.v1`` HWM stays 0 — NOTHING is ever produced to it,
    so the broker (Redpanda, auto-create-on-produce) advertises NO partitions
    for it, and
  - the runner logs ``wait failed ... topic=...failed.v1`` with an EMPTY
    exception every ~30s ( == the assign cap), and
  - per-cell the matrix re-fires ~40x and stalls; ZERO usable rows in window.

Root cause modeled (the FAILED-topic assign-cap stall):
``HandlerContextRoiRunner._run_trial`` opens BOTH terminal sessions BEFORE it
publishes (subscribe-before-publish, OMN-13038): first COMPLETED, then FAILED,
serially, in a dict comprehension. ``open_direct_terminal_consumer`` runs
``start -> _assign_direct_terminal_partitions -> seek_to_end``. For the FAILED
topic, which has had ZERO messages produced to it, the broker never advertises
any partitions, so ``_assign_direct_terminal_partitions`` spins forcing metadata
refreshes until the 30s ``_DIRECT_TERMINAL_ASSIGN_TIMEOUT_CAP_SECONDS`` cap and
raises a bare ``TimeoutError`` (the empty-message ``wait failed``). Because that
open is INLINE in the per-trial pre-publish phase, EVERY trial blocks ~30s on
the FAILED open before it can publish and correlate the COMPLETED terminal. A
160-trial battery therefore needs >=160 x 30s = 80 min and never completes in
any usable window — the operational wedge.

The consume leg must not let a terminal topic that legitimately has no
partitions yet (zero messages produced) burn the whole assign cap on every
trial. A partition-less reply topic is a valid steady state (the FAILED topic is
only written on contract_passed=False), not a 30s-blocking error.

RED before the fix: with the FAILED reply topic partition-less, each trial
blocks the full assign cap, so the K>=10 x 2-cell x 2-arm matrix cannot finish
within the battery's per-arm window — most cells time out / produce no
correlated row and the elapsed wall time blows past any usable bound. GREEN
after: the consume leg tolerates a partition-less reply topic (the FAILED open
returns promptly without burning the cap), so every cell across both arms
correlates its COMPLETED terminal quickly under the same sustained load.
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

# Battery shape: K trials over >=2 cells x >=2 arms (mirrors spec 1.B at a
# size that exercises the multi-trial churn path without a real broker).
_TRIALS_PER_CELL = 10
_CELLS = ("bool_flag", "address_norm")
_ARMS = ("off", "golden_exemplar")

# The live assign cap is 30s; scale it to 1.5s here so a cap-burning FAILED open
# is observable in a fast test. The wedge is a PER-TRIAL stall of one full cap.
_SCALED_ASSIGN_CAP_SECONDS = 1.5
# The live partition-less grace is 2s; scale it to 0.2s so a FIXED consume leg
# returns the partition-less FAILED open promptly (the proof the fix works).
_SCALED_PARTITIONLESS_GRACE_SECONDS = 0.2
# A correctly-behaving trial finishes in a small fraction of the cap (just the
# grace window + correlate). A wedged trial blocks ~one full cap on the FAILED
# open. This bound sits between them and is the wall-clock wedge gate.
_MAX_HEALTHY_TRIAL_SECONDS = 0.8


class _BrokerLog:
    """Shared in-memory broker partition log for one terminal topic.

    Models a Redpanda partition: an append-only list of (correlation_id, value)
    with a high-water-mark. ``seek_to_end`` captures the HWM as the resume
    offset; ``read_from`` returns records at-or-after that offset. The COMPLETED
    topic's HWM advances every trial (generation succeeds and publishes its
    correlated terminal); the FAILED topic's HWM stays 0 (nothing ever lands).
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

    A single cluster instance is shared by every ephemeral ``AIOKafkaConsumer``
    the consume leg builds. It owns the per-topic partition logs and the set of
    topics the broker advertises partitions for. The COMPLETED topic has a
    partition (it is produced to every trial); the FAILED topic does NOT — it is
    only written on contract_passed=False and in this battery nothing fails, so
    the broker (auto-create-on-produce) never advertises any partition for it.
    A consumer assigning the FAILED topic therefore sees partitions==None
    forever and the assign loop burns the whole cap.
    """

    def __init__(self) -> None:
        self.logs: dict[str, _BrokerLog] = {
            _COMPLETED_TOPIC: _BrokerLog(),
            _FAILED_TOPIC: _BrokerLog(),
        }
        # Topics the broker advertises partitions for. The FAILED topic, never
        # produced to, is absent — modeling Redpanda's auto-create-on-produce.
        self.topics_with_partitions: set[str] = {_COMPLETED_TOPIC}


class _FakeConsumer:
    """Faithful ``AIOKafkaConsumer`` stand-in driving the real consume leg.

    Exercises ``open_direct_terminal_consumer`` /
    ``poll_direct_terminal_consumer`` UNCHANGED: ``start`` ->
    ``partitions_for_topic`` -> ``assign`` -> ``seek_to_end`` -> ``getone``.

    The critical fidelity: ``partitions_for_topic`` returns an empty set for a
    topic the broker does not advertise (``FAILED`` topic, never produced to),
    exactly like a real ``AIOKafkaConsumer`` over a non-existent / unwritten
    Redpanda topic. The production assign loop then spins forcing metadata
    refreshes until its cap and raises a bare ``TimeoutError`` — the live
    empty-message ``wait failed``.

    The assign cap is patched DOWN for the test (real cap is 30s) so a
    cap-burning open is observable in seconds; the per-trial wall-time blowup is
    measured against that scaled cap.
    """

    cluster: ClassVar[_Cluster | None] = None
    # Each refresh cycle costs a little real time so a cap-burning assign loop
    # blocks measurable wall time (mirrors the live 30s stall, scaled down).
    refresh_latency_seconds: ClassVar[float] = 0.01

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.kwargs = kwargs
        self._client = _FakeClient(self)
        self._topic: str | None = None
        self._assigned_partitions: set[Any] = set()
        self._cursor: int = 0
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
        # The broker advertises partitions ONLY for topics that have been
        # produced to. A never-written reply topic surfaces none, forever.
        if topic not in self._cluster.topics_with_partitions:
            return set()
        return {0}

    def assign(self, partitions: Any) -> None:
        # Faithful: assigning an empty list yields an empty assignment (the
        # OMN-13118 partition-less path), matching aiokafka semantics.
        self._assigned_partitions = set(partitions)

    def assignment(self) -> set[Any]:
        return set(self._assigned_partitions)

    async def seek_to_end(self, *assignment: object) -> None:
        # Position to the CURRENT end of the partition — the live high-water
        # mark captured pre-publish at open() time. A no-op for an empty
        # assignment (partition-less reply topic).
        if not assignment:
            return
        assert self._topic is not None
        self._cursor = self._cluster.logs[self._topic].hwm

    async def getone(self) -> SimpleNamespace:
        assert self._topic is not None
        log = self._cluster.logs[self._topic]
        while True:
            records = log.read_from(self._cursor)
            if records:
                _cid, value = records[0]
                self._cursor += 1
                return SimpleNamespace(topic=self._topic, value=value)
            await asyncio.sleep(0.005)

    async def stop(self) -> None:
        self.stopped = True


class _FakeClient:
    """Models ``AIOKafkaClient`` topic tracking with a real metadata round-trip.

    OMN-13012's metadata no-op is already fixed (PR #1969): every
    ``force_metadata_update`` here actually "fetches" (and costs a little wall
    time). The point this test isolates is that a topic the broker does not
    advertise NEVER gains partitions no matter how many refreshes are forced —
    so the assign loop runs to its cap. ``set_topics`` /
    ``force_metadata_update`` return awaitables, matching the real shape the
    consume leg awaits via ``_await_metadata_op``.
    """

    def __init__(self, consumer: _FakeConsumer) -> None:
        self._consumer = consumer
        self.tracked: set[str] = set()

    async def _refresh(self) -> bool:
        await asyncio.sleep(type(self._consumer).refresh_latency_seconds)
        return True

    def set_topics(self, topics: list[str]) -> Any:
        self.tracked = set(topics)
        return self._refresh()

    def force_metadata_update(self) -> Any:
        return self._refresh()


class _KafkaLikeBus:
    config = SimpleNamespace(
        session_timeout_ms=45000,
        heartbeat_interval_ms=15000,
        max_poll_interval_ms=1800000,
        reconnect_backoff_ms=2000,
    )
    _bootstrap_servers = "omn-13118-test-broker"

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

    Mirrors ``HandlerContextRoiRunner._open_terminal_session``: a failed open
    (e.g. the FAILED topic's assign loop burning the whole cap and raising a
    bare ``TimeoutError``) is caught, and the runner falls back to the legacy
    single-call seek-at-wait path for that topic.
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
) -> tuple[dict[str, object] | None, float]:
    """Drive ONE trial as ``HandlerContextRoiRunner._run_trial`` does; time it.

    Returns ``(result, elapsed_seconds)``.

    Ordering (subscribe-before-publish, OMN-13012/13038):
      1. open the COMPLETED terminal session to current end;
      2. open the FAILED terminal session to current end — for a never-written
         FAILED topic this is where the wedge lives: the assign loop never sees
         partitions and burns the whole assign cap before raising;
      3. "publish" the generation command — FAST generation appends the
         correlated COMPLETED terminal immediately (dispatch loop freed,
         OMN-13010, gen ~1s);
      4. wait on the COMPLETED session.

    The whole open->publish->wait is timed so the per-trial cap-burn stall the
    battery dies on is measured, not merely whether a row eventually appears.
    """
    started = time.monotonic()
    completed_session = _open_session_or_none(consumer_factory, _COMPLETED_TOPIC)
    failed_session = _open_session_or_none(consumer_factory, _FAILED_TOPIC)

    # FAST generation: the correlated COMPLETED terminal is published right
    # after the command (mirrors ~1s gen vs the per-trial consumer open).
    cluster.logs[_COMPLETED_TOPIC].append(
        correlation_id, _terminal_envelope_json(correlation_id)
    )

    try:
        if completed_session is None:
            return None, time.monotonic() - started
        result = completed_session.wait(correlation_id, timeout_seconds)
        return result, time.monotonic() - started
    finally:
        if failed_session is not None:
            failed_session.close()


@pytest.mark.timeout(180)
def test_full_battery_consume_leg_does_not_wedge_under_sustained_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K>=10 x 2 cells x 2 arms must each correlate; matrix must not wedge.

    Reproduces the verifier-confirmed battery wedge: the COMPLETED terminal IS
    published every trial (its HWM climbs) but the per-trial COMPLETED consumer
    skips it because ``seek_to_end`` runs AFTER the slow open window in which the
    fast terminal already landed. RED: most cells produce no correlated terminal
    (degenerate), the matrix never advances through both arms with real rows.
    GREEN: every cell across both arms correlates.
    """
    cluster = _Cluster()
    _FakeConsumer.cluster = cluster
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeConsumer,
    )
    # Scale the assign cap down so a partition-less FAILED open burns ~1.5s, not
    # the live 30s — the per-trial stall is the wedge; this keeps the test fast
    # while preserving the cap-burn behavior exactly.
    monkeypatch.setattr(
        service_pattern_b_broker,
        "_DIRECT_TERMINAL_ASSIGN_TIMEOUT_CAP_SECONDS",
        _SCALED_ASSIGN_CAP_SECONDS,
    )
    # Scale the partition-less grace down proportionally: a FIXED consume leg
    # returns the partition-less FAILED open after this grace, well under the
    # healthy-trial bound. (Pre-fix the constant is unused — the loop runs to the
    # full cap — so this is a no-op on the unfixed code and the wedge still REDs.)
    monkeypatch.setattr(
        service_pattern_b_broker,
        "_DIRECT_TERMINAL_PARTITIONLESS_GRACE_SECONDS",
        _SCALED_PARTITIONLESS_GRACE_SECONDS,
        raising=False,
    )

    bus = _KafkaLikeBus()
    consumer = make_terminal_event_consumer(event_bus=bus, handler_name=_HANDLER)

    # The runner's per-arm timeout (generation_timeout_seconds=120 in the
    # battery). Kept short here; a correctly-positioned consumer correlates well
    # under this, a wedged one burns its window then returns None.
    timeout_seconds = 8.0

    correlated_cells: list[tuple[str, str, int]] = []
    degenerate_cells: list[tuple[str, str, int]] = []
    slow_trials: list[tuple[str, str, int, float]] = []
    arms_with_real_rows: set[str] = set()
    cells_with_real_rows: set[str] = set()

    run_order = 0
    for cell in _CELLS:
        for trial in range(1, _TRIALS_PER_CELL + 1):
            for arm in _ARMS:
                run_order += 1
                cid = f"omn-13118-{cell}-{arm}-t{trial}"
                result, elapsed = _run_one_trial(
                    consumer_factory=consumer,
                    cluster=cluster,
                    correlation_id=cid,
                    timeout_seconds=timeout_seconds,
                )
                if elapsed > _MAX_HEALTHY_TRIAL_SECONDS:
                    slow_trials.append((cell, arm, trial, elapsed))
                if result is not None and str(result.get("correlation_id")) == cid:
                    correlated_cells.append((cell, arm, trial))
                    arms_with_real_rows.add(arm)
                    cells_with_real_rows.add(cell)
                else:
                    degenerate_cells.append((cell, arm, trial))

    total = len(_CELLS) * _TRIALS_PER_CELL * len(_ARMS)

    # (b) The COMPLETED-topic HWM CLIMBED (every trial published its terminal) —
    # confirms the wedge is NOT a missing-publish problem but a consume-leg
    # correlation/throughput failure, exactly as the verifier observed live.
    assert cluster.logs[_COMPLETED_TOPIC].hwm == total, (
        "every trial's correlated COMPLETED terminal must have been published "
        f"(HWM should climb to {total}, got {cluster.logs[_COMPLETED_TOPIC].hwm})"
    )
    assert cluster.logs[_FAILED_TOPIC].hwm == 0, (
        "the FAILED topic must stay at HWM 0 (nothing lands there) — the runner "
        "relies on the COMPLETED correlate, never a FAILED terminal"
    )

    # WEDGE GATE (the load-dependent fault the single-trial probes miss):
    # every trial blocks ~one full assign cap on the partition-less FAILED open.
    # A 160-trial battery at 30s/trial cannot finish in any usable window.
    assert not slow_trials, (
        f"{len(slow_trials)}/{total} trials each blocked > "
        f"{_MAX_HEALTHY_TRIAL_SECONDS}s — the FAILED reply topic (zero partitions, "
        "never produced to) burns the whole assign cap on EVERY trial before the "
        "COMPLETED terminal can be correlated. At the live 30s cap a 160-trial "
        "battery needs >80 min and never completes. First few slow: "
        f"{[(c, a, t, round(e, 2)) for c, a, t, e in slow_trials[:5]]}"
    )

    # (a) + (c) The matrix advances through BOTH arms past cell 1 with
    # NON-DEGENERATE rows for EVERY cell.
    assert not degenerate_cells, (
        f"{len(degenerate_cells)}/{total} trials produced a degenerate row — the "
        "per-trial consume leg failed to correlate the already-published "
        f"COMPLETED terminal. First few degenerate: {degenerate_cells[:5]}"
    )
    assert len(correlated_cells) == total, (
        f"expected all {total} trials correlated, got {len(correlated_cells)}"
    )
    assert cells_with_real_rows == set(_CELLS), (
        f"matrix did not advance through both cells with real rows: "
        f"{cells_with_real_rows} != {set(_CELLS)}"
    )
    assert arms_with_real_rows == set(_ARMS), (
        f"matrix did not advance through both arms with real rows: "
        f"{arms_with_real_rows} != {set(_ARMS)}"
    )
