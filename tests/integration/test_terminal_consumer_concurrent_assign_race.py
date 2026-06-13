# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-dispatch-path regression for the runner consume-leg wedge (OMN-13012).

Ticket: OMN-13012 (lineage OMN-13038, OMN-13099, OMN-13005, OMN-13058).
Surfaces the ``EXP1-3_RUNNER_CONSUME_LEG_BLOCKER`` class proven live on the
stability image ``c0505521f1fa``
(``docs/evidence/2026-06-12-weekend-pass/experiments/exp1/
HALT_RUNNER_CONSUME_LEG_REGRESSION.md`` and ``RUNNER_FIX_DIAGNOSIS.md``).

Why this lives in omnibase_infra, not omnimarket
-------------------------------------------------
A prior two-strike diagnosis proved the omnimarket handler
(``HandlerContextRoiRunner._await_first_terminal``) is correct: a faithful
handler-layer repro of the documented failure mode PASSES on dev. The wedge is
in the omnibase_infra runtime consume leg — the ``TerminalConsumerSession`` /
``AIOKafkaConsumer`` assign path. These tests therefore drive the REAL
``TerminalEventConsumer`` (the object the runtime injects as ``event_consumer``)
through the REAL ``open_direct_terminal_consumer`` /
``poll_direct_terminal_consumer`` code path, with ``AIOKafkaConsumer``
monkeypatched to a fake that faithfully models aiokafka 0.13.0 client
semantics — the same harness shape as
``test_pattern_b_broker_terminal_waiter.py``.

The live defect (per the halt diagnosis + ``runner-logs-tail.txt``)
-------------------------------------------------------------------
Per ~30s cycle the effects container emits EXACTLY:
  1. ONE ``terminal-event consumer: wait failed ... (topic=...failed.v1,
     cid=<new>):`` with an EMPTY exception message, and
  2. ONE ``aiokafka ... Updating subscribed topics to:
     frozenset({...failed...})`` — the FAILED topic ONLY.
The COMPLETED topic NEVER subscribes and never logs a delivery, so the
correlated COMPLETED terminal is never read and the matrix re-fires cell 1
forever.

Root mechanism modeled here: ``AIOKafkaClient.set_topics`` returns a metadata
future and only forces a metadata refresh when the requested topic set DIFFERS
from the currently-tracked set. ``_assign_direct_terminal_partitions`` ignores
the returned future and, on its second loop iteration, calls
``set_topics([topic])`` again with the SAME topic — taking the no-refresh
branch — so a consumer whose first metadata fetch has not yet surfaced
partitions never forces another fetch and burns the full assign cap, raising a
bare ``TimeoutError`` (empty message). With per-trial assign timing skewed
across the two ephemeral consumers, one terminal topic times out at the assign
cap while the other never delivers.

The defining assertions (RED before the fix, GREEN after)
---------------------------------------------------------
  - both terminal topics reliably subscribe (``set_topics`` recorded for each)
    and assign within the cap when broker metadata is merely slow, and
  - the correlated COMPLETED terminal is delivered to ``wait`` even when the
    FAILED-topic consumer's metadata is slow.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from omnibase_infra.runtime.service_terminal_event_consumer import (
    make_terminal_event_consumer,
)

pytestmark = [pytest.mark.integration]

_COMPLETED_TOPIC = "onex.evt.omnimarket.node-generation-completed.v1"
_FAILED_TOPIC = "onex.evt.omnimarket.node-generation-failed.v1"
_HANDLER = "HandlerContextRoiRunner"


class _FakeClient:
    """Models ``AIOKafkaClient`` metadata tracking for one ephemeral consumer.

    ``set_topics`` mirrors aiokafka 0.13.0: it only forces a metadata refresh
    when the requested topics differ from the tracked set; calling it again with
    the same set is a no-op. Partition metadata becomes available only after
    ``metadata_latency_calls`` *refresh-forcing* ``set_topics`` calls — modeling
    a broker whose topic metadata is slow to surface. A consumer that re-asks
    with the SAME topic set (the bug) never forces the additional refresh and so
    never sees partitions.
    """

    def __init__(self, metadata_latency_calls: int) -> None:
        self._tracked: set[str] = set()
        self._metadata_latency_calls = metadata_latency_calls
        self._forced_refreshes = 0
        self.set_topics_calls: list[tuple[str, ...]] = []

    def set_topics(self, topics: list[str]) -> bool:
        self.set_topics_calls.append(tuple(topics))
        requested = set(topics)
        # aiokafka: refresh only when the set actually changes.
        if not topics or requested.difference(self._tracked):
            self._forced_refreshes += 1
        self._tracked = requested
        return True

    def force_metadata_update(self) -> bool:
        # aiokafka: always fetches fresh metadata, regardless of tracked set.
        self._forced_refreshes += 1
        return True

    def partitions_ready(self) -> bool:
        return self._forced_refreshes >= self._metadata_latency_calls


class _FakeConsumer:
    """Faithful ``AIOKafkaConsumer`` stand-in for the real consume-leg path.

    Drives the production ``open_direct_terminal_consumer`` /
    ``poll_direct_terminal_consumer`` code unchanged: ``start`` -> client
    ``set_topics`` -> ``partitions_for_topic`` -> ``assign`` -> ``seek_to_end``
    -> ``getone``. Each constructed consumer registers itself keyed by the topic
    it is asked to track so the test can deliver a correlated terminal to it.
    """

    by_topic: ClassVar[dict[str, _FakeConsumer]] = {}
    metadata_latency_calls: ClassVar[int] = 1
    # Per-correlation terminal payloads the COMPLETED-topic consumer should
    # deliver from getone() once it has assigned (models the broker emitting the
    # correlated terminal in the publish->wait gap, loop-agnostic).
    completed_payloads: ClassVar[dict[str, bytes]] = {}

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.kwargs = kwargs
        self._client = _FakeClient(type(self).metadata_latency_calls)
        self._assigned: set[Any] = set()
        self.started = False
        self.stopped = False
        self._topic: str | None = None
        self._delivered: set[str] = set()

    async def start(self) -> None:
        self.started = True

    def partitions_for_topic(self, topic: str) -> set[int]:
        self._topic = topic
        if topic not in self._client._tracked:
            return set()
        if not self._client.partitions_ready():
            return set()
        type(self).by_topic[topic] = self
        return {0}

    def assign(self, partitions: set[Any]) -> None:
        self._assigned = set(partitions)

    def assignment(self) -> set[Any]:
        return self._assigned

    async def seek_to_end(self, *_assignment: object) -> None:
        return None

    async def getone(self) -> SimpleNamespace:
        # The COMPLETED-topic consumer delivers each pending correlated terminal
        # exactly once; otherwise (FAILED topic, or no pending payload) block
        # like a real broker with nothing correlated to deliver.
        if self._topic == _COMPLETED_TOPIC:
            for cid, value in type(self).completed_payloads.items():
                if cid not in self._delivered:
                    self._delivered.add(cid)
                    return SimpleNamespace(topic=self._topic, value=value)
        await asyncio.Event().wait()  # block until cancelled by timeout/stop
        raise AssertionError("unreachable")  # pragma: no cover

    async def stop(self) -> None:
        self.stopped = True


class _KafkaLikeBus:
    config = SimpleNamespace(
        session_timeout_ms=45000,
        heartbeat_interval_ms=15000,
        max_poll_interval_ms=1800000,
        reconnect_backoff_ms=2000,
    )
    _bootstrap_servers = "omn-13012-test-broker"

    def _build_auth_kwargs(self) -> dict[str, object]:
        return {}


def _terminal_envelope_json(correlation_id: str) -> bytes:
    import json

    return json.dumps(
        {
            "correlation_id": correlation_id,
            "payload": {
                "attempt_count": 1,
                "contract_passed": True,
                "model_id": "Qwen3.6-35B-A3B",
                "provider": "local",
            },
        }
    ).encode("utf-8")


@pytest.mark.timeout(60)
def test_both_terminal_consumers_assign_when_metadata_is_slow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent ephemeral sessions must each subscribe + assign + deliver.

    Models the live wedge: broker topic metadata is slow (needs more than one
    forced refresh). ``set_topics`` only forces a refresh when the topic set
    changes, so the assign loop's second ``set_topics([same_topic])`` does NOT
    re-fetch — a consumer that relies on re-asking with the same topic never
    sees partitions and times out at the assign cap with a bare ``TimeoutError``
    (the empty-message ``wait failed`` log), exactly as observed live.

    RED before the fix (the COMPLETED-topic ``wait`` returns None because its
    consumer never assigned / the correlated terminal is never read); GREEN
    after the consume-leg actually waits on the metadata future so both topics
    assign.
    """
    _FakeConsumer.by_topic = {}
    _FakeConsumer.completed_payloads = {}
    # Require TWO forced refreshes before partitions surface: the first
    # set_topics forces one; a consumer that re-asks with the SAME topic on the
    # next loop iteration does NOT force the second, so it never assigns.
    _FakeConsumer.metadata_latency_calls = 2
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeConsumer,
    )

    bus = _KafkaLikeBus()
    consumer = make_terminal_event_consumer(event_bus=bus, handler_name=_HANDLER)
    correlation_id = "omn-13012-cid-0001"

    # The correlated COMPLETED terminal lands in the publish->wait gap; the
    # assigned COMPLETED-topic consumer delivers it from getone().
    _FakeConsumer.completed_payloads = {
        correlation_id: _terminal_envelope_json(correlation_id)
    }

    # Subscribe-before-publish: open BOTH terminal topics (OMN-13038), as the
    # runner does.
    completed_session = consumer.open(_COMPLETED_TOPIC)
    failed_session = consumer.open(_FAILED_TOPIC)

    # Both topics must have subscribed (the live wedge subscribes ONLY failed).
    assert _COMPLETED_TOPIC in _FakeConsumer.by_topic, (
        "the COMPLETED-topic consumer never assigned partitions — it relied on "
        "re-asking set_topics with the same topic, which aiokafka does not "
        "re-fetch on; this is the live wedge (only the FAILED topic subscribes)"
    )
    assert _FAILED_TOPIC in _FakeConsumer.by_topic, (
        "the FAILED-topic consumer never assigned partitions"
    )

    result = completed_session.wait(correlation_id, timeout_seconds=10.0)
    failed_session.close()

    assert result is not None, (
        "the correlated COMPLETED terminal was never delivered to wait() — the "
        "consume leg wedged (EXP1-3_RUNNER_CONSUME_LEG_BLOCKER)"
    )
    assert str(result.get("correlation_id")) == correlation_id


@pytest.mark.timeout(60)
def test_no_terminal_consumer_thread_leak_across_repeated_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K>=2 trials over the real consumer must not leak worker threads.

    Each trial opens two sessions (completed + failed). Over repeated trials the
    session worker threads must be joined on close, or a 160-560 trial battery
    accumulates hundreds of leaked threads (OMN-13058 class). This asserts the
    consume leg is reusable across the multi-trial matrix the battery drives.
    """
    _FakeConsumer.by_topic = {}
    _FakeConsumer.completed_payloads = {}
    _FakeConsumer.metadata_latency_calls = 2
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeConsumer,
    )

    bus = _KafkaLikeBus()
    consumer = make_terminal_event_consumer(event_bus=bus, handler_name=_HANDLER)
    thread_name = f"terminal-consumer-{_HANDLER}"

    def _alive() -> int:
        return sum(
            1 for t in threading.enumerate() if t.name == thread_name and t.is_alive()
        )

    baseline = _alive()
    for trial in range(3):
        cid = f"omn-13012-cid-trial-{trial}"
        _FakeConsumer.by_topic = {}
        _FakeConsumer.completed_payloads = {cid: _terminal_envelope_json(cid)}
        completed_session = consumer.open(_COMPLETED_TOPIC)
        failed_session = consumer.open(_FAILED_TOPIC)
        result = completed_session.wait(cid, timeout_seconds=10.0)
        failed_session.close()
        assert result is not None, f"trial {trial}: correlated terminal not read"

    # Allow joined threads to fully exit.
    assert _alive() <= baseline, (
        f"worker threads leaked across trials (baseline={baseline}, alive={_alive()})"
    )
