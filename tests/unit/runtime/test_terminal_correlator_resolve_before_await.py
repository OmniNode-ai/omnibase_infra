# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Focused regression for the P_RESOLVE_BEFORE_AWAIT correlator race (OMN-13118).

The live wedge (and the defect this test pins) is a delivery-ordering race in
``LongLivedTerminalCorrelator``: the always-on poll loop runs continuously, so it
typically consumes and matches the correlated terminal BEFORE the runner's
``wait()`` attaches. The publish -> generation -> wait latency is large (~28s
observed live), so by the time the runner blocks on its future the terminal has
already been delivered.

Before the fix, ``_deliver`` popped the future on match. When delivery beat the
awaiter, the resolved future was removed from the registry, so the late ``_wait``
found nothing, created a fresh (unresolved) future, and timed out at the trial
cap -> ``None`` -> degenerate generation-failure row -> zero rows.

The fix: ``_deliver`` resolves the future IN PLACE (``get``, not ``pop``); the
awaiter (``_wait``) owns removal and pops on exit (``finally``). A future resolved
before the awaiter attaches therefore survives in the registry until ``_wait``
reads it.

These tests prove the resolve-before-await ordering deterministically: register
the cid, append the terminal, and wait (in the test) until the registry future is
``done()`` — i.e. ``_deliver`` has already resolved it IN PLACE — and ONLY THEN
call ``wait()``. The fix means ``wait()`` returns the body; the pre-fix
pop-on-match would have returned ``None``/timed out.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from omnibase_infra.runtime.service_terminal_event_consumer import (
    LongLivedTerminalCorrelator,
)

pytestmark = [pytest.mark.unit]

_TERMINAL_TOPIC = "onex.evt.omnimarket.node-generation-completed.v1"
_HANDLER = "HandlerContextRoiRunner"


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


class _FakeClient:
    def __init__(self) -> None:
        self.tracked: set[str] = set()

    async def _refresh(self) -> bool:
        return True

    def set_topics(self, topics: list[str]) -> Any:
        self.tracked = set(topics)
        return self._refresh()

    def force_metadata_update(self) -> Any:
        return self._refresh()


class _FakeConsumer:
    """Single long-lived ``AIOKafkaConsumer`` stand-in over one terminal topic."""

    log: ClassVar[_PartitionLog | None] = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._client = _FakeClient()
        self._assigned: set[Any] = set()
        self._cursor = 0

    async def start(self) -> None:
        return None

    def partitions_for_topic(self, topic: str) -> set[int]:
        self._client.tracked.add(topic)
        return {0}

    def assign(self, partitions: Any) -> None:
        self._assigned = set(partitions)

    def assignment(self) -> set[Any]:
        return set(self._assigned)

    async def seek_to_end(self, *assignment: object) -> None:
        return None

    async def end_offsets(self, partitions: Any) -> dict[Any, int]:
        assert type(self).log is not None
        return dict.fromkeys(partitions, type(self).log.hwm)

    def seek(self, partition: Any, offset: int) -> None:
        self._cursor = offset

    async def getone(self) -> SimpleNamespace:
        import asyncio

        assert type(self).log is not None
        while True:
            record = type(self).log.read_at(self._cursor)
            if record is not None:
                self._cursor += 1
                _cid, value = record
                return SimpleNamespace(topic=_TERMINAL_TOPIC, value=value)
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
    _bootstrap_servers = "omn-13118-resolve-before-await-broker"

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


def _await_future_resolved(
    correlator: LongLivedTerminalCorrelator,
    correlation_id: str,
    timeout_seconds: float = 5.0,
) -> bool:
    """Block until the registry future for ``correlation_id`` is ``done()``.

    Returning True proves ``_deliver`` matched the terminal and resolved the
    future IN PLACE *before* any ``wait()`` attached — the exact resolve-before-
    await ordering. The future is read on the worker loop (where it lives) via a
    thread-safe submit to avoid touching loop state from the test thread.
    """
    import asyncio

    deadline = time.monotonic() + timeout_seconds

    async def _is_done() -> bool:
        future = correlator._pending.get(correlation_id)
        return future is not None and future.done()

    while time.monotonic() < deadline:
        resolved = asyncio.run_coroutine_threadsafe(
            _is_done(), correlator._loop
        ).result(timeout=1.0)
        if resolved:
            return True
        time.sleep(0.005)
    return False


@pytest.mark.timeout(30)
def test_wait_returns_body_when_terminal_resolved_before_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """register -> deliver (resolved IN PLACE) -> THEN wait must return the body.

    This is the P_RESOLVE_BEFORE_AWAIT fix: the poll loop delivers and matches the
    terminal before ``wait()`` attaches. The test makes the ordering deterministic
    by blocking on the registry future being ``done()`` (proving ``_deliver``
    resolved it in place) BEFORE calling ``wait()``.

    Pre-fix (``_deliver`` popped on match): the resolved future was removed, the
    late ``wait()`` created a fresh future and timed out -> ``None``. The
    resolve-in-place fix keeps the resolved future in the registry until ``wait()``
    reads it, then ``_wait`` pops it on exit.
    """
    log = _PartitionLog()
    _FakeConsumer.log = log
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeConsumer,
    )

    correlator = LongLivedTerminalCorrelator(
        event_bus=_KafkaLikeBus(), handler_name=_HANDLER
    )
    cid = "omn-13118-resolve-before-await-0001"
    try:
        # 1. Subscribe + register the cid BEFORE the terminal exists (pre-publish).
        correlator.ensure_topic(_TERMINAL_TOPIC)
        correlator.register(cid)

        # 2. Deliver the terminal. The always-on poll loop consumes + matches it.
        log.append(cid, _terminal_envelope_json(cid))

        # 3. Block (in the test) until the registry future is resolved IN PLACE by
        #    _deliver — i.e. delivery has provably BEATEN the awaiter. This is the
        #    late-awaiter ordering the fix targets.
        assert _await_future_resolved(correlator, cid), (
            "the poll loop never resolved the registry future for the matched "
            "terminal — _deliver did not match/resolve in place"
        )

        # 4. ONLY NOW attach the awaiter. With the resolve-in-place fix the body
        #    is still in the registry; pre-fix pop-on-match would return None.
        result = correlator.wait(cid, timeout_seconds=2.0)
        assert result is not None, (
            "wait() returned None for a terminal that was delivered and resolved "
            "BEFORE the awaiter attached — the P_RESOLVE_BEFORE_AWAIT race "
            "(_deliver must resolve in place, not pop on match)"
        )
        assert str(result.get("correlation_id")) == cid

        # 5. The awaiter owns removal: the cid is popped from the registry on exit.
        import asyncio

        still_pending = asyncio.run_coroutine_threadsafe(
            _pending_contains(correlator, cid), correlator._loop
        ).result(timeout=1.0)
        assert not still_pending, (
            "_wait must pop the cid on exit so a resolved-and-read future does not "
            "linger in the registry"
        )
    finally:
        correlator.close()


@pytest.mark.timeout(30)
def test_wait_still_returns_body_when_terminal_arrives_during_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling ordering: terminal arrives DURING the wait — body still returned.

    Confirms the fix does not regress the in-window delivery path: the awaiter
    attaches first, then the terminal lands and the poll loop resolves the
    already-attached future.
    """
    log = _PartitionLog()
    _FakeConsumer.log = log
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _FakeConsumer,
    )

    correlator = LongLivedTerminalCorrelator(
        event_bus=_KafkaLikeBus(), handler_name=_HANDLER
    )
    cid = "omn-13118-resolve-during-await-0001"
    try:
        correlator.ensure_topic(_TERMINAL_TOPIC)
        correlator.register(cid)

        def _publish_during_wait() -> None:
            time.sleep(0.2)
            log.append(cid, _terminal_envelope_json(cid))

        threading.Thread(target=_publish_during_wait, daemon=True).start()
        result = correlator.wait(cid, timeout_seconds=5.0)
        assert result is not None, (
            "wait() returned None for a terminal that arrived during the await"
        )
        assert str(result.get("correlation_id")) == cid
    finally:
        correlator.close()


async def _pending_contains(
    correlator: LongLivedTerminalCorrelator, correlation_id: str
) -> bool:
    return correlation_id in correlator._pending
