# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for auto-wired handler event_consumer injection.

The dispatch INPUT crash was OMN-13003. OMN-13005 was the SECOND defect: a
request/response EFFECT handler declares an injectable blocking ``event_consumer``
(publish command -> block on correlated terminal event -> read result fields
back), but the runtime auto-wiring materialized no consumer, so the handler fell
back to a no-op default that returned ``None`` immediately and every row was a
degenerate generation-failure.

OMN-13012 is the FOURTH defect, exposed once OMN-13010 freed the dispatch loop and
generation began completing in ~1s: the OMN-13005 consumer is a single callable
that does ``assign -> seek_to_end -> poll`` AFTER the handler has already
published, so a terminal emitted in the publish->seek gap is seeked PAST and lost
(probe3). The fix makes the injected consumer two-phase — ``open(topic)`` assigns
and seeks to end NOW (before publish), ``session.wait(cid, timeout)`` blocks from
that captured position (after publish) — so a fast terminal is not skipped.

These tests drive the REAL wiring path (``_prepare_handler_wiring`` ->
``_materialize_known_handler_dependencies``) rather than constructing the handler
directly, so they exercise the dispatch surface that the handler-isolation golden
chain never touches (memory ``feedback_real_dispatch_path_tests``). The Kafka
layer is faked at the two service seams (``_open_positioned_consumer`` and
``_poll_correlated_terminal``) via a shared in-memory terminal log that models
seek-to-end semantics: a poll only sees messages at or after the offset captured
when its consumer was opened.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _prepare_handler_wiring
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)

_TERMINAL_TOPIC = "onex.evt.omnimarket.node-generation-completed.v1"
_COMMAND_TOPIC = "onex.cmd.omnimarket.node-generation-requested.v1"


class RecordingEventBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes | None, bytes]] = []

    async def publish(self, topic: str, key: bytes | None, value: bytes) -> None:
        self.published.append((topic, key, value))


# ----------------------------------------------------------------------------
# Fake Kafka terminal log modeling seek-to-end semantics
# ----------------------------------------------------------------------------


class FakeTerminalLog:
    """Append-only terminal-event log with seek-to-end offset capture.

    Models the exact race surface: a consumer opened at offset N never sees
    messages appended before N (``seek_to_end`` skipped past them). A consumer
    is a thin handle holding its captured start offset; ``poll`` returns the
    first message at index >= start_offset whose ``correlation_id`` matches,
    waiting up to ``timeout`` for one to be appended.
    """

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []
        self._cond = threading.Condition()

    def append(self, payload: dict[str, Any]) -> None:
        with self._cond:
            self._messages.append(payload)
            self._cond.notify_all()

    def current_end(self) -> int:
        with self._cond:
            return len(self._messages)

    def poll(
        self, start_offset: int, correlation_id: str, timeout_seconds: float
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_seconds
        with self._cond:
            idx = start_offset
            while True:
                while idx < len(self._messages):
                    msg = self._messages[idx]
                    idx += 1
                    if str(msg.get("correlation_id")) == correlation_id:
                        return msg
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)


class _FakeConsumerHandle:
    """Opaque handle returned by the patched ``_open_positioned_consumer``."""

    def __init__(self, log: FakeTerminalLog, start_offset: int) -> None:
        self.log = log
        self.start_offset = start_offset


def install_fake_kafka(log: FakeTerminalLog) -> tuple[Any, Any]:
    """Build patched replacements for the two service seams, bound to ``log``."""

    async def _fake_open(
        *, event_bus: object, terminal_topic: str
    ) -> _FakeConsumerHandle:
        # seek_to_end NOW: capture the current end offset.
        return _FakeConsumerHandle(log, log.current_end())

    async def _fake_poll(
        *,
        consumer: _FakeConsumerHandle,
        terminal_topic: str,
        correlation_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any] | None:
        # Poll from the offset captured at open; blocking wait runs off-thread.
        return await asyncio.to_thread(
            consumer.log.poll,
            consumer.start_offset,
            correlation_id,
            timeout_seconds,
        )

    return _fake_open, _fake_poll


# ----------------------------------------------------------------------------
# Handler shapes
# ----------------------------------------------------------------------------


class HandlerRoiRunnerSingleCall:
    """Legacy seek-after-publish shape: publish, THEN single-call consume.

    This is the OMN-13005 form. Against a fast terminal (appended immediately
    after publish) it loses the race because the consumer opens — and so seeks
    to end — only when the single call is made, i.e. after the terminal is
    already in the log.

    The active ``FakeTerminalLog`` is bound on the class (``_TEST_LOG``) by the
    test driver before dispatch; the constructor keeps the runtime-known
    ``event_publisher``/``event_consumer`` shape so the resolver introspects and
    materializes both injected deps exactly as it does for the real handler.
    """

    ROW_TOPIC = "onex.evt.omnimarket.context-roi-run-completed.v1"
    _TEST_LOG: FakeTerminalLog | None = None

    def __init__(
        self,
        event_publisher: Callable[[str, bytes], None] | None = None,
        event_consumer: Callable[[str, str, float], dict[str, Any] | None]
        | None = None,
    ) -> None:
        self._event_publisher = event_publisher
        self._event_consumer = event_consumer or (lambda _t, _c, _to: None)

    async def handle(self, envelope: object) -> None:
        correlation_id = "cid-roi-runner-1"
        if self._event_publisher is not None:
            self._event_publisher(_COMMAND_TOPIC, b'{"task":"invoice"}')
        # Fast terminal: appended the instant after publish, before the consume
        # call seeks to end.
        if self._TEST_LOG is not None:
            self._TEST_LOG.append(_terminal_payload(correlation_id))
        terminal = self._event_consumer(_TERMINAL_TOPIC, correlation_id, 5.0)
        _emit_row(self._event_publisher, self.ROW_TOPIC, terminal)


class HandlerRoiRunnerTwoPhase:
    """Subscribe-before-publish shape (OMN-13012): open, publish, then wait.

    Opens the terminal session (assign + seek_to_end NOW) BEFORE publishing, so
    a terminal appended immediately after publish is at-or-after the captured
    offset and is delivered by ``wait``.
    """

    ROW_TOPIC = "onex.evt.omnimarket.context-roi-run-completed.v1"
    _TEST_LOG: FakeTerminalLog | None = None

    def __init__(
        self,
        event_publisher: Callable[[str, bytes], None] | None = None,
        event_consumer: Any = None,
    ) -> None:
        self._event_publisher = event_publisher
        self._event_consumer = event_consumer

    async def handle(self, envelope: object) -> None:
        correlation_id = "cid-roi-runner-1"
        # Two-phase: open BEFORE publishing.
        session = None
        opener = getattr(self._event_consumer, "open", None)
        if callable(opener):
            session = opener(_TERMINAL_TOPIC)
        if self._event_publisher is not None:
            self._event_publisher(_COMMAND_TOPIC, b'{"task":"invoice"}')
        # Fast terminal: appended immediately after publish.
        if self._TEST_LOG is not None:
            self._TEST_LOG.append(_terminal_payload(correlation_id))
        if session is not None:
            terminal = session.wait(correlation_id, 5.0)
        elif self._event_consumer is not None:
            terminal = self._event_consumer(_TERMINAL_TOPIC, correlation_id, 5.0)
        else:
            terminal = None
        _emit_row(self._event_publisher, self.ROW_TOPIC, terminal)


def _terminal_payload(correlation_id: str) -> dict[str, Any]:
    return {
        "correlation_id": correlation_id,
        "attempt_count": 1,
        "model_id": "Qwen3.6-35B-A3B",
        "provider": "local",
        "contract_passed": True,
        "prompt_tokens": 10,
        "completion_tokens": 20,
    }


def _emit_row(
    publisher: Callable[[str, bytes], None] | None,
    row_topic: str,
    terminal: dict[str, Any] | None,
) -> None:
    if terminal is None:
        row = {"failure_stage": "generation", "attempt_count": 0, "model_id": ""}
    else:
        row = {
            "failure_stage": "none",
            "attempt_count": int(terminal["attempt_count"]),
            "model_id": str(terminal["model_id"]),
        }
    if publisher is not None:
        publisher(row_topic, json.dumps(row).encode("utf-8"))


def _make_roi_runner_contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_context_roi_runner",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_context_roi_runner",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.cmd.omnimarket.context-roi-run-requested.v1",),
            publish_topics=(
                "onex.evt.omnimarket.context-roi-run-completed.v1",
                _COMMAND_TOPIC,
            ),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerContextRoiRunner",
                        module="omnimarket.nodes.node_context_roi_runner.handlers.handler_context_roi_runner",
                    ),
                ),
            ),
        ),
    )


async def _drive_handler(
    handler_cls: type, log: FakeTerminalLog
) -> list[dict[str, Any]]:
    """Wire ``handler_cls`` through the real injection path and dispatch once.

    Binds the active ``FakeTerminalLog`` on the handler class (``_TEST_LOG``) so
    the constructor keeps the exact runtime-known dependency shape the resolver
    introspects, and patches the two Kafka seams so the real
    ``TerminalEventConsumer`` two-phase logic runs against the in-memory log.
    """
    contract = _make_roi_runner_contract()
    event_bus = RecordingEventBus()
    resolver = ServiceHandlerResolver()
    ownership_query = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )
    fake_open, fake_poll = install_fake_kafka(log)
    handler_cls._TEST_LOG = log  # type: ignore[attr-defined]

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
        ),
        patch(
            "omnibase_infra.runtime.service_terminal_event_consumer._open_positioned_consumer",
            side_effect=fake_open,
        ),
        patch(
            "omnibase_infra.runtime.service_terminal_event_consumer._poll_correlated_terminal",
            side_effect=fake_poll,
        ),
    ):
        prepared = _prepare_handler_wiring(
            contract=contract,
            entry=contract.handler_routing.handlers[0],  # type: ignore[union-attr]
            dispatch_engine=None,
            resolver=resolver,
            ownership_query=ownership_query,
            event_bus=event_bus,
            container=None,
        )

        await prepared.dispatcher(
            ModelEventEnvelope[dict[str, str]](
                payload={"run_id": "t1"},
                event_type="omnimarket.context-roi-run-requested",
            )
        )
        await asyncio.sleep(0)

    return [
        json.loads(value.decode("utf-8"))
        for topic, _key, value in event_bus.published
        if topic == handler_cls.ROW_TOPIC
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_single_call_consumer_loses_fast_terminal_race() -> None:
    """RED for OMN-13012: seek-after-publish misses a terminal emitted at t+0.

    The legacy single-call consumer opens (and seeks to end) only when the call
    is made — after the handler has already published AND the terminal has landed.
    seek_to_end skips past the terminal, so the poll times out and the row is
    degenerate. This is the exact probe3 failure; it MUST be red so the two-phase
    fix below cannot be a no-op.
    """
    log = FakeTerminalLog()
    rows = await _drive_handler(HandlerRoiRunnerSingleCall, log)
    assert rows, "handler never emitted a result row"
    row = rows[-1]
    assert row["failure_stage"] == "generation", (
        "single-call seek-after-publish unexpectedly caught the fast terminal — "
        "the race this ticket fixes is not being exercised"
    )
    assert row["attempt_count"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_phase_consumer_catches_fast_terminal_race() -> None:
    """GREEN for OMN-13012: open-before-publish captures the t+0 terminal.

    The two-phase session seeks to end BEFORE the handler publishes, so a terminal
    appended immediately after publish is at-or-after the captured offset and is
    delivered by ``wait`` — a NON-degenerate row. With the pre-fix single-call
    consumer this assertion fails (see the RED test above).
    """
    log = FakeTerminalLog()
    rows = await _drive_handler(HandlerRoiRunnerTwoPhase, log)
    assert rows, "handler never emitted a result row"
    row = rows[-1]
    assert row["failure_stage"] != "generation", (
        "two-phase open-before-publish failed to catch the fast terminal — the "
        "subscribe-after-publish race (OMN-13012) is not closed"
    )
    assert row["attempt_count"] >= 1
    assert row["model_id"] == "Qwen3.6-35B-A3B"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auto_wired_event_consumer_blocks_and_returns_non_degenerate_row() -> (
    None
):
    """OMN-13005 preserved: a delayed terminal is still correlated and returned.

    Drives the trial through the real wiring with the two-phase session and a
    terminal that arrives shortly after the wait begins (a delayed append rather
    than a t+0 one). Asserts a NON-DEGENERATE row, proving the injected consumer
    blocks-and-correlates (the OMN-13005 contract) under the new two-phase shape.
    """
    log = FakeTerminalLog()

    class _DelayedTerminalTwoPhase(HandlerRoiRunnerTwoPhase):
        async def handle(self, envelope: object) -> None:
            correlation_id = "cid-roi-runner-1"
            session = None
            opener = getattr(self._event_consumer, "open", None)
            if callable(opener):
                session = opener(_TERMINAL_TOPIC)
            if self._event_publisher is not None:
                self._event_publisher(_COMMAND_TOPIC, b'{"task":"invoice"}')

            active_log = self._TEST_LOG

            # Terminal arrives AFTER a short delay, while wait() is blocking.
            def _delayed_append() -> None:
                import time

                time.sleep(0.05)
                if active_log is not None:
                    active_log.append(_terminal_payload(correlation_id))

            threading.Thread(target=_delayed_append, daemon=True).start()
            terminal = (
                session.wait(correlation_id, 5.0) if session is not None else None
            )
            _emit_row(self._event_publisher, self.ROW_TOPIC, terminal)

    rows = await _drive_handler(_DelayedTerminalTwoPhase, log)
    assert rows, "handler never emitted a result row"
    row = rows[-1]
    assert row["failure_stage"] != "generation", (
        "event_consumer was not injected or did not block-correlate — the "
        "OMN-13005 degenerate-row defect"
    )
    assert row["attempt_count"] >= 1
    assert row["model_id"] == "Qwen3.6-35B-A3B"
