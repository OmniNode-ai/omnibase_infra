# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for auto-wired handler event_consumer injection.

The dispatch INPUT crash was OMN-13003. OMN-13005 was the SECOND defect: a
request/response EFFECT handler declares an injectable blocking ``event_consumer``
(publish command -> block on correlated terminal event -> read result fields
back), but the runtime auto-wiring materialized no consumer, so the handler fell
back to a no-op default that returned ``None`` immediately and every row was a
degenerate generation-failure.

OMN-13118 Tier B replaced the per-trial ephemeral consumer with ONE long-lived
correlator (one consumer subscribed once to the terminal topics, a single poll
loop, demux by ``correlation_id``). These tests drive the REAL wiring path
(``_prepare_handler_wiring`` -> ``_materialize_known_handler_dependencies`` ->
``make_terminal_event_consumer`` -> ``LongLivedTerminalCorrelator``) rather than
constructing the handler directly, so they exercise the dispatch surface that the
handler-isolation golden chain never touches (memory
``feedback_real_dispatch_path_tests``). The Kafka layer is faked at the single
seam the correlator builds its consumer through
(``service_pattern_b_broker.AIOKafkaConsumer``) via a shared in-memory partition
log; the real correlator subscribes, runs its poll loop, and demuxes terminals to
the registered futures.

These assert the two surviving contracts:
  - OMN-13005: the injected consumer blocks-and-correlates (a delayed terminal is
    still returned, not a degenerate row); and
  - OMN-13012/13118: a terminal that lands in the open->wait gap is delivered (the
    correlator registers the cid before publish, so the always-running poll loop
    demuxes the gap-landed record).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
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
    """Event bus exposing the Kafka attributes the correlator's consumer needs."""

    config = SimpleNamespace(
        session_timeout_ms=45000,
        heartbeat_interval_ms=15000,
        max_poll_interval_ms=1800000,
        reconnect_backoff_ms=2000,
    )
    _bootstrap_servers = "omn-13118-injection-test-broker"

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes | None, bytes]] = []

    async def publish(self, topic: str, key: bytes | None, value: bytes) -> None:
        self.published.append((topic, key, value))

    def _build_auth_kwargs(self) -> dict[str, object]:
        return {}


# ----------------------------------------------------------------------------
# Fake Kafka partition log + single long-lived consumer
# ----------------------------------------------------------------------------


class _PartitionLog:
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


class FakeTerminalLog:
    """In-memory terminal log the correlator's single consumer reads from."""

    def __init__(self) -> None:
        self.partition = _PartitionLog()

    def append(self, correlation_id: str, payload: dict[str, Any]) -> None:
        self.partition.append(correlation_id, json.dumps(payload).encode("utf-8"))


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

    log: ClassVar[FakeTerminalLog | None] = None

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
        return dict.fromkeys(partitions, type(self).log.partition.hwm)

    def seek(self, partition: Any, offset: int) -> None:
        self._cursor = offset

    async def getone(self) -> SimpleNamespace:
        assert type(self).log is not None
        while True:
            record = type(self).log.partition.read_at(self._cursor)
            if record is not None:
                self._cursor += 1
                _cid, value = record
                return SimpleNamespace(topic=_TERMINAL_TOPIC, value=value)
            await asyncio.sleep(0.002)

    async def stop(self) -> None:
        return None


# ----------------------------------------------------------------------------
# Handler shape (two-phase open/register/wait — the only live shape)
# ----------------------------------------------------------------------------


class HandlerRoiRunnerTwoPhase:
    """Subscribe-before-publish shape: open + register, publish, then wait."""

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
        session = None
        opener = getattr(self._event_consumer, "open", None)
        if callable(opener):
            session = opener(_TERMINAL_TOPIC)
            register = getattr(session, "register", None)
            if callable(register):
                register(correlation_id)
        if self._event_publisher is not None:
            self._event_publisher(_COMMAND_TOPIC, b'{"task":"invoice"}')
        if self._TEST_LOG is not None:
            self._TEST_LOG.append(correlation_id, _terminal_payload(correlation_id))
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

    Patches the single Kafka seam the correlator builds its consumer through
    (``service_pattern_b_broker.AIOKafkaConsumer``) so the REAL
    ``TerminalEventConsumer`` / ``LongLivedTerminalCorrelator`` logic runs against
    the in-memory log.
    """
    contract = _make_roi_runner_contract()
    event_bus = RecordingEventBus()
    resolver = ServiceHandlerResolver()
    ownership_query = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )
    _FakeConsumer.log = log
    handler_cls._TEST_LOG = log  # type: ignore[attr-defined]

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
        ),
        patch(
            "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
            _FakeConsumer,
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
async def test_two_phase_consumer_catches_fast_terminal_in_gap() -> None:
    """OMN-13118: a terminal landing in the open->wait gap is demuxed and returned.

    The correlator registers the cid before publish, so the always-running poll
    loop delivers the gap-landed COMPLETED terminal to the registered future — a
    NON-degenerate row through the real wiring path.
    """
    log = FakeTerminalLog()
    rows = await _drive_handler(HandlerRoiRunnerTwoPhase, log)
    assert rows, "handler never emitted a result row"
    row = rows[-1]
    assert row["failure_stage"] != "generation", (
        "two-phase open/register/wait failed to catch the gap-landed terminal — "
        "the long-lived correlator did not demux it to the registered future"
    )
    assert row["attempt_count"] >= 1
    assert row["model_id"] == "Qwen3.6-35B-A3B"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auto_wired_event_consumer_blocks_and_returns_non_degenerate_row() -> (
    None
):
    """OMN-13005 preserved: a delayed terminal is still correlated and returned."""
    log = FakeTerminalLog()

    class _DelayedTerminalTwoPhase(HandlerRoiRunnerTwoPhase):
        async def handle(self, envelope: object) -> None:
            correlation_id = "cid-roi-runner-1"
            session = None
            opener = getattr(self._event_consumer, "open", None)
            if callable(opener):
                session = opener(_TERMINAL_TOPIC)
                register = getattr(session, "register", None)
                if callable(register):
                    register(correlation_id)
            if self._event_publisher is not None:
                self._event_publisher(_COMMAND_TOPIC, b'{"task":"invoice"}')

            active_log = self._TEST_LOG

            def _delayed_append() -> None:
                time.sleep(0.05)
                if active_log is not None:
                    active_log.append(correlation_id, _terminal_payload(correlation_id))

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
