# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection-runner dispatch: pool lifecycle + routing gate (OMN-16874/OMN-16875).

Two defects are pinned here, both read off the live `.201` dev lane on
2026-08-28 and both living in ``handler_wiring``:

1. **Pool lifecycle.** The runtime pre-connected a *handler-owned* async DB
   adapter with ``asyncio.run(db.connect())``. ``asyncio.run`` closes the loop
   it opened, so an asyncpg pool created there is bound to a dead loop before
   the handler's first message is even handled. Live effect:
   ``RuntimeError: Event loop is closed`` on every heartbeat carrying a window,
   34 occurrences, ``consumer_flow_windows`` stuck at 0 rows, DLQ climbing.
   A runtime cannot own the lifetime of a pool it did not create; the handler
   opens and closes it inside the loop that uses it.

2. **Routing gate.** ``_is_projection_runner_handler`` decided which of the two
   dispatch branches a handler took by testing whether its *class name* ended
   in ``ProjectionRunner``. ``ConsumerFlowProjectionWriter`` is named ``Writer``
   deliberately (the OMN-14350 ratchet rejects ``Runner``), so a naming
   constraint silently moved it onto a different branch. Dispatch shape is a
   declared capability, not a spelling.

The pool test drives **two consecutive messages**: a single-message test cannot
observe a loop-scoped pool defect, and that is exactly what let this ship.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDispatchSinks,
    _make_projection_dispatch_callback,
)
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)

_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"

DB_TARGET = projection_database_target("delegation_events")
SUBSCRIBE_TOPICS = ("onex.evt.platform.node-heartbeat.v1",)
TERMINAL_TOPIC = "onex.evt.omnimarket.projection-consumer-flow-applied.v1"


class _LoopBoundPool:
    """A pool that is only usable from the loop that created it.

    This is asyncpg's real constraint, reduced to the one property that
    matters: the connections are attached to a specific loop's transport, so
    reaching for them from a different loop raises. The live error text is
    reproduced verbatim so the test fails the way production failed.
    """

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.closed = False

    def use(self) -> None:
        if self.closed:
            raise RuntimeError("pool is closed")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("Event loop is closed")


class _SelfServedAdapter:
    """A handler-owned async DB adapter, shaped like ``AsyncpgAdapter``."""

    def __init__(self) -> None:
        self._pool: _LoopBoundPool | None = None
        # OMN-16911: the real adapter carries the DSN it will dial. The runtime
        # rebinds it to the topology-resolved workload identity before dispatch.
        self.dsn = "postgresql://unbound@host/db"

    async def connect(self) -> None:
        self._pool = _LoopBoundPool()

    async def close(self) -> None:
        if self._pool is not None:
            self._pool.closed = True
            self._pool = None

    async def execute(self, *_: Any) -> list[dict[str, Any]]:
        assert self._pool is not None, "call connect() first"
        self._pool.use()
        return [{"ok": True}]


class _InProcessProjectionWriter:
    """A runner-shaped handler that declares in-process runtime dispatch.

    It scopes its own pool to the single loop that uses it — connect, use,
    close, all inside one ``asyncio.run`` — which is the contract the runtime
    now requires of a handler that serves its own database.
    """

    onex_runtime_inprocess_dispatch = True

    def __init__(self) -> None:
        self.db = _SelfServedAdapter()
        self.writes = 0

    def bind_projection_database_url(self, dsn: str) -> None:
        # OMN-16911: declaring in-process dispatch obliges a handler that owns
        # a pool to accept the DSN the runtime resolved and privilege-proved.
        self.db.dsn = dsn

    @property
    def topics(self) -> list[str]:
        return list(SUBSCRIBE_TOPICS)

    async def run(self) -> None:  # standalone consume loop, unused here
        raise AssertionError("run() must not be driven by the runtime")

    async def project_event(self, topic: str, data: dict[str, Any], meta: Any) -> bool:
        # Guarded exactly like the real adapter's callers: a pool that already
        # exists is reused. That guard is why a pool minted by the runtime on a
        # throwaway loop is not repaired here — it is silently inherited.
        if self.db._pool is None:
            await self.db.connect()
        try:
            await self.db.execute("INSERT ...")
            self.writes += 1
            return True
        finally:
            await self.db.close()

    def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
        ok = asyncio.run(self.project_event("", input_data, None))
        return {"rows_upserted": 1 if ok else 0}


class _StandaloneProjectionRunner:
    """The same shape, WITHOUT the capability: deployed standalone."""

    def __init__(self) -> None:
        self.db = _SelfServedAdapter()
        self.dispatched = 0

    @property
    def topics(self) -> list[str]:
        return list(SUBSCRIBE_TOPICS)

    async def run(self) -> None:
        raise AssertionError("run() must not be driven by the runtime")

    async def project_event(self, topic: str, data: dict[str, Any], meta: Any) -> bool:
        return True

    def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.dispatched += 1
        return {"rows_upserted": 1}


def _envelope() -> MagicMock:
    envelope = MagicMock()
    envelope.topic = SUBSCRIBE_TOPICS[0]
    envelope.payload = {"flow_window": {"node_id": str(uuid.uuid4())}}
    envelope.correlation_id = uuid.uuid4()
    return envelope


def _drive(callback: Any, times: int) -> None:
    """Drive ``times`` consecutive messages through one dispatch callback."""

    async def _run() -> None:
        for _ in range(times):
            await callback(_envelope())

    with (
        patch(_PATCH_ENVIRON_GET, return_value="postgresql://fixture"),
        patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()),
    ):
        asyncio.run(_run())


@pytest.fixture(autouse=True)
def _configured_projection_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(monkeypatch)


@pytest.mark.unit
def test_two_consecutive_messages_both_write_through_a_handler_owned_pool() -> None:
    """The runtime must not bind a handler-owned pool to a throwaway loop.

    Before OMN-16874 the runtime called ``asyncio.run(db.connect())`` ahead of
    every dispatch. That created the pool on a loop which closed immediately,
    so the handler's own loop found a dead pool and every message raised
    ``Event loop is closed``. The handler here connects and closes inside the
    single loop it uses, so both messages must write.
    """
    handler = _InProcessProjectionWriter()
    callback = _make_projection_dispatch_callback(handler, DB_TARGET, SUBSCRIBE_TOPICS)
    _drive(callback, times=2)

    assert handler.writes == 2, (
        "both consecutive messages must complete a real write; a single-message "
        "test cannot observe a loop-scoped pool defect"
    )


@pytest.mark.unit
def test_runner_shaped_handler_without_the_capability_is_not_dispatched() -> None:
    """A standalone runner keeps its own consume loop — the runtime skips it.

    This is the branch every ``*ProjectionRunner`` sibling has always taken.
    It must keep taking it once the gate stops reading class names.
    """
    handler = _StandaloneProjectionRunner()
    callback = _make_projection_dispatch_callback(handler, DB_TARGET, SUBSCRIBE_TOPICS)
    _drive(callback, times=2)

    assert handler.dispatched == 0


@pytest.mark.unit
def test_dispatch_branch_is_decided_by_capability_not_by_class_name() -> None:
    """Renaming a handler must not silently move it between branches.

    ``ConsumerFlowProjectionWriter`` was named ``...Writer`` to satisfy the
    OMN-14350 type-word ratchet and thereby fell out of a name-suffix gate into
    the DB-injection branch. Two classes with identical shape and opposite
    names must land on the branch their declared capability names, not the one
    their spelling implies.
    """
    named_runner = _InProcessProjectionWriter()
    named_runner.__class__ = type(
        "SomethingProjectionRunner",
        (_InProcessProjectionWriter,),
        {},
    )
    callback = _make_projection_dispatch_callback(
        named_runner, DB_TARGET, SUBSCRIBE_TOPICS
    )
    _drive(callback, times=1)
    assert named_runner.writes == 1, (
        "a handler declaring in-process dispatch is dispatched even when its "
        "class name ends in ProjectionRunner"
    )

    named_writer = _StandaloneProjectionRunner()
    named_writer.__class__ = type(
        "SomethingProjectionWriter",
        (_StandaloneProjectionRunner,),
        {},
    )
    callback = _make_projection_dispatch_callback(
        named_writer, DB_TARGET, SUBSCRIBE_TOPICS
    )
    _drive(callback, times=1)
    assert named_writer.dispatched == 0, (
        "a handler that does NOT declare in-process dispatch is skipped even "
        "when its class name avoids the ProjectionRunner suffix"
    )


@pytest.mark.unit
def test_applied_event_carries_the_handler_result_not_a_hardcoded_literal() -> None:
    """OMN-16875: the applied event must carry the facts the handler produced.

    ``payload={"projected": True}`` was a literal in the emitter, so every
    bus-backed projection on the platform published a contentless ack and every
    consumer written against a real record failed validation. The ack key stays
    (existing Pattern-B consumers and golden chains read it); the handler's own
    result is what supplies the facts.
    """
    facts = {
        "rows_upserted": 1,
        "consumer_group": "local.omnimarket.node_registration_orchestrator",
        "topic": "onex.evt.platform.node-heartbeat.v1",
        "messages_in": 229150,
        "messages_out": 0,
        "messages_dlq": 0,
        "flow_state": "STALLED",
    }

    class _FactsHandler:
        def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
            return dict(facts)

    published: list[tuple[str, object, bytes]] = []

    class _FakeBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            published.append((topic, key, value))

    callback = _make_projection_dispatch_callback(
        _FactsHandler(),
        DB_TARGET,
        SUBSCRIBE_TOPICS,
        sinks=ProjectionDispatchSinks(
            event_bus=_FakeBus(), terminal_event=TERMINAL_TOPIC
        ),
    )
    _drive(callback, times=1)

    assert len(published) == 1
    payload = json.loads(published[0][2].decode("utf-8"))["payload"]
    for key, value in facts.items():
        assert payload[key] == value, f"applied event dropped {key!r}"
    assert payload["projected"] is True


@pytest.mark.unit
def test_applied_event_payload_is_json_safe() -> None:
    """A handler result carrying rich types still produces a wire payload.

    The emitter publishes bytes onto a topic; a value it cannot encode must
    degrade to a readable form rather than kill the terminal event and take
    the whole projection's observability with it.
    """
    from datetime import UTC, datetime

    class _RichHandler:
        def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
            return {
                "rows_upserted": 1,
                "window_start": datetime(2026, 8, 28, 18, 0, tzinfo=UTC),
                "node_id": uuid.UUID("11111111-2222-3333-4444-555555555555"),
            }

    published: list[tuple[str, object, bytes]] = []

    class _FakeBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            published.append((topic, key, value))

    callback = _make_projection_dispatch_callback(
        _RichHandler(),
        DB_TARGET,
        SUBSCRIBE_TOPICS,
        sinks=ProjectionDispatchSinks(
            event_bus=_FakeBus(), terminal_event=TERMINAL_TOPIC
        ),
    )
    _drive(callback, times=1)

    assert len(published) == 1
    payload = json.loads(published[0][2].decode("utf-8"))["payload"]
    assert payload["node_id"] == "11111111-2222-3333-4444-555555555555"
    assert payload["window_start"].startswith("2026-08-28T18:00:00")
    assert payload["projected"] is True
