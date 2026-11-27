# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A handler that owns its DB adapter is never run by two threads at once (OMN-26430).

Read off the dev lane on 2026-11-23: two consecutive runtime containers
(``63d03d3afee8`` at 22:14:25Z and its predecessor at 22:04:19Z) died with
``python: segfault at 0 ... in protocol.cpython-312-x86_64-linux-gnu.so`` (the
asyncpg protocol extension, the same instruction offset both times) and no log
line. The line before the second death was ``asyncpg.exceptions.InterfaceError:
cannot perform operation: another operation is in progress`` out of a projection
writer whose ``project_request`` runs ``asyncio.run`` per message and opens and
closes ``self.db`` inside it.

The runtime dispatches a projection handler with ``asyncio.to_thread`` and lets
up to ``PROJECTION_HANDLER_MAX_INFLIGHT`` threads in at once. For a handler that
owns one adapter instance that is one adapter connected, used and closed by
several threads on several loops at the same time, which is the state asyncpg
crashes the whole process in. The handler's pool lifetime is per message and per
instance, so the runtime must hand it one message at a time.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
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
MESSAGES = 6
HOLD_SECONDS = 0.05


class _ConcurrencyMeter:
    """Counts how many threads are inside ``handle`` at the same moment."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inside = 0
        self.peak = 0
        self.entered = 0

    def hold(self) -> None:
        with self._lock:
            self._inside += 1
            self.entered += 1
            self.peak = max(self.peak, self._inside)
        time.sleep(HOLD_SECONDS)
        with self._lock:
            self._inside -= 1


class _SelfServedAdapter:
    """Shaped like ``AsyncpgAdapter``: it owns a pool and connects and closes it."""

    def __init__(self) -> None:
        self.dsn = "postgresql://unbound@host/db"

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None


class _DbOwningWriter(_ConcurrencyMeter):
    """In-process projection writer that opens its own adapter per message."""

    onex_runtime_inprocess_dispatch = True

    def __init__(self) -> None:
        super().__init__()
        self.db = _SelfServedAdapter()

    def bind_projection_database_url(self, dsn: str) -> None:
        self.db.dsn = dsn

    @property
    def topics(self) -> list[str]:
        return list(SUBSCRIBE_TOPICS)

    async def run(self) -> None:
        raise AssertionError("run() must not be driven by the runtime")

    async def project_event(self, topic: str, data: dict[str, Any], meta: Any) -> bool:
        return True

    def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.hold()
        return {"rows_upserted": 1}


class _InjectedDbHandler(_ConcurrencyMeter):
    """A handler on the DB-injection arm: it holds no adapter of its own."""

    def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.hold()
        return {"rows_upserted": 1}


def _envelope() -> MagicMock:
    envelope = MagicMock()
    envelope.topic = SUBSCRIBE_TOPICS[0]
    envelope.payload = {"flow_window": {"node_id": str(uuid.uuid4())}}
    envelope.correlation_id = uuid.uuid4()
    return envelope


def _drive_concurrently(callback: Any, messages: int) -> None:
    async def _run() -> None:
        await asyncio.gather(*(callback(_envelope()) for _ in range(messages)))

    with (
        patch(_PATCH_ENVIRON_GET, return_value="postgresql://fixture"),
        patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()),
    ):
        asyncio.run(_run())


@pytest.fixture(autouse=True)
def _configured_projection_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(monkeypatch)


@pytest.mark.unit
def test_db_owning_handler_is_never_inside_handle_on_two_threads() -> None:
    handler = _DbOwningWriter()
    callback = _make_projection_dispatch_callback(handler, DB_TARGET, SUBSCRIBE_TOPICS)

    _drive_concurrently(callback, MESSAGES)

    assert handler.entered == MESSAGES, "every message must still be handled"
    assert handler.peak == 1, (
        "a handler that owns its adapter connects, uses and closes it per "
        f"message; {handler.peak} threads were inside handle() together, which "
        "is the shared-pool state asyncpg segfaults the process in"
    )


@pytest.mark.unit
def test_injected_db_handler_keeps_its_parallelism() -> None:
    """Positive control: the serialisation is scoped to adapter owners.

    A handler that is handed its adapter per call shares nothing between
    threads, and the inflight gate (OMN-17888) already bounds it. Serialising
    it would throw away the throughput that gate was sized for.
    """
    handler = _InjectedDbHandler()
    callback = _make_projection_dispatch_callback(handler, DB_TARGET, SUBSCRIBE_TOPICS)

    _drive_concurrently(callback, MESSAGES)

    assert handler.entered == MESSAGES
    assert handler.peak > 1, (
        "the control must observe real thread overlap, or the test above "
        "proves nothing about serialisation"
    )
