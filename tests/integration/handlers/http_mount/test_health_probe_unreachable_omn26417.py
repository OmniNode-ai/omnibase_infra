# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26417: concurrent /health probes while the runtime loop is blocked.

The dev container healthcheck (``python /usr/local/bin/onex-container-healthcheck
--degraded-policy fail``, container_healthcheck.py:DEFAULT_HEALTH_URL) calls
``http://localhost:8085/health`` inside the container with a 5 s HTTP timeout and
reports ``probe_unreachable`` when it reads no answer (container_healthcheck.py:193).

``HealthProbeAcceptor`` (OMN-26316) answers a stalled-loop probe with a typed 503
after its 4 s wait budget. It routed every connection on one accept thread, so a
second probe that arrived while the first was waiting for its budget sat in the
listen backlog until the first was answered: it was served at roughly twice the
budget, past the 5 s probe timeout, and read as ``probe_unreachable`` while the
runtime was alive. Each probe must get its typed answer inside the probe timeout
however many are in flight.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.runtime.health.container_healthcheck import (
    evaluate_health_response,
    fetch_health,
)
from omnibase_infra.services.health_checker import ServiceHealth

pytestmark = pytest.mark.integration

# container_healthcheck._DEFAULT_TIMEOUT_SECONDS: the probe's own HTTP timeout.
_PROBE_TIMEOUT_SECONDS = 5.0
_CONCURRENT_PROBES = 3


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _healthy_runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.health_check = AsyncMock(
        return_value={
            "healthy": True,
            "degraded": False,
            "is_running": True,
            "event_bus_healthy": True,
        }
    )
    return runtime


class _ContainerProbe:
    """The container healthcheck's own read, from a thread: fetch_health + verdict."""

    def __init__(self, port: int) -> None:
        self.url = f"http://127.0.0.1:{port}/health"
        self.reason = ""
        self.http_status: int | None = None
        self.elapsed = -1.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        started = time.monotonic()
        self.http_status, payload = fetch_health(self.url, _PROBE_TIMEOUT_SECONDS)
        self.elapsed = time.monotonic() - started
        self.reason = evaluate_health_response(
            http_status=self.http_status, payload=payload
        ).reason


async def test_probe_unreachable_not_read_by_concurrent_probes_while_loop_blocked() -> (
    None
):
    port = _free_port()
    server = ServiceHealth(
        runtime=_healthy_runtime(), port=port, host="127.0.0.1", version="t"
    )
    await server.start()
    await asyncio.sleep(0.5)
    probes = [_ContainerProbe(port) for _ in range(_CONCURRENT_PROBES)]
    try:
        for probe in probes:
            probe.thread.start()
        # Synchronous work on the loop thread that outlasts every probe window.
        time.sleep(_PROBE_TIMEOUT_SECONDS * 2 + 2.0)
        for probe in probes:
            probe.thread.join(timeout=_PROBE_TIMEOUT_SECONDS + 5.0)
    finally:
        await server.stop()

    readings = [(p.reason, p.http_status, round(p.elapsed, 2)) for p in probes]
    assert all(p.reason != "probe_unreachable" for p in probes), (
        f"a probe read probe_unreachable while the loop was blocked: {readings}"
    )
    assert all(p.http_status == 503 for p in probes), readings
    assert all(p.elapsed < _PROBE_TIMEOUT_SECONDS for p in probes), readings


async def test_probe_unreachable_connections_past_inflight_cap_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection past the in-flight cap is closed, not queued behind a stalled wait."""
    from omnibase_infra.handlers.http_mount import health_probe_acceptor

    monkeypatch.setattr(health_probe_acceptor, "_MAX_INFLIGHT_CONNECTIONS", 1)
    port = _free_port()
    server = ServiceHealth(
        runtime=_healthy_runtime(), port=port, host="127.0.0.1", version="t"
    )
    await server.start()
    await asyncio.sleep(0.5)
    first = _ContainerProbe(port)
    second = _ContainerProbe(port)
    try:
        first.thread.start()
        await asyncio.sleep(0)  # let the first connect before the loop blocks
        time.sleep(0.3)
        second.thread.start()
        time.sleep(_PROBE_TIMEOUT_SECONDS + 1.5)
        first.thread.join(timeout=_PROBE_TIMEOUT_SECONDS + 5.0)
        second.thread.join(timeout=_PROBE_TIMEOUT_SECONDS + 5.0)
    finally:
        await server.stop()

    assert first.http_status == 503
    assert second.http_status is None
    assert second.elapsed < _PROBE_TIMEOUT_SECONDS
