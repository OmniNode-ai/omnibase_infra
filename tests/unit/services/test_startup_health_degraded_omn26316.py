# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316: /health answers while the runtime event loop is blocked.

Built from the recorded dev roll 1e15179f (fixture
``tests/fixtures/omn26316/dev_roll_1e15179f_startup_block.json``). That roll's
probe never got an answer within the healthcheck's HTTP timeout
(``probe_unreachable``) for the whole start period. A reproduction showed the
kernel's ``bootstrap`` coroutine running synchronous contract wiring on the
loop thread that serves ``/health``.

Here a subsystem holds the loop for longer than the probe's HTTP timeout and
never completes inside the probe window. The probe must still get an answer
inside that timeout, and well inside the start period. The answer is
``degraded`` with the typed reason ``event_loop_unresponsive``. The semantic
container healthcheck still FAILs that answer, so the gate is unchanged.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import socket
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.runtime.health.container_healthcheck import (
    evaluate_health_response,
)
from omnibase_infra.services.health_checker import ServiceHealth

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "omn26316"
    / "dev_roll_1e15179f_startup_block.json"
)


def _incident() -> dict[str, object]:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


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


class _Probe:
    """One GET /health from a thread, recording what came back and when."""

    def __init__(self, port: int, timeout_seconds: float) -> None:
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.status: int | None = None
        self.body: dict[str, object] | None = None
        self.error: str = ""
        self.elapsed: float = -1.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        started = time.monotonic()
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=self.timeout_seconds
        )
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            self.status = response.status
            self.body = json.loads(response.read())
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()
        self.elapsed = time.monotonic() - started


async def _started_server(port: int) -> ServiceHealth:
    server = ServiceHealth(
        runtime=_healthy_runtime(), port=port, host="127.0.0.1", version="t"
    )
    await server.start()
    # Let the server settle so the first probe sees a responsive loop.
    await asyncio.sleep(0.5)
    return server


async def test_startup_health_degraded_answers_while_loop_is_blocked() -> None:
    incident = _incident()
    job = incident["job_record"]
    healthcheck = incident["dev_healthcheck"]
    assert isinstance(job, dict) and isinstance(healthcheck, dict)
    assert "probe_unreachable" in str(job["last_probe_output"])
    probe_timeout = float(healthcheck["probe_http_timeout_seconds"])
    start_period = float(healthcheck["start_period_seconds"])

    port = _free_port()
    server = await _started_server(port)
    try:
        probe = _Probe(port, probe_timeout)
        probe.thread.start()
        # The slow subsystem: synchronous work on the loop thread that does not
        # finish inside the probe's HTTP timeout.
        time.sleep(probe_timeout + 2.0)
        probe.thread.join(timeout=probe_timeout + 5.0)
    finally:
        await server.stop()

    assert probe.error == "", (
        f"/health gave no answer within {probe_timeout}s while the loop was "
        f"blocked ({probe.error}); this is the probe_unreachable of the incident"
    )
    assert probe.elapsed < probe_timeout
    assert probe.elapsed < start_period
    assert probe.status == 503
    assert probe.body is not None
    assert probe.body["status"] == "degraded"
    details = probe.body["details"]
    assert isinstance(details, dict)
    assert details["degraded_reason"] == "event_loop_unresponsive"
    assert float(details["event_loop_stalled_seconds"]) > 0.0
    verdict = evaluate_health_response(http_status=probe.status, payload=probe.body)
    assert verdict.verdict == "FAIL"


async def test_startup_health_degraded_reason_absent_when_loop_responsive() -> None:
    port = _free_port()
    server = await _started_server(port)
    try:
        probe = _Probe(port, 5.0)
        probe.thread.start()
        await asyncio.to_thread(probe.thread.join, 10.0)
    finally:
        await server.stop()

    assert probe.error == ""
    assert probe.status == 200
    assert probe.body is not None
    assert probe.body["status"] == "healthy"
    details = probe.body["details"]
    assert isinstance(details, dict)
    assert "degraded_reason" not in details
