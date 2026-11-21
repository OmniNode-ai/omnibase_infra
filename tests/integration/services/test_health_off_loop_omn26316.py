# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316: live /health socket answers while the runtime loop is blocked.

Boots the real ``ServiceHealth`` server on a real TCP port and probes it over
HTTP from a thread while synchronous work holds the event loop thread.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import socket
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.services.health_checker import ServiceHealth

pytestmark = pytest.mark.integration

_PROBE_TIMEOUT_SECONDS = 5.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def test_health_socket_answers_degraded_while_loop_blocked() -> None:
    runtime = MagicMock()
    runtime.health_check = AsyncMock(
        return_value={
            "healthy": True,
            "degraded": False,
            "is_running": True,
            "event_bus_healthy": True,
        }
    )
    port = _free_port()
    server = ServiceHealth(runtime=runtime, port=port, host="127.0.0.1", version="t")
    await server.start()
    await asyncio.sleep(0.5)

    result: dict[str, object] = {}

    def _probe() -> None:
        started = time.monotonic()
        conn = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=_PROBE_TIMEOUT_SECONDS
        )
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            result["status"] = response.status
            result["body"] = json.loads(response.read())
        except OSError as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()
        result["elapsed"] = time.monotonic() - started

    thread = threading.Thread(target=_probe, daemon=True)
    try:
        thread.start()
        time.sleep(_PROBE_TIMEOUT_SECONDS + 2.0)  # blocks the loop thread
        thread.join(timeout=_PROBE_TIMEOUT_SECONDS + 5.0)
    finally:
        await server.stop()

    assert "error" not in result, result.get("error")
    assert float(str(result["elapsed"])) < _PROBE_TIMEOUT_SECONDS
    assert result["status"] == 503
    body = result["body"]
    assert isinstance(body, dict)
    assert body["status"] == "degraded"
    assert body["details"]["degraded_reason"] == "event_loop_unresponsive"
