# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26430: the stalled-loop /health answer names the frame that holds the loop.

The OMN-26316 acceptor answers ``/health`` with a typed 503 while a synchronous
call holds the runtime event loop, but the answer says only that the loop is
stalled and for how long. On the dev rollback image the loop was held for about
17 minutes and, once that container was gone, the call that held it could not be
named. A stalled-loop answer now carries the loop thread's innermost frames
(``loop_thread_stack``) so the holder is in the probe's own reading and in the
container log.

The 503 status, the ``event_loop_unresponsive`` reason and the wait budget are
unchanged; the semantic container healthcheck still FAILs this answer.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import socket
import threading
import time

import pytest

from omnibase_infra.handlers.http_mount.health_probe_acceptor import (
    HealthProbeAcceptor,
)
from omnibase_infra.runtime.health.container_healthcheck import (
    evaluate_health_response,
)

pytestmark = pytest.mark.unit

_WAIT_BUDGET_SECONDS = 0.3
_HOLD_SECONDS = 1.5


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _NeverServedProtocol(asyncio.Protocol):
    """A connection the loop would claim; the stalled loop never gets to."""


def _blocking_call_holding_the_loop() -> None:
    time.sleep(_HOLD_SECONDS)


def _read_health(port: int, result: dict[str, object]) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        conn.request("GET", "/health")
        response = conn.getresponse()
        result["status"] = response.status
        result["body"] = json.loads(response.read())
    finally:
        conn.close()


async def test_stalled_loop_answer_names_the_frame_holding_the_loop() -> None:
    port = _free_port()
    acceptor = HealthProbeAcceptor(
        host="127.0.0.1",
        port=port,
        version="t",
        protocol_factory=_NeverServedProtocol,
        wait_budget_seconds=_WAIT_BUDGET_SECONDS,
    )
    await acceptor.start()
    result: dict[str, object] = {}
    reader = threading.Thread(target=_read_health, args=(port, result), daemon=True)
    try:
        await asyncio.sleep(0.5)
        reader.start()
        # The held call: synchronous work on the loop thread that outlasts the
        # acceptor's wait budget, so the acceptor answers for the loop.
        _blocking_call_holding_the_loop()
        await asyncio.to_thread(reader.join, 5.0)
    finally:
        await acceptor.stop()

    assert result.get("status") == 503
    body = result["body"]
    assert isinstance(body, dict)
    details = body["details"]
    assert isinstance(details, dict)
    assert details["degraded_reason"] == "event_loop_unresponsive"
    stack = details["loop_thread_stack"]
    assert isinstance(stack, list)
    assert stack, "the stalled-loop answer carries no loop thread frame"
    assert "_blocking_call_holding_the_loop" in stack[0], stack
    assert len(stack) <= 12
    verdict = evaluate_health_response(
        http_status=int(str(result["status"])), payload=body
    )
    assert verdict.verdict == "FAIL"
