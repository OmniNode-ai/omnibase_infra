# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The deploy-agent readers cross a real socket (OMN-19811).

The unit suite for the chain-canary's deploy-awareness
(``test_handler_chain_canary_deploy_aware.py``) says so itself: "Every test
drives the real handler with injected transports, a fake clock and a fake
sleep. No network, no wall-clock waits." That suite, and
``test_handler_chain_canary_deploy_aware.py``'s own unit coverage of the pure
helpers in ``deploy_agent_window.py`` (``snapshot_from_payloads``,
``deploys_in_window``, ``queued_commands_from_payload``), never once calls
``read_deploy_agent_via_httpx`` or ``lane_ready_via_httpx`` themselves --
the two functions that actually open an HTTP connection. Nothing anywhere
proved that a real ``503`` body, a real non-JSON body, or a real closed
socket lands where the handler expects: ``readable=False`` with a sanitized
error, never an exception and never "no deploy".

This module drives those two functions against a real ``ThreadingHTTPServer``
on the loopback interface, the same shape
``test_llm_endpoint_probe_credential_omn19129.py`` uses for the same reason:
prove the wire behaviour the unit suite's injected transport cannot.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.deploy_agent_window import (
    lane_ready_via_httpx,
    read_deploy_agent_via_httpx,
)

pytestmark = pytest.mark.integration

_OBSERVED_AT = datetime(2026, 9, 27, 0, 0, tzinfo=UTC)
_JOB_ID = str(uuid4())
_ACTIVE_ID = str(uuid4())


class _Agent(BaseHTTPRequestHandler):
    """A minimal stand-in for the real deploy agent's HTTP surface."""

    #: set per test before the client fires, read by the handler methods
    health_body: bytes = b"{}"
    health_status: int = 200
    job_body: bytes = b"{}"
    job_status: int = 200
    queue_body: bytes | None = b"{}"
    queue_status: int = 200
    #: whether /queue answers at all, vs a closed connection
    queue_serves: bool = True
    seen_paths: list[str] = []

    def do_GET(self) -> None:
        type(self).seen_paths.append(self.path)
        if self.path == "/health":
            self._answer(self.health_status, self.health_body)
        elif self.path.startswith("/job/"):
            self._answer(self.job_status, self.job_body)
        elif self.path == "/queue":
            if not self.queue_serves:
                # A closed connection, not a body: read_deploy_agent_via_httpx
                # must treat this the same as any other unread /queue --
                # additive, never poisoning the rest of the snapshot.
                self.close_connection = True
                return
            self._answer(self.queue_status, self.queue_body or b"")
        else:
            self._answer(404, b"{}")

    def _answer(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def agent() -> Iterator[Callable[..., str]]:
    """Serve ``_Agent`` on an ephemeral loopback port; yield a base-URL factory.

    Each test configures the class attributes it needs before calling the
    factory, mirroring the module-under-test's own ``base = agent_url...``
    convention.
    """
    _Agent.health_body = b"{}"
    _Agent.health_status = 200
    _Agent.job_body = b"{}"
    _Agent.job_status = 200
    _Agent.queue_body = b"{}"
    _Agent.queue_status = 200
    _Agent.queue_serves = True
    _Agent.seen_paths = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Agent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield lambda: f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_a_healthy_agent_with_an_active_job_reads_as_a_real_snapshot(
    agent: Callable[[], str],
) -> None:
    _Agent.health_body = json.dumps(
        {
            "state": "deploying",
            "active_job": {
                "correlation_id": _ACTIVE_ID,
                "started_at": "2026-09-27T00:00:05+00:00",
            },
            "last_result": {"correlation_id": _JOB_ID},
        }
    ).encode()
    _Agent.job_body = json.dumps({"accepted_at": "2026-09-26T23:58:00+00:00"}).encode()
    _Agent.queue_body = json.dumps({"commands_ahead": 0}).encode()

    snapshot = await read_deploy_agent_via_httpx(agent(), 5.0, _OBSERVED_AT)

    assert snapshot.readable is True
    assert snapshot.state == "deploying"
    assert str(snapshot.active_correlation_id) == _ACTIVE_ID
    assert str(snapshot.last_correlation_id) == _JOB_ID
    assert snapshot.queued_commands == 0
    assert snapshot.busy is True
    assert "/health" in _Agent.seen_paths
    assert f"/job/{_JOB_ID}" in _Agent.seen_paths
    assert "/queue" in _Agent.seen_paths


@pytest.mark.asyncio
async def test_a_real_503_body_is_still_read_omn18636_ac4(
    agent: Callable[[], str],
) -> None:
    """``/health`` may answer 503 with a full body when the agent's accept
    backlog is unhealthy (OMN-18636 AC4); the body is read regardless of the
    real HTTP status the socket actually carried."""
    _Agent.health_status = 503
    _Agent.health_body = json.dumps({"state": "overloaded"}).encode()

    snapshot = await read_deploy_agent_via_httpx(agent(), 5.0, _OBSERVED_AT)

    assert snapshot.readable is True
    assert snapshot.state == "overloaded"


@pytest.mark.asyncio
async def test_a_real_non_json_body_is_unreadable_not_an_exception(
    agent: Callable[[], str],
) -> None:
    _Agent.health_body = b"not json at all"

    snapshot = await read_deploy_agent_via_httpx(agent(), 5.0, _OBSERVED_AT)

    assert snapshot.readable is False
    assert snapshot.error
    assert snapshot.active_correlation_id is None


@pytest.mark.asyncio
async def test_a_closed_queue_connection_is_additive_not_poisoning(
    agent: Callable[[], str],
) -> None:
    """``/queue`` failing over the real wire must not take down a readable
    ``/health`` snapshot -- it only leaves ``queued_commands`` unread."""
    _Agent.health_body = json.dumps({"state": "idle", "last_result": {}}).encode()
    _Agent.queue_serves = False

    snapshot = await read_deploy_agent_via_httpx(agent(), 5.0, _OBSERVED_AT)

    assert snapshot.readable is True
    assert snapshot.state == "idle"
    assert snapshot.queued_commands is None


@pytest.mark.asyncio
async def test_a_connection_refused_agent_is_unreadable_never_no_deploy() -> None:
    """No server on the port at all: a real connection refusal, not a mock."""
    snapshot = await read_deploy_agent_via_httpx(
        "http://127.0.0.1:1", 1.0, _OBSERVED_AT
    )

    assert snapshot.readable is False
    assert snapshot.error
    assert snapshot.active_correlation_id is None
    assert snapshot.last_correlation_id is None


@pytest.mark.asyncio
async def test_lane_ready_reads_a_real_200_as_ready(
    agent: Callable[[], str],
) -> None:
    assert await lane_ready_via_httpx(agent(), 5.0) is True


@pytest.mark.asyncio
async def test_lane_ready_reads_a_real_refusal_as_not_ready() -> None:
    assert await lane_ready_via_httpx("http://127.0.0.1:1", 1.0) is False
