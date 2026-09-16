# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18421: the canary's gateway submission, over a REAL HTTP transport.

WHY THIS EXISTS BESIDE THE UNIT MODULE
--------------------------------------
``tests/unit/nodes/node_chain_canary_effect/test_handler_chain_canary_tenant_route_omn18421.py``
injects the submission transport, so it proves the ROUTING decision: that the
authenticated route is chosen, that the tenant-less one is never called behind
its back, and that a missing credential is red rather than a silent fallback.
What it cannot prove is the one thing an injected callable always stubs — that
the real function puts the credential in a header, reads the status code the
gateway actually returns, and turns each of those answers into the shape the
rest of the handler reads.

That gap is exactly where this class of defect lives. The whole reason this
canary reported ``PROBE-GREEN`` for weeks on a chain that projected nothing is
that a claim was read off a response instead of off the thing it described. A
transport tested only against a stub of itself repeats that shape one layer
down.

So these cases run ``_post_workflow_via_gateway`` against a real socket, with a
real server on the other end answering the way the gateway answers, and assert
on what came out the far side rather than on what was handed in.

NO NETWORK LEAVES THE HOST. The server binds ``127.0.0.1`` on an
ephemeral port, is torn down in a fixture, and speaks only to this test.
"""

from __future__ import annotations

import json
import socket
import typing as t

import pytest
from aiohttp import web

from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    _post_workflow_via_gateway,
)

pytestmark = pytest.mark.integration

_KEY = "onex_test_key_never_a_real_credential"


class _RecordingServer:
    """A stand-in gateway that records what it was sent and answers as told."""

    def __init__(self, status: int, body: object, *, raw: str | None = None) -> None:
        self.status = status
        self.body = body
        self.raw = raw
        self.seen_headers: dict[str, str] = {}
        self.seen_body: dict[str, t.Any] = {}
        self.seen_path = ""
        self.request_count = 0

    async def handle(self, request: web.Request) -> web.Response:
        self.request_count += 1
        self.seen_path = request.path
        self.seen_headers = dict(request.headers)
        try:
            self.seen_body = await request.json()
        except Exception:  # noqa: BLE001 - a malformed body is itself a finding
            self.seen_body = {}
        if self.raw is not None:
            return web.Response(
                status=self.status, text=self.raw, content_type="text/plain"
            )
        return web.json_response(self.body, status=self.status)


@pytest.fixture
async def gateway() -> t.AsyncIterator[t.Callable[..., t.Awaitable[t.Any]]]:
    """Start one loopback server per case and hand back a submit helper."""
    runners: list[web.AppRunner] = []

    async def start(
        status: int, body: object, *, raw: str | None = None
    ) -> tuple[_RecordingServer, str]:
        server = _RecordingServer(status, body, raw=raw)
        app = web.Application()
        app.router.add_post("/v1/workflows", server.handle)
        runner = web.AppRunner(app)
        await runner.setup()
        runners.append(runner)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        return server, f"http://127.0.0.1:{port}/v1/workflows"

    yield start  # type: ignore[misc]

    for runner in runners:
        await runner.cleanup()


_BODY: dict[str, object] = {
    "workflow_type": "delegate-skill",
    "correlation_id": "6f1b0f1e-0000-4000-8000-00000000abcd",
    "payload": {
        "prompt": "Reply with the single word: alive.",
        "task_type": "test",
        "source": "external-client",
        "max_tokens": 32,
    },
}


@pytest.mark.asyncio
async def test_the_credential_travels_in_a_header_and_not_in_the_body(
    gateway: t.Any,
) -> None:
    """The load-bearing one.

    A key in the submission body would be serialised into the node payload,
    onto the bus and into the event log. This asserts against what the SERVER
    received, which is the only place the question can be settled.
    """
    server, url = await gateway(202, {"workflow_id": "w-1", "correlation_id": "c-1"})
    await _post_workflow_via_gateway(url, dict(_BODY), _KEY, 10.0)

    assert server.seen_headers.get("X-API-Key") == _KEY
    assert _KEY not in json.dumps(server.seen_body)
    assert server.seen_path == "/v1/workflows"


@pytest.mark.asyncio
async def test_the_submission_body_arrives_in_the_gateways_own_shape(
    gateway: t.Any,
) -> None:
    """``ModelWorkflowSubmitRequest`` is ``extra="forbid"`` and the catalog
    schema is ``additionalProperties: false``. A ``/skill``-shaped key here is a
    400, not a field the gateway ignores, so the shape is asserted on arrival."""
    server, url = await gateway(202, {"workflow_id": "w-2", "correlation_id": "c-2"})
    await _post_workflow_via_gateway(url, dict(_BODY), _KEY, 10.0)

    assert set(server.seen_body) == {"workflow_type", "correlation_id", "payload"}
    assert server.seen_body["workflow_type"] == "delegate-skill"
    assert set(server.seen_body["payload"]) == {
        "prompt",
        "task_type",
        "source",
        "max_tokens",
    }


@pytest.mark.asyncio
async def test_a_202_is_accepted_and_claims_no_terminal(gateway: t.Any) -> None:
    """The ack is an ack. A 202 says the envelope was published and nothing
    about whether the chain terminalized, and the gateway's publish path is
    fail-open on an absent topic — so a terminal key here would manufacture
    exactly the claim-by-the-request-path that OMN-16931 removed."""
    _server, url = await gateway(
        202, {"workflow_id": "w-3", "correlation_id": "c-3", "status": "published"}
    )
    response, error, elapsed_ms = await _post_workflow_via_gateway(
        url, dict(_BODY), _KEY, 10.0
    )

    assert error == ""
    assert response is not None
    assert response["ok"] is True
    assert "terminal_event" not in response
    assert response["gateway_workflow_id"] == "w-3"
    assert elapsed_ms >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, {"detail": "unknown workflow_type"}),
        (400, {"detail": "workflow type is fenced", "fenced": True}),
        (401, {"detail": "invalid api key"}),
        (403, {"detail": "tenant not entitled"}),
        (429, {"detail": "workflow rate limit exceeded"}),
    ],
)
async def test_a_refusal_is_an_answer_carrying_its_reason(
    gateway: t.Any, status: int, body: dict[str, object]
) -> None:
    """A refusal is not a transport failure, and collapsing the two would send a
    reader to the lane when the answer is in how the canary was wired. The typed
    reason the gateway returns rides into the receipt."""
    _server, url = await gateway(status, body)
    response, error, _elapsed = await _post_workflow_via_gateway(
        url, dict(_BODY), _KEY, 10.0
    )

    assert error == ""
    assert response is not None
    assert response["ok"] is False
    assert response["error"]["code"] == f"gateway_http_{status}"  # type: ignore[index]
    assert str(body["detail"]) in response["error"]["message"]  # type: ignore[index]


@pytest.mark.asyncio
async def test_a_non_json_body_is_reported_rather_than_guessed(
    gateway: t.Any,
) -> None:
    """A gateway behind a proxy answering HTML is a real failure mode, and
    ``response.json()`` raising into a bare except is how it becomes an
    unattributable red."""
    _server, url = await gateway(502, None, raw="<html>bad gateway</html>")
    response, error, _elapsed = await _post_workflow_via_gateway(
        url, dict(_BODY), _KEY, 10.0
    )

    assert response is None
    assert "502" in error
    assert "non-JSON" in error


@pytest.mark.asyncio
async def test_an_unreachable_gateway_is_a_transport_error_not_a_refusal(
    gateway: t.Any,
) -> None:
    """The negative control for the refusal cases above.

    A closed port must NOT come back as ``ok: False`` with a gateway reason:
    that would tell a reader the gateway answered when nothing did, and send
    them to the catalog when the answer is that the lane is down.

    The closed port is obtained by binding a real socket and releasing it, so
    it is genuinely closed on this host rather than a guessed number. The
    positive control immediately above it is a live server on the same
    loopback address answering 202 — without that, a transport error here
    would also be what you would see if the client were simply broken.
    """
    server, live_url = await gateway(202, {"workflow_id": "w-4"})
    live_response, live_error, _ = await _post_workflow_via_gateway(
        live_url, dict(_BODY), _KEY, 5.0
    )
    assert server.request_count == 1
    assert live_error == ""
    assert live_response is not None and live_response["ok"] is True

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    dead_url = f"http://127.0.0.1:{closed_port}/v1/workflows"

    response, error, _elapsed = await _post_workflow_via_gateway(
        dead_url, dict(_BODY), _KEY, 5.0
    )
    assert response is None, "a closed port must not read as a gateway refusal"
    assert error
