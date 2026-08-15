# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The EFFECT adapter's own contract (OMN-15922).

New code, so it carries its own tests: in ``omnibase_core`` this adapter could
not exist (ADR-005 bans a transport import there), and every other test in this
directory drives the services against an in-memory fake of this seam. That fake
is only as honest as this file -- if the real adapter classified statuses, or
raised where the fake returns, the whole suite above it would be proving
something about a shape nothing implements.

The load-bearing claim here is the NEGATIVE one: a reached server's non-2xx is
returned, never raised. The services' fail-closed messages are built from the
status, and each names a different remediation for a 401 from Keycloak versus a
401 from the gateway; an adapter that raised would collapse both into one
transport error and take the status with it.

These run against ``httpx.MockTransport``, so real httpx request construction
(form encoding, header assembly, body bytes) is exercised with no socket.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.gateway.client import gateway_transport_httpx as module
from omnibase_infra.gateway.client.gateway_transport_httpx import (
    GatewayTransportHttpx,
)

pytestmark = pytest.mark.unit

_SECRET = "s3cr3t-not-a-real-value"  # pragma: allowlist secret
_URL = "https://keycloak.invalid/realms/acme/protocol/openid-connect/token"


def _install(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    seen: list[httpx.Request] | None = None,
) -> None:
    """Route the adapter's AsyncClient through an in-memory MockTransport."""

    def _record(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return handler(request)

    # Bind the real class BEFORE patching: ``module.httpx`` is the global httpx
    # module, so a factory that called ``httpx.AsyncClient`` would call itself.
    real_client = httpx.AsyncClient

    def _factory(**kwargs: object) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(_record))

    monkeypatch.setattr(module.httpx, "AsyncClient", _factory)


async def test_a_non_2xx_from_a_reached_server_is_returned_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contract's central negative: classification belongs to the caller."""
    _install(monkeypatch, lambda _: httpx.Response(401, text='{"error":"invalid"}'))

    response = await GatewayTransportHttpx().post_form(
        _URL, form={"grant_type": "client_credentials"}, headers={}
    )

    assert response.status == 401
    assert await response.text() == '{"error":"invalid"}'


async def test_a_server_that_was_never_reached_raises_instead_of_faking_a_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synthetic 503 would be indistinguishable from a real one."""

    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    _install(monkeypatch, _boom)

    with pytest.raises(InfraUnavailableError):
        await GatewayTransportHttpx().post_form(
            _URL, form={"client_secret": _SECRET}, headers={}
        )


async def test_an_unreachable_host_error_never_carries_the_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The form carries client_secret, so the error path must not interpolate it."""

    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    _install(monkeypatch, _boom)

    with pytest.raises(InfraUnavailableError) as caught:
        await GatewayTransportHttpx().post_form(
            _URL,
            form={"client_secret": _SECRET},
            headers={"Authorization": f"Bearer {_SECRET}"},
        )

    assert _SECRET not in str(caught.value)
    assert _SECRET not in repr(caught.value)


async def test_post_form_sends_url_encoded_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []
    _install(monkeypatch, lambda _: httpx.Response(200, text="{}"), seen=seen)

    await GatewayTransportHttpx().post_form(
        _URL,
        form={"grant_type": "client_credentials", "client_id": "ga-acme"},
        headers={"Accept": "application/json"},
    )

    request = seen[-1]
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    body = request.content.decode()
    assert "grant_type=client_credentials" in body
    assert "client_id=ga-acme" in body


async def test_post_json_sends_the_body_verbatim_and_keeps_the_caller_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller's serialization is the authority -- the adapter re-encodes nothing."""
    seen: list[httpx.Request] = []
    _install(monkeypatch, lambda _: httpx.Response(200, text="{}"), seen=seen)

    payload = '{"edge_instance_id":"test-edge"}'
    await GatewayTransportHttpx().post_json(
        "https://api.invalid/v1/gateway/attach",
        body=payload,
        headers={
            "Authorization": "Bearer token-value",
            "Content-Type": "application/json",
        },
    )

    request = seen[-1]
    assert request.content.decode() == payload
    assert request.headers["Authorization"] == "Bearer token-value"


async def test_the_response_body_is_read_before_the_client_context_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``text()`` must work after the AsyncClient that produced it is gone."""
    _install(monkeypatch, lambda _: httpx.Response(200, text='{"access_token":"x"}'))

    response = await GatewayTransportHttpx().post_json(
        "https://api.invalid/v1/gateway/attach", body="{}", headers={}
    )

    # The client context exited inside post_json; the body must still be here.
    assert await response.text() == '{"access_token":"x"}'
    assert await response.json() == {"access_token": "x"}
