# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The etcd adapter retains its original timeout API alongside client injection."""

from __future__ import annotations

import httpx
import pytest

from omnibase_infra.handlers.placement import HandlerPlacementEtcd

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("timeout_s", [None, 0.25])
async def test_original_constructor_owns_a_bounded_client(
    timeout_s: float | None,
) -> None:
    if timeout_s is None:
        adapter = HandlerPlacementEtcd("http://etcd.invalid:2379")
    else:
        adapter = HandlerPlacementEtcd("http://etcd.invalid:2379", timeout_s=timeout_s)
    client = adapter._client
    try:
        assert client.timeout == httpx.Timeout(1.0 if timeout_s is None else timeout_s)
    finally:
        await adapter.aclose()
    assert client.is_closed


async def test_injected_client_retains_auth_transport_and_timeout() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ID": "23", "TTL": "5"})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        headers={"Authorization": "Bearer test-token"},
        timeout=3.0,
    )
    adapter = HandlerPlacementEtcd("https://etcd.invalid:2379/", client, timeout_s=0.25)
    try:
        assert await adapter.lease_grant(5) == (23, 5)
        assert len(requests) == 1
        request = requests[0]
        assert str(request.url) == "https://etcd.invalid:2379/v3/lease/grant"
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.extensions["timeout"] == httpx.Timeout(3.0).as_dict()
    finally:
        await adapter.aclose()
    assert client.is_closed
