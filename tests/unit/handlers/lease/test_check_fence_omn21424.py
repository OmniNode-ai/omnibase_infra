# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21424 AC6 (unreachable store): check_fence answers unknown, never ok."""

from __future__ import annotations

import httpx
import pytest

from omnibase_infra.handlers.lease.handler_lease_fence import check_fence
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import EnumLeaseFenceStatus

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("transport_error", [httpx.ConnectError, httpx.ReadTimeout])
async def test_ac6_unreachable_etcd_is_unknown(
    transport_error: type[httpx.TransportError], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unexpected_network(*args: object, **kwargs: object) -> httpx.Response:
        pytest.fail("unit tests must use an injected transport")

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", unexpected_network
    )

    def fail_request(request: httpx.Request) -> httpx.Response:
        raise transport_error("etcd unavailable", request=request)

    adapter = HandlerPlacementEtcd(
        "http://etcd.invalid",
        httpx.AsyncClient(transport=httpx.MockTransport(fail_request)),
    )
    try:
        status = await check_fence(adapter, kind="k", resource="r", fence=5)
    finally:
        await adapter.aclose()
    assert status is EnumLeaseFenceStatus.UNKNOWN
