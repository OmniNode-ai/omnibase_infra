# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21424 AC6 (unreachable store): check_fence answers unknown, never ok."""

from __future__ import annotations

import httpx
import pytest

from omnibase_infra.handlers.lease.handler_lease_store import check_fence
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import EnumLeaseFenceStatus

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        pytest.param(httpx.ConnectError, "connection refused", id="connect-error"),
        pytest.param(httpx.ReadTimeout, "timed out", id="read-timeout"),
    ],
)
async def test_ac6_unreachable_etcd_is_unknown(
    error_type: type[httpx.RequestError], message: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error_type(message, request=request)

    adapter = HandlerPlacementEtcd(
        "http://etcd.invalid:2379",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        status = await check_fence(adapter, kind="k", resource="r", fence=5)
    finally:
        await adapter.aclose()
    assert status is EnumLeaseFenceStatus.UNKNOWN
