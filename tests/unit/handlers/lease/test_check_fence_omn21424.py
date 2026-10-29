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


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("no answer"),
    ],
)
async def test_ac6_unreachable_etcd_is_unknown(failure: httpx.TransportError) -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise failure

    async with httpx.AsyncClient(transport=httpx.MockTransport(unreachable)) as client:
        adapter = HandlerPlacementEtcd("http://etcd.test", client)
        status = await check_fence(adapter, kind="k", resource="r", fence=5)
    assert status is not EnumLeaseFenceStatus.OK
    assert status is EnumLeaseFenceStatus.UNKNOWN


@pytest.mark.parametrize(
    "entry",
    [
        {"key": "aw==", "value": "dg==", "create_revision": "bad", "lease": "1"},
        {"key": "aw==", "value": "dg==", "create_revision": "5", "lease": "bad"},
        {"key": "/w==", "value": "dg==", "create_revision": "5", "lease": "1"},
    ],
)
async def test_ac6_malformed_etcd_entry_is_unknown(entry: dict[str, str]) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"kvs": [entry]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter = HandlerPlacementEtcd("http://etcd.test", client)
        status = await check_fence(adapter, kind="k", resource="r", fence=5)
    assert status is EnumLeaseFenceStatus.UNKNOWN
