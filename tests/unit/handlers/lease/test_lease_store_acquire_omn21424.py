# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21424: an acquire whose transaction raises revokes the etcd lease it granted."""

from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.lease.handler_lease_store import HandlerLeaseStore
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import ModelLeaseAcquireRequest, ModelLeaseKindPolicy

pytestmark = pytest.mark.unit


async def test_acquire_revokes_the_etcd_lease_when_the_txn_raises() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/v3/lease/grant":
            return httpx.Response(200, json={"ID": "7", "TTL": "30"})
        if request.url.path == "/v3/kv/txn":
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json={})

    adapter = HandlerPlacementEtcd(
        "http://etcd.test", httpx.AsyncClient(transport=httpx.MockTransport(route))
    )
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="k", capacity=2)])
    try:
        with pytest.raises(InfraConnectionError):
            await store.acquire(
                ModelLeaseAcquireRequest(
                    kind="k", resource="r", holder="a", request_id=uuid4()
                )
            )
    finally:
        await adapter.aclose()
    assert [path for path, _ in calls] == [
        "/v3/kv/range",
        "/v3/lease/grant",
        "/v3/kv/txn",
        "/v3/lease/revoke",
    ]
    assert calls[-1][1] == {"ID": "7"}
