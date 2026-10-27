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


@pytest.mark.parametrize("revoke_fails", [False, True])
async def test_acquire_revokes_the_etcd_lease_when_the_txn_raises(
    revoke_fails: bool, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/v3/lease/grant":
            return httpx.Response(200, json={"ID": "7", "TTL": "30"})
        if request.url.path == "/v3/kv/txn":
            return httpx.Response(503, json={"error": "unavailable"})
        if request.url.path == "/v3/lease/revoke" and revoke_fails:
            return httpx.Response(503, json={"error": "revoke unavailable"})
        return httpx.Response(200, json={})

    adapter = HandlerPlacementEtcd(
        "http://etcd.test", httpx.AsyncClient(transport=httpx.MockTransport(route))
    )
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="k", capacity=2)])
    try:
        with pytest.raises(InfraConnectionError) as raised:
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
    # Cleanup failure must not replace the original transaction error.
    cause = raised.value.__cause__
    assert isinstance(cause, httpx.HTTPStatusError)
    assert cause.request.url.path == "/v3/kv/txn"
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == int(revoke_fails)
    if revoke_fails:
        assert "etcd lease 7 may remain until TTL" in warnings[0].getMessage()
        assert warnings[0].exc_info is not None


async def test_acquire_revokes_the_etcd_lease_when_every_slot_is_held() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/v3/lease/grant":
            return httpx.Response(200, json={"ID": "7", "TTL": "30"})
        if request.url.path == "/v3/kv/txn":
            return httpx.Response(200, json={"succeeded": False, "responses": []})
        return httpx.Response(200, json={})

    adapter = HandlerPlacementEtcd(
        "http://etcd.test", httpx.AsyncClient(transport=httpx.MockTransport(route))
    )
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="k", capacity=2)])
    try:
        reply = await store.acquire(
            ModelLeaseAcquireRequest(
                kind="k", resource="r", holder="a", request_id=uuid4()
            )
        )
    finally:
        await adapter.aclose()
    assert reply.outcome == "lease-refused"
    assert reply.reason == "all 2 slot(s) held"
    assert [path for path, _ in calls] == [
        "/v3/kv/range",
        "/v3/lease/grant",
        "/v3/kv/txn",
        "/v3/kv/txn",
        "/v3/lease/revoke",
    ]
    assert calls[-1][1] == {"ID": "7"}
