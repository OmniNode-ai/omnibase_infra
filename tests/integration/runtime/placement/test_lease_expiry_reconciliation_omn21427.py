# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reconcile TTL expiry into projection events without renewing the grant."""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.lease import (
    HandlerLeaseReconcile,
    HandlerLeaseStore,
    reduce_lease_event,
)
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import ModelLeaseAcquireRequest, ModelLeaseKindPolicy

from .conftest import EtcdProcess

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


async def test_ttl_expiry_reconciles_original_grant_without_renewal(
    etcd: EtcdProcess,
) -> None:
    adapter = HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=1.0))
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="slot", ttl_s=2)])
    reconcile = HandlerLeaseReconcile(store)
    try:
        grant = await store.acquire(
            ModelLeaseAcquireRequest(
                kind="slot", resource="host", holder="old", request_id=uuid4()
            )
        )
        state, _ = reduce_lease_event({}, grant)
        known = state[("slot", "host")]
        assert (await reconcile.handle(known)).outcome == "lease-granted"
        deadline = time.monotonic() + 15
        while True:
            expired = await reconcile.handle(known)
            if expired.outcome == "lease-expired":
                break
            assert expired.outcome == "lease-granted"
            assert time.monotonic() < deadline, "TTL did not free the slot"
            await asyncio.sleep(0.1)
        assert await store.show("slot", "host") == []
        assert expired.outcome == "lease-expired"
        assert (expired.lease_id, expired.fence, expired.holder) == (
            grant.lease_id,
            grant.fence,
            grant.holder,
        )
        state, intents = reduce_lease_event(state, expired)
        assert state[("slot", "host")].state == "expired"
        assert len(intents) == 1
        assert reduce_lease_event(state, expired) == (state, ())

        newer = await store.acquire(
            ModelLeaseAcquireRequest(
                kind="slot", resource="host", holder="new", request_id=uuid4()
            )
        )
        assert newer.fence > expired.fence
        late_expiry = await reconcile.handle(known)
        assert late_expiry == expired
        state, _ = reduce_lease_event(state, newer)
        assert reduce_lease_event(state, late_expiry) == (state, ())
        assert state[("slot", "host")].holder == "new"
        assert state[("slot", "host")].state == "held"
    finally:
        await adapter.aclose()


async def test_unreachable_etcd_does_not_emit_expiry(etcd: EtcdProcess) -> None:
    adapter = HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=0.2))
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="slot", ttl_s=2)])
    try:
        grant = await store.acquire(
            ModelLeaseAcquireRequest(
                kind="slot", resource="host", holder="h", request_id=uuid4()
            )
        )
        state, _ = reduce_lease_event({}, grant)
        etcd.stop()
        with pytest.raises(InfraConnectionError):
            await HandlerLeaseReconcile(store).handle(state[("slot", "host")])
    finally:
        await adapter.aclose()
