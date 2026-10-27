# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21429 AC2: each slice-6 lease kind grants one of two concurrent holders.

The kinds and resource shapes are the slice-6 rows of section 7 of the
OMN-21279 runtime cluster and generic lease plan (inventory rows E1, E4, E6,
E7, E10, E11, E12, E13). Two lease stores, one per simulated runtime, share
the single-member test etcd, as in the OMN-21424 tests. ``once`` is
insert-only: its grant has no etcd lease, so it never expires and release
refuses it (``ExpiredLeaseFreesResource`` of ``GenericLease.tla`` holds
vacuously). The expiring kinds also keep ``FenceMonotonicPerResource``: the
grant after a release carries a strictly higher fence.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.handlers.lease import HandlerLeaseStore, resource_prefix
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import (
    ModelLeaseAcquireRequest,
    ModelLeaseKindPolicy,
    ModelLeaseReply,
)

from .conftest import EtcdProcess

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

# kind -> (policy, resource of the shape section 7 names, inventory rows)
SLICE6_KINDS: dict[str, tuple[ModelLeaseKindPolicy, str, str]] = {
    "dispatch": (
        ModelLeaseKindPolicy(kind="dispatch", ttl_s=300),
        "dispatch:3f786850e387550fdab836ed7e6dc881de23001b",
        "E1",
    ),
    "merge-sweep": (
        ModelLeaseKindPolicy(kind="merge-sweep", ttl_s=900),
        "global",
        "E4",
    ),
    "once": (
        ModelLeaseKindPolicy(kind="once", expires=False),
        "lane-key:fix-omn-17427-h101",
        "E6, E10",
    ),
    "queue-item": (
        ModelLeaseKindPolicy(kind="queue-item", ttl_s=900),
        "queue-item:0f8fad5b-d9cb-469f-a165-70867728950e",
        "E7, E12",
    ),
    "close-out-run": (
        ModelLeaseKindPolicy(kind="close-out-run", ttl_s=600),
        "close-out:dev",
        "E11",
    ),
    "token": (
        ModelLeaseKindPolicy(kind="token", ttl_s=3600),
        "account:claude-pool-1",
        "E13",
    ),
}


@pytest.fixture
async def two_stores(
    etcd: EtcdProcess,
) -> AsyncIterator[tuple[HandlerLeaseStore, HandlerLeaseStore]]:
    kinds = [policy for policy, _, _ in SLICE6_KINDS.values()]
    adapters = [
        HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=10.0))
        for _ in range(2)
    ]
    try:
        yield (
            HandlerLeaseStore(adapters[0], kinds),
            HandlerLeaseStore(adapters[1], kinds),
        )
    finally:
        for adapter in adapters:
            await adapter.aclose()


def _request(kind: str, resource: str, holder: str) -> ModelLeaseAcquireRequest:
    return ModelLeaseAcquireRequest(
        kind=kind, resource=resource, holder=holder, request_id=uuid4()
    )


def _granted(replies: list[ModelLeaseReply]) -> list[ModelLeaseReply]:
    return [reply for reply in replies if reply.outcome == "lease-granted"]


@pytest.mark.parametrize("kind", sorted(SLICE6_KINDS))
async def test_ac2_two_concurrent_holders_get_exactly_one_grant(
    two_stores: tuple[HandlerLeaseStore, HandlerLeaseStore], kind: str
) -> None:
    a, b = two_stores
    _, resource, rows = SLICE6_KINDS[kind]
    replies = list(
        await asyncio.gather(
            a.acquire(_request(kind, resource, "h201/lane-a")),
            b.acquire(_request(kind, resource, "h105/lane-b")),
        )
    )
    granted = _granted(replies)
    assert len(granted) == 1, (kind, rows, replies)
    assert [r.outcome for r in replies].count("lease-refused") == 1
    shown = await a.show(kind, resource)
    assert [(r.lease_id, r.fence) for r in shown] == [
        (granted[0].lease_id, granted[0].fence)
    ]


@pytest.mark.parametrize("kind", sorted(SLICE6_KINDS))
async def test_ac2_many_racing_holders_across_two_runtimes_get_one_grant(
    two_stores: tuple[HandlerLeaseStore, HandlerLeaseStore], kind: str
) -> None:
    a, b = two_stores
    _, resource, _ = SLICE6_KINDS[kind]
    resource = f"{resource}-race"
    replies = list(
        await asyncio.gather(
            *(
                (a if i % 2 else b).acquire(_request(kind, resource, f"holder-{i}"))
                for i in range(20)
            )
        )
    )
    granted = _granted(replies)
    assert len(granted) == 1, (kind, replies)
    assert [r.outcome for r in replies].count("lease-refused") == 19, (kind, replies)
    shown = await a.show(kind, resource)
    assert [(r.lease_id, r.fence) for r in shown] == [
        (granted[0].lease_id, granted[0].fence)
    ]


@pytest.mark.parametrize(
    "kind", sorted(k for k, (p, _, _) in SLICE6_KINDS.items() if p.expires)
)
async def test_expiring_kind_regrant_after_release_has_a_higher_fence(
    two_stores: tuple[HandlerLeaseStore, HandlerLeaseStore], kind: str
) -> None:
    a, b = two_stores
    _, resource, _ = SLICE6_KINDS[kind]
    resource = f"{resource}-regrant"
    first = await a.acquire(_request(kind, resource, "first"))
    assert first.outcome == "lease-granted" and first.lease_id is not None
    released = await a.release(kind, resource, first.lease_id)
    assert released.outcome == "lease-released"
    second = await b.acquire(_request(kind, resource, "second"))
    assert second.outcome == "lease-granted"
    assert second.fence > first.fence


async def test_once_kind_is_insert_only_and_never_expires(
    two_stores: tuple[HandlerLeaseStore, HandlerLeaseStore], etcd: EtcdProcess
) -> None:
    a, b = two_stores
    resource = "lane-key:sent-once"
    first = await a.acquire(_request("once", resource, "launchd-tick"))
    assert first.outcome == "lease-granted" and first.lease_id is not None
    assert first.ttl_s == 0
    reader = HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=10.0))
    try:
        entries = await reader.get_prefix_entries(resource_prefix("once", resource))
    finally:
        await reader.aclose()
    assert [lease for _, _, _, lease in entries] == [0]
    for refused in (
        await a.release("once", resource, first.lease_id),
        await a.renew("once", resource, first.lease_id),
    ):
        assert refused.outcome == "lease-refused", refused
    again = await b.acquire(_request("once", resource, "session-tick"))
    assert again.outcome == "lease-refused"
