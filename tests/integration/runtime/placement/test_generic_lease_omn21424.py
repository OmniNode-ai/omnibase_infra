# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21424 AC1, AC2, AC3, AC5 and AC6 against a real etcd.

AC1-AC3 and AC6 run two lease stores, one per simulated runtime, on the
single-member test etcd. AC5 and the failover half of AC2 run two runtime
processes, each beside its own member of a three-member etcd, and lose the
whole host holding the lease primary: its runtime and its etcd member.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.handlers.lease import HandlerLeaseStore
from omnibase_infra.handlers.lease.handler_lease_fence import HandlerLeaseFence
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import (
    EnumLeaseFenceStatus,
    ModelLeaseAcquireRequest,
    ModelLeaseKindPolicy,
    ModelLeaseReply,
)

from .conftest import EtcdCluster, EtcdProcess

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

PROC = Path(__file__).with_name("lease_runtime_proc.py")
KINDS = [
    ModelLeaseKindPolicy(kind="exclusive", capacity=1, ttl_s=2),
    ModelLeaseKindPolicy(kind="pair", capacity=2, ttl_s=30),
]
H202_ANSWER_BOUND_S = 30.0


def _request(
    resource: str, holder: str, kind: str = "exclusive"
) -> ModelLeaseAcquireRequest:
    return ModelLeaseAcquireRequest(
        kind=kind, resource=resource, holder=holder, request_id=uuid4()
    )


@pytest.fixture
async def two_stores(
    etcd: EtcdProcess,
) -> AsyncIterator[tuple[HandlerLeaseStore, HandlerLeaseStore]]:
    adapters = [
        HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=10.0))
        for _ in range(2)
    ]
    try:
        yield (
            HandlerLeaseStore(adapters[0], KINDS),
            HandlerLeaseStore(adapters[1], KINDS),
        )
    finally:
        for adapter in adapters:
            await adapter.aclose()


async def test_ac1_hundred_concurrent_acquires_grant_exactly_one(
    two_stores: tuple[HandlerLeaseStore, HandlerLeaseStore],
) -> None:
    a, b = two_stores
    replies = await asyncio.gather(
        *(
            (a if i % 2 else b).acquire(_request("only-one", f"holder-{i}"))
            for i in range(100)
        )
    )
    outcomes = [r.outcome for r in replies]
    assert outcomes.count("lease-granted") == 1, outcomes
    assert outcomes.count("lease-refused") == 99
    assert len({r.lease_id for r in replies if r.outcome == "lease-granted"}) == 1


async def test_capacity_two_kind_grants_two_with_distinct_fences(
    two_stores: tuple[HandlerLeaseStore, HandlerLeaseStore],
) -> None:
    a, b = two_stores
    replies = await asyncio.gather(
        *(
            (a if i % 2 else b).acquire(_request("two", f"h{i}", "pair"))
            for i in range(20)
        )
    )
    granted = [r for r in replies if r.outcome == "lease-granted"]
    assert len(granted) == 2
    assert len({r.fence for r in granted}) == 2


async def test_ac2_regrant_after_expiry_has_a_strictly_greater_fence(
    two_stores: tuple[HandlerLeaseStore, HandlerLeaseStore],
) -> None:
    a, b = two_stores
    fences: list[int] = []
    for store in (a, b, a):
        reply = await _acquire_until_granted(store, "expiring", deadline_s=15)
        assert all(reply.fence > earlier for earlier in fences), (reply.fence, fences)
        fences.append(reply.fence)
        renewed = await store.renew("exclusive", "expiring", reply.lease_id)
        assert renewed.outcome == "lease-renewed" and renewed.fence == reply.fence
        refused = await (b if store is a else a).acquire(_request("expiring", "other"))
        assert refused.outcome == "lease-refused"


async def test_ac3_repeated_request_id_returns_the_same_lease_and_fence(
    two_stores: tuple[HandlerLeaseStore, HandlerLeaseStore],
) -> None:
    a, b = two_stores
    request = _request("idempotent", "holder")
    first = await a.acquire(request)
    second = await b.acquire(request)
    concurrent = await asyncio.gather(*(a.acquire(request) for _ in range(10)))
    assert first.outcome == "lease-granted"
    for reply in (second, *concurrent):
        assert reply.outcome == "lease-granted"
        assert (reply.lease_id, reply.fence) == (first.lease_id, first.fence)
    shown = await a.show("exclusive", "idempotent")
    assert [(r.lease_id, r.fence) for r in shown] == [(first.lease_id, first.fence)]
    fresh = _request("same-request-id-race", "holder")
    raced = await asyncio.gather(
        *((a if i % 2 else b).acquire(fresh) for i in range(10))
    )
    assert len({(r.outcome, r.lease_id, r.fence) for r in raced}) == 1
    assert raced[0].outcome == "lease-granted"


async def test_ac6_check_fence_stale_expired_and_ok(
    two_stores: tuple[HandlerLeaseStore, HandlerLeaseStore], etcd: EtcdProcess
) -> None:
    a, b = two_stores
    adapter = HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=5.0))
    fences = HandlerLeaseFence(adapter)
    try:
        old = await a.acquire(_request("fenced", "old", "pair"))
        released = await a.release("pair", "fenced", old.lease_id)
        assert released.outcome == "lease-released"
        assert (
            await fences.check_fence(kind="pair", resource="fenced", fence=old.fence)
            is EnumLeaseFenceStatus.EXPIRED
        )
        new = await b.acquire(_request("fenced", "new", "pair"))
        assert new.fence > old.fence
        lower = await fences.check_fence(
            kind="pair", resource="fenced", fence=old.fence
        )
        assert lower is EnumLeaseFenceStatus.STALE
        assert (
            await fences.check_fence(kind="pair", resource="fenced", fence=new.fence)
            is EnumLeaseFenceStatus.OK
        )
        assert (
            await fences.check_fence(kind="pair", resource="absent", fence=new.fence)
            is EnumLeaseFenceStatus.EXPIRED
        )
        assert (
            await fences.check_fence(
                kind="pair", resource="fenced", fence=new.fence + 1000
            )
            is EnumLeaseFenceStatus.UNKNOWN
        )
        etcd.stop()
        unreachable = await fences.check_fence(
            kind="pair", resource="fenced", fence=new.fence
        )
        assert unreachable is EnumLeaseFenceStatus.UNKNOWN
    finally:
        await adapter.aclose()


async def _acquire_until_granted(
    store: HandlerLeaseStore, resource: str, *, deadline_s: float
) -> ModelLeaseReply:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        reply = await store.acquire(_request(resource, "retrying"))
        if reply.outcome == "lease-granted":
            return reply
        await asyncio.sleep(0.2)
    raise AssertionError(f"no grant for {resource} within {deadline_s}s")


def _lines(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ask(requests: Path, request: ModelLeaseAcquireRequest) -> float:
    with requests.open("a") as handle:
        handle.write(request.model_dump_json() + "\n")
    return time.time()


async def _reply(
    replies: Path, request: ModelLeaseAcquireRequest, timeout_s: float
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for line in _lines(replies):
            if line["request_id"] == str(request.request_id):
                return line
        await asyncio.sleep(0.05)
    raise AssertionError(f"no reply to {request.request_id} within {timeout_s}s")


async def test_ac5_ac2_losing_the_lease_primary_host_answers_within_30s(
    etcd_cluster: EtcdCluster, tmp_path: Path
) -> None:
    """AC5: replay of the h202 outage; AC2 across the lease-primary failover."""
    requests, replies = tmp_path / "requests.jsonl", tmp_path / "replies.jsonl"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    hosts = {}
    for i, runtime_id in enumerate(("runtime-host0", "runtime-host1")):
        ready = tmp_path / f"{runtime_id}.ready"
        hosts[runtime_id] = (
            subprocess.Popen(
                [
                    sys.executable,
                    str(PROC),
                    f"--endpoint={etcd_cluster.members[i].endpoint}",
                    f"--runtime-id={runtime_id}",
                    f"--requests={requests}",
                    f"--replies={replies}",
                    f"--ready={ready}",
                ],
                env=env,
                start_new_session=True,
            ),
            etcd_cluster.members[i],
        )
    try:
        before = _request("survivor", "before-outage")
        _ask(requests, before)
        first = await _reply(replies, before, 30)
        assert first["outcome"] == "lease-granted"
        primary = str(first["runtime"])
        proc, member = hosts[primary]
        # The whole host goes: its runtime and its etcd member.
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        member.stop()
        lost_at = time.time()
        after = _request("next", "after-outage")
        asked_at = _ask(requests, after)
        answer = await _reply(replies, after, H202_ANSWER_BOUND_S + 5)
        gap = float(answer["t"]) - asked_at
        receipt = {
            "primary_lost": primary,
            "answered_by": answer["runtime"],
            "gap_s": round(gap, 3),
            "outcome": answer["outcome"],
            "lost_at": lost_at,
        }
        (tmp_path / "ac5_receipt.json").write_text(json.dumps(receipt))
        print(f"AC5 receipt: {json.dumps(receipt)}")
        assert answer["runtime"] != primary
        assert answer["outcome"] == "lease-granted"
        assert gap <= H202_ANSWER_BOUND_S, receipt
        # AC2 across the failover: "survivor" expires (TTL 3 s, no renewals) and
        # the new lease primary's re-grant carries a strictly greater fence.
        deadline = time.monotonic() + 30
        regrant: dict[str, object] | None = None
        while regrant is None and time.monotonic() < deadline:
            retry = _request("survivor", "after-failover")
            _ask(requests, retry)
            line = await _reply(replies, retry, 15)
            if line["outcome"] == "lease-granted":
                regrant = line
            else:
                await asyncio.sleep(0.5)
        assert regrant is not None, "survivor was never re-granted after expiry"
        assert int(str(regrant["fence"])) > int(str(first["fence"])), (regrant, first)
    finally:
        for proc, _ in hosts.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
