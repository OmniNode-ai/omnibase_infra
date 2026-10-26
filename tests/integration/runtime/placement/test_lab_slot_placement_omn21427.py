# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21427 AC1 and AC3 against a real etcd.

AC1: placement takes ``slot`` leases. Two placers, one per simulated runtime,
race for the slots of two hosts on the single-member test etcd; the grants
never exceed a host's capacity, a retry replays its grant, and a release frees
the slot. AC3: a placer beside one member of a three-member etcd holds a
host's slots and renews them; the whole placing host goes (its process and its
etcd member), and another host's placement onto those slots proceeds once the
leases expire, never past expiry plus the etcd failover bound.
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

from omnibase_infra.handlers.lease import HandlerLeaseSlotPlacement, HandlerLeaseStore
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import ModelLeaseKindPolicy

from .conftest import EtcdCluster, EtcdProcess

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

HOLDER = Path(__file__).with_name("slot_holder_proc.py")
SLOT = ModelLeaseKindPolicy(
    kind="slot", capacity=2, ttl_s=30, resource_capacity={"slot:h-big": 3}
)
HOLD_TTL_S = 3
# etcd restarts every lease's TTL when a new leader takes over, so a lease
# held across a leader loss may live one election plus one TTL past the kill.
EXPIRY_BOUND_S = 2 * HOLD_TTL_S + 5


@pytest.fixture
async def two_placers(
    etcd: EtcdProcess,
) -> AsyncIterator[tuple[HandlerLeaseSlotPlacement, HandlerLeaseSlotPlacement]]:
    adapters = [
        HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=10.0))
        for _ in range(2)
    ]
    try:
        yield (
            HandlerLeaseSlotPlacement(HandlerLeaseStore(adapters[0], [SLOT])),
            HandlerLeaseSlotPlacement(HandlerLeaseStore(adapters[1], [SLOT])),
        )
    finally:
        for adapter in adapters:
            await adapter.aclose()


async def test_ac1_placement_never_exceeds_a_hosts_slot_capacity(
    two_placers: tuple[HandlerLeaseSlotPlacement, HandlerLeaseSlotPlacement],
) -> None:
    a, b = two_placers
    hosts = ["h-small", "h-big"]
    replies = await asyncio.gather(
        *((a if i % 2 else b).place(hosts, f"unit-{i}", uuid4()) for i in range(30))
    )
    granted = [r for r in replies if r.outcome == "lease-granted"]
    by_host = {h: [r for r in granted if r.resource == f"slot:{h}"] for h in hosts}
    assert len(by_host["h-small"]) == 2
    assert len(by_host["h-big"]) == 3
    assert len(granted) == 5
    assert [r.outcome for r in replies].count("lease-refused") == 25
    assert len({r.fence for r in granted}) == 5
    assert await a.free_slots("h-small") == 0
    assert await b.free_slots("h-big") == 0


async def test_ac1_a_retried_placement_replays_its_grant(
    two_placers: tuple[HandlerLeaseSlotPlacement, HandlerLeaseSlotPlacement],
) -> None:
    a, b = two_placers
    request_id = uuid4()
    first = await a.place(["h-small", "h-big"], "unit", request_id)
    retried = await b.place(["h-big", "h-small"], "unit", request_id)
    assert first.outcome == retried.outcome == "lease-granted"
    assert (retried.resource, retried.lease_id, retried.fence) == (
        first.resource,
        first.lease_id,
        first.fence,
    )
    assert await a.free_slots("h-small") == 1
    assert await a.free_slots("h-big") == 3


async def test_ac1_release_frees_the_slot_for_another_placer(
    two_placers: tuple[HandlerLeaseSlotPlacement, HandlerLeaseSlotPlacement],
) -> None:
    a, b = two_placers
    held = [await a.place(["h-small"], f"u{i}", uuid4()) for i in range(2)]
    assert all(r.outcome == "lease-granted" for r in held)
    refused = await b.place(["h-small"], "late", uuid4())
    assert refused.outcome == "lease-refused"
    assert held[0].lease_id is not None
    released = await a.release("h-small", held[0].lease_id)
    assert released.outcome == "lease-released"
    late = await b.place(["h-small"], "late", uuid4())
    assert late.outcome == "lease-granted"
    assert late.fence > max(r.fence for r in held)


async def test_ac3_losing_the_slot_holding_host_frees_its_slots_at_expiry(
    etcd_cluster: EtcdCluster, tmp_path: Path
) -> None:
    ready = tmp_path / "holder.ready"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    holder = subprocess.Popen(
        [
            sys.executable,
            str(HOLDER),
            f"--endpoint={etcd_cluster.members[0].endpoint}",
            "--host=h-lost",
            f"--count={SLOT.capacity}",
            f"--ttl-s={HOLD_TTL_S}",
            f"--ready={ready}",
        ],
        env=env,
    )
    adapter = HandlerPlacementEtcd(
        etcd_cluster.members[1].endpoint, httpx.AsyncClient(timeout=2.0)
    )
    other = HandlerLeaseSlotPlacement(HandlerLeaseStore(adapter, [SLOT]))
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        assert ready.exists(), "the slot holder never placed its units"
        held = json.loads(ready.read_text())
        # Held and renewed: the other host's placement is refused past one TTL.
        await asyncio.sleep(HOLD_TTL_S + 1)
        assert (await other.place(["h-lost"], "early", uuid4())).outcome == (
            "lease-refused"
        )
        # The whole host goes: its placer and its etcd member.
        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=10)
        etcd_cluster.members[0].stop()
        lost_at = time.monotonic()
        placed = []
        while len(placed) < SLOT.capacity and time.monotonic() - lost_at < 60:
            try:
                reply = await other.place(["h-lost"], "after-loss", uuid4())
            except httpx.HTTPError:
                reply = None
            if reply is not None and reply.outcome == "lease-granted":
                placed.append(reply)
            else:
                await asyncio.sleep(0.2)
        freed_after_s = time.monotonic() - lost_at
        receipt = {
            "held_fences": [g["fence"] for g in held],
            "placed_fences": [r.fence for r in placed],
            "placed": len(placed),
            "freed_after_s": round(freed_after_s, 3),
            "ttl_s": HOLD_TTL_S,
            "bound_s": EXPIRY_BOUND_S,
        }
        (tmp_path / "ac3_receipt.json").write_text(json.dumps(receipt))
        print(f"AC3 receipt: {json.dumps(receipt)}")
        assert len(placed) == SLOT.capacity, receipt
        assert freed_after_s <= EXPIRY_BOUND_S, receipt
        assert min(r.fence for r in placed) > max(int(g["fence"]) for g in held)
    finally:
        if holder.poll() is None:
            holder.send_signal(signal.SIGKILL)
            holder.wait(timeout=10)
        await adapter.aclose()
