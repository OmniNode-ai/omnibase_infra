# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One host's placer for the OMN-21427 host-loss test.

Places ``--count`` units onto ``--host`` through ``slot`` leases against the
etcd member beside it, writes the grants to ``--ready`` as JSON, then renews
them every second until it is killed. It keeps no lock file and no marker:
the only record of the slots it holds is the etcd lease.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx

from omnibase_infra.handlers.lease import HandlerLeaseSlotPlacement, HandlerLeaseStore
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import ModelLeaseKindPolicy


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--ttl-s", type=int, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    adapter = HandlerPlacementEtcd(args.endpoint, httpx.AsyncClient(timeout=2.0))
    store = HandlerLeaseStore(
        adapter,
        [ModelLeaseKindPolicy(kind="slot", capacity=args.count, ttl_s=args.ttl_s)],
    )
    placement = HandlerLeaseSlotPlacement(store)
    grants = []
    for i in range(args.count):
        reply = await placement.place([args.host], f"unit-{i}", uuid4())
        assert reply.outcome == "lease-granted", reply
        grants.append(reply)
    args.ready.write_text(json.dumps([g.model_dump(mode="json") for g in grants]))
    while True:
        await asyncio.sleep(1.0)
        for grant in grants:
            assert grant.lease_id is not None
            await placement.renew(args.host, grant.lease_id)


if __name__ == "__main__":
    asyncio.run(main())
