# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One host's placer for the OMN-21427 host-loss test.

Places ``--count`` units onto ``--host`` through ``slot`` leases against the
etcd member beside it, writes the grants to ``--ready`` as JSON, then renews
them every second until it is killed. It keeps no lock file and no marker:
the only record of the slots it holds is the etcd lease. A renew that raises
or is not renewed is written to stderr and retried, so the process outlives a
transient etcd error and only the test's kill ends it; the test checks it is
still alive before the kill.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
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
        if reply.outcome != "lease-granted" or reply.lease_id is None:
            raise SystemExit(f"slot holder could not place unit-{i}: {reply!r}")
        grants.append(reply)
    staged = args.ready.with_name(args.ready.name + ".tmp")
    staged.write_text(json.dumps([g.model_dump(mode="json") for g in grants]))
    staged.replace(args.ready)
    while True:
        await asyncio.sleep(1.0)
        for grant in grants:
            assert grant.lease_id is not None  # checked at placement
            try:
                renewed = await placement.renew(args.host, grant.lease_id)
            except (httpx.HTTPError, OSError) as exc:
                print(f"renew {grant.lease_id} raised: {exc!r}", file=sys.stderr)
                continue
            if renewed.outcome != "lease-renewed":
                print(f"renew {grant.lease_id} refused: {renewed!r}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
