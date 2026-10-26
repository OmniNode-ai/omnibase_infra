# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lane runner process for the OMN-21428 tests.

Acquires a ``lane-work`` lease on the test etcd, renews it every ``ttl_s/3``,
and, given ``--release-after``, releases it with ``--outcome``. It writes one
JSON lease reply per line to stdout, standing in for the lease orchestrator's
published events; it has no ledger path and writes no ledger row.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from omnibase_infra.handlers.lease import HandlerLeaseStore
from omnibase_infra.handlers.lease.handler_lane_work_lease import (
    HandlerLaneWorkLease,
    StoreLeaseCommands,
)
from omnibase_infra.handlers.lease.handler_lease_ledger import LANE_WORK_KIND
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import (
    ModelLaneWorkHolder,
    ModelLeaseKindPolicy,
    ModelLeaseReply,
)


def _write(reply: ModelLeaseReply) -> None:
    sys.stdout.write(reply.model_dump_json() + "\n")
    sys.stdout.flush()


async def _main(args: argparse.Namespace) -> None:
    adapter = HandlerPlacementEtcd(args.endpoint, timeout_s=5.0)
    store = HandlerLeaseStore(
        adapter, [ModelLeaseKindPolicy(kind=LANE_WORK_KIND, ttl_s=args.ttl_s)]
    )
    holder = ModelLaneWorkHolder(
        host=args.host,
        lane=args.lane,
        run=args.run,
        principal="dev-cli-test",
        ticket="OMN-21428",
    )
    runner = HandlerLaneWorkLease(StoreLeaseCommands(store), holder, ttl_s=args.ttl_s)
    _write(await runner.acquire())
    started = time.monotonic()
    while args.release_after is None or time.monotonic() - started < args.release_after:
        await asyncio.sleep(runner.renew_interval_s)
        reply = await runner.renew()
        _write(reply)
        if reply.outcome != "lease-renewed":
            return
    _write(await runner.release(args.outcome))
    await adapter.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--host", default="h-test")
    parser.add_argument("--run", default="run-test")
    parser.add_argument("--ttl-s", type=int, default=3)
    parser.add_argument("--release-after", type=float, default=None)
    parser.add_argument("--outcome", default="done")
    asyncio.run(_main(parser.parse_args()))
