# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One runtime process hosting the lease primary for the OMN-21424 failover tests.

Loads ``node_lease_orchestrator`` as a ``primary`` node. While it is the acting
primary it answers every acquire in the requests file (one JSON
``ModelLeaseAcquireRequest`` per line) that has no reply yet, appending one
JSON reply per line to the replies file with this runtime's id and the time.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
from pathlib import Path

import httpx

from omnibase_infra.handlers.lease import LeaseStore
from omnibase_infra.handlers.lease.lease_etcd_gateway import LeaseEtcdGateway
from omnibase_infra.handlers.placement import (
    HandlerPlacementEtcd,
    HandlerPlacementRuntime,
    ModelPlacementPolicy,
)
from omnibase_infra.models.lease import ModelLeaseAcquireRequest, ModelLeaseKindPolicy

NODE = "node_lease_orchestrator"


def _lines(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class LeasePrimaryNode:
    def __init__(
        self, runtime_id: str, store: LeaseStore, requests: Path, replies: Path
    ) -> None:
        self.runtime_id = runtime_id
        self.store = store
        self.requests = requests
        self.replies = replies
        self.runtime: HandlerPlacementRuntime | None = None
        self._task: asyncio.Task[None] | None = None

    async def rebuild(self, placement_epoch: int) -> None:
        """The lease state lives in etcd; there is no local projection to fold."""

    async def start_consuming(self, placement_epoch: int) -> None:
        self._task = asyncio.create_task(self._serve(placement_epoch))

    async def stop_consuming(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _serve(self, placement_epoch: int) -> None:
        while True:
            assert self.runtime is not None
            if self.runtime.may_publish(NODE):
                answered = {str(r["request_id"]) for r in _lines(self.replies)}
                for raw in _lines(self.requests):
                    request = ModelLeaseAcquireRequest.model_validate(raw)
                    if str(request.request_id) in answered:
                        continue
                    reply = await self.store.acquire(request)
                    if not self.runtime.may_publish(NODE):
                        break
                    line = json.dumps(
                        {
                            "t": time.time(),
                            "runtime": self.runtime_id,
                            "epoch": placement_epoch,
                            **reply.model_dump(mode="json"),
                        }
                    )
                    with self.replies.open("a") as handle:
                        handle.write(line + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
            await asyncio.sleep(0.05)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--replies", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    gateway = LeaseEtcdGateway(args.endpoint, httpx.AsyncClient(timeout=2.0))
    store = LeaseStore(
        gateway, [ModelLeaseKindPolicy(kind="exclusive", capacity=1, ttl_s=3)]
    )
    node = LeasePrimaryNode(args.runtime_id, store, args.requests, args.replies)
    runtime = HandlerPlacementRuntime(
        runtime_id=args.runtime_id,
        adapter=HandlerPlacementEtcd(args.endpoint, timeout_s=2.0),
        nodes={
            NODE: (ModelPlacementPolicy(mode="primary", failover_target_s=10.0), node)
        },
    )
    node.runtime = runtime
    await runtime.start()
    args.ready.write_text("ready")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
