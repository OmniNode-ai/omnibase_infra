# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One runtime process for the OMN-21421 failover tests.

Loads one ``primary`` node whose consumer folds the events of a topic file (one
JSON event per line, appended by the test as it "produces") into a projection, checkpointing rows and next offsets to a shared store file every
``CHECKPOINT_EVERY`` events. Every step is written to ``--out`` as JSON lines.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
from pathlib import Path

from omnibase_infra.handlers.placement import (
    HandlerPlacementEtcd,
    HandlerPlacementRuntime,
    ModelPlacementPolicy,
    ModelPlacementReplayEvent,
    fold_projection_from_offsets,
)

NODE = "node_placement_probe_reducer"
CHECKPOINT_EVERY = 7


def read_topic(path: Path) -> list[ModelPlacementReplayEvent]:
    lines = path.read_text().splitlines() if path.exists() else []
    return [
        ModelPlacementReplayEvent.model_validate_json(line) for line in lines if line
    ]


def fold(rows: dict[str, str], event: ModelPlacementReplayEvent) -> dict[str, str]:
    current = json.loads(rows.get(event.key, '{"count": 0, "values": []}'))
    current["count"] += 1
    current["values"].append(event.value)
    return {**rows, event.key: json.dumps(current, sort_keys=True)}


class ProbeNode:
    def __init__(self, runtime_id: str, out: Path, store: Path, topic: Path) -> None:
        self.runtime_id = runtime_id
        self.topic = topic
        self.out = out
        self.store = store
        self.runtime: HandlerPlacementRuntime | None = None
        self.rows: dict[str, str] = {}
        self.offsets: dict[int, int] = {}
        self._task: asyncio.Task[None] | None = None

    def record(self, event: str, **fields: object) -> None:
        line = json.dumps(
            {"t": time.time(), "runtime": self.runtime_id, "event": event, **fields}
        )
        with self.out.open("a") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    async def rebuild(self, placement_epoch: int) -> None:
        stored = json.loads(self.store.read_text()) if self.store.exists() else {}
        stored_rows = dict(stored.get("rows", {}))
        stored_offsets = {int(k): int(v) for k, v in stored.get("offsets", {}).items()}
        self.rows, self.offsets = fold_projection_from_offsets(
            rows=stored_rows,
            offsets=stored_offsets,
            events=read_topic(self.topic),
            fold=fold,
        )
        self.record(
            "rebuild",
            epoch=placement_epoch,
            stored_offsets=stored_offsets,
            offsets=self.offsets,
            rows=self.rows,
        )

    async def start_consuming(self, placement_epoch: int) -> None:
        self.record("start", epoch=placement_epoch)
        self._task = asyncio.create_task(self._consume(placement_epoch))

    async def stop_consuming(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self.record("stop")

    async def _consume(self, placement_epoch: int) -> None:
        while True:
            assert self.runtime is not None
            if self.runtime.may_publish(NODE):
                events = read_topic(self.topic)
                nxt = self.offsets.get(0, 0)
                if nxt < len(events):
                    self.rows, self.offsets = fold_projection_from_offsets(
                        rows=self.rows,
                        offsets=self.offsets,
                        events=[events[nxt]],
                        fold=fold,
                    )
                    if self.offsets[0] % CHECKPOINT_EVERY == 0:
                        tmp = self.store.with_suffix(".tmp")
                        tmp.write_text(
                            json.dumps({"rows": self.rows, "offsets": self.offsets})
                        )
                        tmp.replace(self.store)
                self.record(
                    "consume",
                    epoch=placement_epoch,
                    offsets=self.offsets,
                    rows=self.rows,
                )
            await asyncio.sleep(0.05)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--topic", type=Path, required=True)
    parser.add_argument("--failover-target-s", type=float, default=10.0)
    args = parser.parse_args()
    node = ProbeNode(args.runtime_id, args.out, args.store, args.topic)
    adapter = HandlerPlacementEtcd(args.endpoint)
    runtime = HandlerPlacementRuntime(
        runtime_id=args.runtime_id,
        adapter=adapter,
        nodes={
            NODE: (
                ModelPlacementPolicy(
                    mode="primary", failover_target_s=args.failover_target_s
                ),
                node,
            )
        },
    )
    node.runtime = runtime
    await runtime.start()
    node.record("ready")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
