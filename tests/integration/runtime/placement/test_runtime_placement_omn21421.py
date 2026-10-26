# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21421 AC1-AC5 against a real etcd: two runtimes, election, self-fence, rebuild."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import httpx
import pytest

from omnibase_infra.handlers.placement import (
    EnumPlacementHealthVerdict,
    HandlerPlacementEpochGuard,
    HandlerPlacementEtcd,
    HandlerPlacementRuntime,
    ModelPlacementPolicy,
    ModelPlacementReplayEvent,
)

from .conftest import BlackholeProxy, EtcdProcess

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

PROC = Path(__file__).with_name("placement_runtime_proc.py")
FAILOVER_TARGET_S = 10.0
FENCE_BOUND_S = 5.0


class RecordingNode:
    """In-process node: publishes every 50 ms while the runtime lets it."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.runtime: HandlerPlacementRuntime | None = None
        self.log: list[tuple[str, float, int]] = []
        self._task: asyncio.Task[None] | None = None

    async def rebuild(self, placement_epoch: int) -> None:
        self.log.append(("rebuild", time.monotonic(), placement_epoch))

    async def start_consuming(self, placement_epoch: int) -> None:
        self.log.append(("start", time.monotonic(), placement_epoch))
        self._task = asyncio.create_task(self._publish(placement_epoch))

    async def stop_consuming(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self.log.append(("stop", time.monotonic(), 0))

    async def _publish(self, placement_epoch: int) -> None:
        while True:
            assert self.runtime is not None
            if self.runtime.may_publish(self.name):
                self.log.append(("publish", time.monotonic(), placement_epoch))
            await asyncio.sleep(0.05)

    def times(self, event: str) -> list[float]:
        return [t for e, t, _ in self.log if e == event]


_Nodes = dict[str, tuple[Literal["any", "primary"], RecordingNode]]


def _runtime(runtime_id: str, endpoint: str, nodes: _Nodes) -> HandlerPlacementRuntime:
    runtime = HandlerPlacementRuntime(
        runtime_id=runtime_id,
        adapter=HandlerPlacementEtcd(endpoint, httpx.AsyncClient(timeout=1.0)),
        nodes={
            name: (
                ModelPlacementPolicy(mode=mode, failover_target_s=FAILOVER_TARGET_S),
                node,
            )
            for name, (mode, node) in nodes.items()
        },
    )
    for _, node in nodes.values():
        node.runtime = runtime
    return runtime


async def _until(predicate: Callable[[], bool], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"condition not met within {timeout_s}s")


def _records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _produce(tmp: Path, offsets: range) -> None:
    with (tmp / "topic.jsonl").open("a") as handle:
        for i in offsets:
            event = ModelPlacementReplayEvent(
                partition=0, offset=i, key=f"k{(i * 5) % 4}", value=f"v{i}"
            )
            handle.write(event.model_dump_json() + "\n")


def _spawn(etcd: EtcdProcess, runtime_id: str, tmp: Path) -> subprocess.Popen[bytes]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    return subprocess.Popen(
        [
            sys.executable,
            str(PROC),
            f"--endpoint={etcd.endpoint}",
            f"--runtime-id={runtime_id}",
            f"--out={tmp / f'{runtime_id}.jsonl'}",
            f"--store={tmp / 'projection_store.json'}",
            f"--topic={tmp / 'topic.jsonl'}",
            f"--failover-target-s={FAILOVER_TARGET_S}",
        ],
        env=env,
        start_new_session=True,
    )


async def test_ac1_ac3_killing_the_primary_fails_over_and_rebuilds(
    etcd: EtcdProcess, tmp_path: Path
) -> None:
    """AC1: the standby consumes within failover_target_s of the primary's SIGKILL.

    AC3: the new primary's fold from the stored table and offsets has the
    same rows, row by row, as the old primary had at the same offset.
    """
    _produce(tmp_path, range(12))
    a = _spawn(etcd, "runtime-a", tmp_path)
    b: subprocess.Popen[bytes] | None = None
    try:
        a_out, b_out = tmp_path / "runtime-a.jsonl", tmp_path / "runtime-b.jsonl"
        # The old primary rebuilds the first 12 events, then consumes 18 more as
        # they are produced, checkpointing at 14, 21 and 28.
        await _until(lambda: any(r["event"] == "start" for r in _records(a_out)), 30)
        for i in range(12, 30):
            _produce(tmp_path, range(i, i + 1))
            await asyncio.sleep(0.1)
        await _until(
            lambda: any(
                r["event"] == "consume" and r["offsets"] == {"0": 30}
                for r in _records(a_out)
            ),
            30,
        )
        b = _spawn(etcd, "runtime-b", tmp_path)
        await _until(lambda: any(r["event"] == "ready" for r in _records(b_out)), 30)
        await asyncio.sleep(1.0)
        assert not any(r["event"] == "start" for r in _records(b_out)), (
            "standby started early"
        )

        os.killpg(a.pid, signal.SIGKILL)
        killed_at = time.time()
        a.wait(timeout=10)
        await _until(
            lambda: any(r["event"] == "consume" for r in _records(b_out)),
            FAILOVER_TARGET_S + 5,
        )
        b_records = _records(b_out)
        first_consume = min(
            float(str(r["t"])) for r in b_records if r["event"] == "consume"
        )
        gap = first_consume - killed_at
        print(f"AC1 failover gap {gap:.3f}s (failover_target_s={FAILOVER_TARGET_S})")
        assert gap <= FAILOVER_TARGET_S

        # Rebuild precedes consume, and starts from the stored offsets, not zero.
        events = [r["event"] for r in b_records]
        assert events.index("rebuild") < events.index("start") < events.index("consume")
        rebuild = next(r for r in b_records if r["event"] == "rebuild")
        assert rebuild["stored_offsets"] == {"0": 28}
        assert rebuild["offsets"] == {"0": 30}

        old_rows = next(
            r["rows"]
            for r in reversed(_records(a_out))
            if r["event"] == "consume" and r["offsets"] == {"0": 30}
        )
        new_rows = rebuild["rows"]
        assert isinstance(old_rows, dict) and isinstance(new_rows, dict)
        diff = sorted(
            k
            for k in old_rows.keys() | new_rows.keys()
            if old_rows.get(k) != new_rows.get(k)
        )
        print(f"AC3 row diff {diff} over {len(old_rows)} rows")
        assert diff == []
        assert len(old_rows) == 4
    finally:
        for proc in (a, b):
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


async def test_ac2_primary_cut_off_from_etcd_stops_within_5s(
    etcd: EtcdProcess, blackhole_proxy: BlackholeProxy
) -> None:
    """AC2: no publish from the cut-off primary later than 5 s after the cut."""
    node_a, node_b = RecordingNode("node_x_reducer"), RecordingNode("node_x_reducer")
    a = _runtime(
        "runtime-a", blackhole_proxy.endpoint, {"node_x_reducer": ("primary", node_a)}
    )
    b = _runtime("runtime-b", etcd.endpoint, {"node_x_reducer": ("primary", node_b)})
    await a.start()
    try:
        await _until(lambda: bool(node_a.times("publish")), 10)
        await b.start()
        await asyncio.sleep(1.5)
        assert not node_b.times("start"), "standby started while the primary is live"

        blackhole_proxy.cut()
        cut_at = time.monotonic()
        await _until(lambda: bool(node_b.times("publish")), 20)
        await asyncio.sleep(1.0)

        last_a_publish = max(node_a.times("publish"))
        a_stop = node_a.times("stop")
        stop_after = f"{a_stop[0] - cut_at:.3f}s" if a_stop else "never"
        print(
            f"AC2 last publish {last_a_publish - cut_at:.3f}s after cut, "
            f"stop {stop_after} after cut, "
            f"standby first publish {min(node_b.times('publish')) - cut_at:.3f}s after cut"
        )
        assert last_a_publish - cut_at <= FENCE_BOUND_S
        assert a_stop and a_stop[0] - cut_at <= FENCE_BOUND_S
        # AtMostOneActingPrimary: the new primary publishes only after the old stopped.
        assert min(node_b.times("publish")) > last_a_publish
        assert "node_x_reducer" not in a.started_nodes()
        with pytest.raises(PermissionError):
            a.stamp("node_x_reducer", "late")
    finally:
        await b.stop()
        await a.stop()


async def test_epoch_guard_refuses_the_old_primary_after_failover(
    etcd: EtcdProcess,
) -> None:
    """Every command carries placement_epoch; the target refuses a stale one."""
    node_a, node_b = RecordingNode("node_g_reducer"), RecordingNode("node_g_reducer")
    a = _runtime("runtime-a", etcd.endpoint, {"node_g_reducer": ("primary", node_a)})
    b = _runtime("runtime-b", etcd.endpoint, {"node_g_reducer": ("primary", node_b)})
    guard = HandlerPlacementEpochGuard()
    await a.start()
    await b.start()
    try:
        await _until(lambda: bool(node_a.times("start")), 10)
        old = a.stamp("node_g_reducer", "sent before the failover")
        await a.stop()
        await _until(lambda: bool(node_b.times("start")), 15)
        new = b.stamp("node_g_reducer", "from the new primary")
        assert new.placement_epoch > old.placement_epoch
        assert guard.accept(new)
        assert not guard.accept(old)
    finally:
        await b.stop()


async def test_ac4_serving_counts_fail_when_one_of_two_runtimes_stops(
    etcd: EtcdProcess,
) -> None:
    """AC4: serving counts per node; fewer than 2 serving runtimes reads FAIL."""
    nodes_a: _Nodes = {
        "node_s_reducer": ("primary", RecordingNode("node_s_reducer")),
        "node_s_effect": ("any", RecordingNode("node_s_effect")),
    }
    nodes_b: _Nodes = {
        "node_s_reducer": ("primary", RecordingNode("node_s_reducer")),
        "node_s_effect": ("any", RecordingNode("node_s_effect")),
    }
    a = _runtime("runtime-a", etcd.endpoint, nodes_a)
    b = _runtime("runtime-b", etcd.endpoint, nodes_b)
    await a.start()
    await b.start()
    try:
        both = await a.serving_report()
        print(f"AC4 two runtimes: {both.model_dump(mode='json')}")
        assert {c.node_key: c.serving for c in both.counts} == {
            "node_s_effect": 2,
            "node_s_reducer": 2,
        }
        assert both.verdict is EnumPlacementHealthVerdict.OK

        await b.stop()
        one = await a.serving_report()
        print(f"AC4 one runtime: {one.model_dump(mode='json')}")
        assert {c.node_key: c.serving for c in one.counts} == {
            "node_s_effect": 1,
            "node_s_reducer": 1,
        }
        assert one.verdict is EnumPlacementHealthVerdict.FAIL
        assert one.under_served == ("node_s_effect", "node_s_reducer")
    finally:
        await a.stop()


async def test_ac5_primary_node_not_started_when_etcd_unreachable(
    etcd: EtcdProcess,
) -> None:
    """AC5: with etcd stopped the primary node's handlers never run; any nodes do."""
    primary, anywhere = RecordingNode("node_p_reducer"), RecordingNode("node_p_effect")
    etcd.stop()
    runtime = _runtime(
        "runtime-a",
        etcd.endpoint,
        {"node_p_reducer": ("primary", primary), "node_p_effect": ("any", anywhere)},
    )
    await runtime.start()
    try:
        await asyncio.sleep(3.0)
        print(
            f"AC5 etcd stopped: primary log {primary.log[:3]}, any started {bool(anywhere.times('start'))}"
        )
        assert primary.log == []
        assert not runtime.may_publish("node_p_reducer")
        assert anywhere.times("start")
        assert "node_p_reducer" not in runtime.started_nodes()

        # The hold lifts once etcd is back: the node starts after its election.
        etcd.start()
        await _until(lambda: bool(primary.times("start")), 15)
        assert primary.times("rebuild")[0] <= primary.times("start")[0]
    finally:
        await runtime.stop()
