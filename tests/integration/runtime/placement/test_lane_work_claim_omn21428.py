# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21428 AC1 and AC2 against a real etcd.

A lane runner process acquires a ``lane-work`` lease on the test etcd and
renews it every ``ttl_s/3``. Its lease replies stand in for the lease
orchestrator's events; the test feeds them to ``node_projection_lease``'s
ledger handler, which writes the ledger rows to a temporary ledger file. The
runner has no ledger path. AC1 reads the CLAIM row's actor; AC2 SIGKILLs the
runner and reads the TERMINAL row the projection writes at lease expiry.
"""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from omnibase_infra.handlers.lease import HandlerLeaseStore
from omnibase_infra.handlers.lease.handler_lease_ledger import (
    LANE_WORK_KIND,
    PROJECTION_ACTOR,
    HandlerLeaseLedger,
    LedgerFileAppender,
)
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import ModelLeaseKindPolicy, ModelLeaseReply

from .conftest import EtcdProcess

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

PROC = Path(__file__).with_name("lane_work_runner_proc.py")
TTL_S = 3
# etcd expires a lease up to one TTL after its last keepalive; the sweep runs
# every SWEEP_S. A CLAIM open longer than this after the kill is open past expiry.
SWEEP_S = 0.25
EXPIRY_BOUND_S = TTL_S + 1.5


def _cells(row: str) -> dict[str, str]:
    parts = [part.strip() for part in row.split(" | ")]
    cells = {"ts": parts[0], "verb": parts[1]}
    for part in parts[2:]:
        name, _, value = part.partition("=")
        cells.setdefault(name, value)
    return cells


def _rows(ledger_file: Path) -> list[dict[str, str]]:
    if not ledger_file.exists():
        return []
    return [_cells(line) for line in ledger_file.read_text().splitlines() if line]


@pytest.fixture
async def store(etcd: EtcdProcess) -> AsyncIterator[HandlerLeaseStore]:
    adapter = HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=10.0))
    try:
        yield HandlerLeaseStore(
            adapter, [ModelLeaseKindPolicy(kind=LANE_WORK_KIND, ttl_s=TTL_S)]
        )
    finally:
        await adapter.aclose()


def _start_runner(etcd: EtcdProcess, lane: str, *extra: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            str(PROC),
            f"--endpoint={etcd.endpoint}",
            f"--lane={lane}",
            f"--ttl-s={TTL_S}",
            *extra,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )


async def _next_reply(proc: subprocess.Popen[str]) -> ModelLeaseReply:
    assert proc.stdout is not None
    line = await asyncio.wait_for(asyncio.to_thread(proc.stdout.readline), 20)
    assert line, "the lane runner exited without a reply"
    return ModelLeaseReply.model_validate(json.loads(line))


async def test_ac1_claim_row_is_written_by_the_projection(
    etcd: EtcdProcess, store: HandlerLeaseStore, tmp_path: Path
) -> None:
    ledger_file = tmp_path / "ROLLING_WORK_LEDGER.md"
    ledger = HandlerLeaseLedger(LedgerFileAppender(ledger_file))
    proc = _start_runner(etcd, "ac1-lane", "--release-after=1.5", "--outcome=done")
    try:
        granted = await _next_reply(proc)
        assert granted.outcome == "lease-granted"
        assert _rows(ledger_file) == [], "the runner wrote a ledger row"
        await ledger.handle(granted)
        (claim,) = _rows(ledger_file)
        assert claim["verb"] == "CLAIM"
        assert claim["actor"] == PROJECTION_ACTOR
        assert claim["lane"] == "ac1-lane"
        assert claim["fence"] == str(granted.fence)
        while (reply := await _next_reply(proc)).outcome == "lease-renewed":
            await ledger.handle(reply)
        assert reply.outcome == "lease-released"
        await ledger.handle(reply)
        assert proc.wait(timeout=20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    rows = _rows(ledger_file)
    assert [row["verb"] for row in rows] == ["CLAIM", "TERMINAL"]
    assert {row["actor"] for row in rows} == {PROJECTION_ACTOR}
    assert rows[1]["outcome"] == "done"
    assert rows[1]["closes-CLAIM"] == rows[0]["ts"]
    assert await store.show(LANE_WORK_KIND, "lane:ac1-lane") == []


async def test_ac2_killed_runner_gets_terminal_at_lease_expiry(
    etcd: EtcdProcess, store: HandlerLeaseStore, tmp_path: Path
) -> None:
    ledger_file = tmp_path / "ROLLING_WORK_LEDGER.md"
    ledger = HandlerLeaseLedger(LedgerFileAppender(ledger_file))
    proc = _start_runner(etcd, "ac2-lane")
    try:
        granted = await _next_reply(proc)
        await ledger.handle(granted)
        granted_at = time.monotonic()
        # Renewals keep the grant past its own TTL: the sweep writes nothing.
        while time.monotonic() - granted_at < TTL_S + 1:
            reply = await _next_reply(proc)
            assert reply.outcome == "lease-renewed"
            assert reply.fence == granted.fence
            assert await ledger.sweep_expired(store) == ()
        last_renewal = time.monotonic()
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        killed = time.monotonic()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    terminal_at: float | None = None
    while time.monotonic() - killed < EXPIRY_BOUND_S + 5:
        if await ledger.sweep_expired(store):
            terminal_at = time.monotonic()
            break
        await asyncio.sleep(SWEEP_S)
    assert terminal_at is not None, "the CLAIM stayed open with no TERMINAL"
    open_after_kill = terminal_at - killed
    print(
        f"AC2 receipt: lane=ac2-lane fence={granted.fence} ttl_s={TTL_S} "
        f"terminal_after_kill_s={open_after_kill:.2f} "
        f"terminal_after_last_renewal_s={terminal_at - last_renewal:.2f} "
        f"bound_s={EXPIRY_BOUND_S}"
    )
    assert open_after_kill <= EXPIRY_BOUND_S
    rows = _rows(ledger_file)
    assert [row["verb"] for row in rows] == ["CLAIM", "TERMINAL"]
    terminal = rows[1]
    assert terminal["actor"] == PROJECTION_ACTOR
    assert terminal["cause"] == "lease-expired"
    assert terminal["lane"] == "ac2-lane"
    assert terminal["closes-CLAIM"] == rows[0]["ts"]
    assert terminal["lease"] == str(granted.lease_id)
    assert ledger.open_claims() == {}
