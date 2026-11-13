# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``scripts/onex`` below the floor: the staging lock counts as a peer too.

OMN-16932. While a reconcile is mid-move it holds a staging lock -- a symlink
next to the dispatch venv naming its holder -- and the venv it is rebuilding is
exactly the one the floor verdict is judged against. So the wrapper treats a
live staging lock as a peer reconcile: it waits a bounded time, re-reading the
floor, instead of refusing while the stamp is seconds away. A lock whose holder
is gone on this host is a corpse and is not waited on, and a lock held past the
budget still ends in a refusal that names its cause.

Hermetic: a hand-written floor, a staging lock symlink, and a fake CLI
entrypoint that records its argv.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.scripts.test_onex_wrapper_peer_lock_omn20111 import (
    _EXIT_BELOW_FLOOR,
    _SENTINEL_OK,
    GOOD,
    STALE,
    _Workspace,
)

pytestmark = pytest.mark.unit

_THIS_HOST = socket.gethostname()
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"


def _take_staging_lock(ws: _Workspace, pid: int, host: str) -> None:
    """Hold the staging lock the way a mid-move reconcile would: a symlink
    whose target text is the holder record."""
    record = f"pid={pid};host={host};at=2026-11-13T00:00:00Z"
    (ws.root / ".onex-dispatch-venv.rebuilding.lock").symlink_to(record)


@pytest.fixture
def ws(tmp_path: Path) -> _Workspace:
    workspace = _Workspace(tmp_path / "workspace")
    # The wrapper asks the node (node_venv_atomic_install) whether the staging lock is
    # held, through its runtime command under the workspace's own omnibase_infra/src.
    (workspace.root / "omnibase_infra" / "src").symlink_to(_REPO_SRC)
    return workspace


def test_a_launch_during_a_staging_move_waits_and_then_places(ws: _Workspace) -> None:
    """While the staging lock is live and the floor is stale, the wrapper must
    wait; once the mover unlinks the lock having stamped a good floor, the
    wrapper must proceed and exec the entrypoint (sentinel 41)."""
    ws.write_floor(STALE)
    _take_staging_lock(ws, os.getpid(), _THIS_HOST)

    def _mover_finishes() -> None:
        ws.write_floor(GOOD)
        (ws.root / ".onex-dispatch-venv.rebuilding.lock").unlink()

    timer = threading.Timer(3.0, _mover_finishes)
    timer.start()
    try:
        t0 = time.monotonic()
        proc = ws.run("delegate", "hello")
        elapsed = time.monotonic() - t0
    finally:
        timer.join()
    assert proc.returncode == _SENTINEL_OK, proc.stderr
    assert "the peer reconcile proved the floor" in proc.stderr
    assert (
        ws.argv_log.exists()
        and ws.argv_log.read_text(encoding="utf-8").strip() == "delegate hello"
    )
    assert elapsed >= 2.5


def test_a_dead_staging_holder_is_not_waited_on(ws: _Workspace) -> None:
    """A staging lock naming a pid that has exited on this host is a corpse:
    the wrapper must refuse at once (exit 3), never printing a wait message."""
    mover = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    mover.wait()
    ws.write_floor(STALE)
    _take_staging_lock(ws, mover.pid, _THIS_HOST)

    t0 = time.monotonic()
    proc = ws.run("delegate", "hello", wait_s=30)
    elapsed = time.monotonic() - t0
    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert "waiting up to" not in proc.stderr
    assert not ws.argv_log.exists()
    assert elapsed < 10


def test_a_staging_lock_held_past_the_budget_still_refuses(ws: _Workspace) -> None:
    """If the staging lock stays live past the wait budget, the wrapper must
    stop waiting, refuse with exit 3, and never exec the venv entrypoint."""
    ws.write_floor(STALE)
    _take_staging_lock(ws, os.getpid(), _THIS_HOST)
    proc = ws.run("delegate", "x", wait_s=2, poll_s=1)
    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert "still holds the lock" in proc.stderr
    assert "REFUSED" in proc.stderr
    assert not ws.argv_log.exists()


def test_without_a_staging_lock_a_below_floor_launch_refuses_at_once(
    ws: _Workspace,
) -> None:
    """The control: with no peer lock of any kind, a stale floor is refused
    immediately -- so the wait in the other tests is the staging lock's doing."""
    ws.write_floor(STALE)
    t0 = time.monotonic()
    proc = ws.run("delegate", "x")
    elapsed = time.monotonic() - t0
    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert "REFUSED" in proc.stderr
    assert not ws.argv_log.exists()
    assert elapsed < 5


def test_a_holder_on_another_host_counts_as_live(ws: _Workspace) -> None:
    """A staging lock naming a holder on another host cannot be judged from
    here, so it counts as live: the wrapper waits its budget, then refuses."""
    ws.write_floor(STALE)
    _take_staging_lock(ws, 1, "some-other-host.invalid")
    proc = ws.run("delegate", "x", wait_s=2, poll_s=1)
    assert "waiting up to" in proc.stderr
    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert not ws.argv_log.exists()
