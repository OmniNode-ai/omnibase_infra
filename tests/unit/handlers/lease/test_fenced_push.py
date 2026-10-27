# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fenced pushes report a lost grant and clean up their transport process tree."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.lease.handler_fenced_push import (
    GitCliPushTransport,
    HandlerFencedPush,
)
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import ModelFencedPushRequest

pytestmark = pytest.mark.unit


def _request(worktree: str = "/worker") -> ModelFencedPushRequest:
    return ModelFencedPushRequest(
        kind="landing-fix",
        resource="pr:repo#1",
        fence=42,
        worktree=worktree,
        branch="fix",
        expected_remote_sha="a" * 40,
    )


class FakeFenceEtcd(HandlerPlacementEtcd):
    """One live slot, replaceable while a transport accepts the push."""

    def __init__(self) -> None:
        self.fence: int | None = 42
        self.unreadable = False
        self.reads = 0

    async def get_prefix_entries(self, prefix: str) -> list[tuple[str, str, int, int]]:
        self.reads += 1
        if self.unreadable:
            raise InfraConnectionError("store unavailable")
        if self.fence is None:
            return []
        return [(f"{prefix}0", "{}", self.fence, 123)]


class FakePushTransport:
    def __init__(self, adapter: FakeFenceEtcd, during_push: str) -> None:
        self.adapter = adapter
        self.during_push = during_push

    async def push(self, request: ModelFencedPushRequest) -> tuple[bool, str]:
        assert self.adapter.reads == 1
        if self.during_push == "regrant":
            # Revoke the old slot and install a new grant before acceptance.
            self.adapter.fence = None
            self.adapter.fence = request.fence + 1
        elif self.during_push == "expire":
            self.adapter.fence = None
        elif self.during_push == "unreadable":
            self.adapter.unreadable = True
        return True, "accepted"


@pytest.mark.parametrize(
    ("during_push", "held"),
    [("held", True), ("regrant", False), ("expire", False), ("unreadable", False)],
)
async def test_accepted_push_reports_whether_the_fence_is_still_held(
    during_push: str, held: bool
) -> None:
    adapter = FakeFenceEtcd()
    result = await HandlerFencedPush(
        adapter, FakePushTransport(adapter, during_push).push
    ).handle(_request())
    assert result.outcome == "pushed"
    assert result.fence_held_after_push is held
    assert adapter.reads == 2


def _running(pid: int) -> bool:
    try:
        # An orphan may briefly remain a zombie until init reaps it.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancellation"])
async def test_push_stops_the_whole_transport_process_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    pids = tmp_path / "pids"
    git = tmp_path / "git"
    git.write_text(
        f"#!{sys.executable}\n"
        "# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.\n"
        "# SPDX-License-Identifier: MIT\n"
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"Path({str(pids)!r}).write_text(f'{{os.getpid()}} {{child.pid}}')\n"
        "time.sleep(30)\n"
    )
    git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    transport = GitCliPushTransport(timeout_s=30 if cancel else 1)
    task = asyncio.create_task(transport.push(_request(str(tmp_path))))
    process_ids: list[int] = []
    try:
        async with asyncio.timeout(5):
            while not pids.exists():
                await asyncio.sleep(0.01)
        process_ids = [int(pid) for pid in pids.read_text().split()]
        assert len(process_ids) == 2
        assert all(_running(pid) for pid in process_ids)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            accepted, detail = await task
            assert accepted is False
            assert "timed out" in detail
        async with asyncio.timeout(2):
            while any(_running(pid) for pid in process_ids):
                await asyncio.sleep(0.01)
        assert all(not _running(pid) for pid in process_ids)
    finally:
        for pid in process_ids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
