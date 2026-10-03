# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``scripts/onex`` below the floor: peer-lock wait and a refusal that names its cause.

OMN-20111. When a peer holds the host lock, the peer is doing the very work
that would clear the refusal, so the wrapper waits a bounded time, re-reading
the floor, instead of refusing while a
stamp is seconds away. Waiting never lowers the bar: a floor that is still not
OK when the wait ends is refused exactly as before.

When no floor is known, the refusal names failing dispatch-premise surfaces
from the last reconcile receipt. A BELOW refusal names only the proven restore.

Hermetic: a peer lock directory, a hand-written floor and receipt, and a fake
CLI entrypoint that records its argv.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WRAPPER_SOURCE = _REPO_ROOT / "scripts" / "onex"
_EXIT_BELOW_FLOOR = 3
_SENTINEL_OK = 41

GOOD = "a" * 40
STALE = "e" * 40


class _Workspace:
    """A fake OMNI_HOME with the wrapper installed and a stub dispatch venv."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.scripts_dir = root / "omnibase_infra" / "scripts"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.wrapper = self.scripts_dir / "onex"
        shutil.copy(_WRAPPER_SOURCE, self.wrapper)
        self.wrapper.chmod(0o755)
        self.venv_site = (
            root / ".onex-dispatch-venv" / "lib" / "python3.12" / "site-packages"
        )
        self.venv_bin = root / ".onex-dispatch-venv" / "bin"
        self.venv_site.mkdir(parents=True, exist_ok=True)
        self.venv_bin.mkdir(parents=True, exist_ok=True)
        dist_info = self.venv_site / "omnimarket-0.4.11.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: omnimarket\nVersion: 0.4.11\n",
            encoding="utf-8",
        )
        (dist_info / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": "https://github.com/OmniNode-ai/omnimarket.git",
                    "vcs_info": {"vcs": "git", "commit_id": GOOD},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        entrypoint = self.venv_bin / "onex"
        entrypoint.write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$*" >> {json.dumps(str(self.argv_log))}\n'
            f"exit {_SENTINEL_OK}\n",
            encoding="utf-8",
        )
        entrypoint.chmod(0o755)

    @property
    def floor(self) -> Path:
        return self.root / ".onex-workspace-floor.json"

    @property
    def receipt(self) -> Path:
        return self.root / ".onex-workspace-reconcile.json"

    @property
    def lock_dir(self) -> Path:
        return self.root / ".onex-reconcile-host.lock"

    @property
    def argv_log(self) -> Path:
        return self.root / "argv.log"

    def write_floor(self, commit: str) -> None:
        payload = (
            json.dumps(
                {
                    "schema": "onex.workspace.floor.v1",
                    "generated_at": "2026-09-29T00:00:00Z",
                    "host": "test",
                    "omni_home": str(self.root),
                    "distributions": {},
                    "omnimarket_commit": commit,
                },
                indent=2,
            )
            + "\n"
        )
        tmp = self.floor.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.floor)

    def write_receipt(self) -> None:
        self.receipt.write_text(
            "{\n"
            '  "schema": "onex.workspace.reconcile.v1",\n'
            '  "generated_at": "2026-09-29T21:00:00Z",\n'
            '  "mode": "repair",\n'
            '  "surfaces": [\n'
            '    {"surface": "clone:omnimarket", "verdict": "DID_NOT_MOVE", "dispatch_premise": true, "remedy": "bash /x/converge-canonical-clone.sh omnimarket --execute", "detail": "observed a but target is b"},\n'
            '    {"surface": "clone:omnibase_core", "verdict": "DID_NOT_MOVE", "dispatch_premise": false, "remedy": "bash /x/converge-canonical-clone.sh omnibase_core --execute", "detail": "x"},\n'
            '    {"surface": "venv:omnimarket", "verdict": "ALREADY_AT_TARGET", "dispatch_premise": true, "remedy": "", "detail": "already at a"}\n'
            "  ],\n"
            '  "failures": 2,\n'
            '  "dispatch_premise_failures": 1\n'
            "}\n",
            encoding="utf-8",
        )

    def run(
        self,
        *args: str,
        wait_s: int = 30,
        poll_s: int = 1,
    ) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["OMNI_HOME"] = str(self.root)
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        env["ONEX_FLOOR_LOCK_WAIT_S"] = str(wait_s)
        env["ONEX_FLOOR_LOCK_POLL_S"] = str(poll_s)
        env.pop("ONEX_RECONCILE_RECEIPT", None)
        env.pop("ONEX_DISPATCH_VENV", None)
        return subprocess.run(
            ["bash", str(self.wrapper), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )


@pytest.fixture
def ws(tmp_path: Path) -> _Workspace:
    return _Workspace(tmp_path / "omni_home")


def test_a_peer_that_stamps_the_floor_within_the_wait_lets_delegate_run(
    ws: _Workspace,
) -> None:
    """When a peer removes its lock after stamping a good floor, the wrapper
    must poll, notice the lock is gone, re-read the floor,
    print the proof message to stderr, and exec the entrypoint (sentinel 41)."""
    ws.write_floor(STALE)
    ws.lock_dir.mkdir()

    def _peer_finishes() -> None:
        ws.write_floor(GOOD)
        ws.lock_dir.rmdir()

    timer = threading.Timer(3.0, _peer_finishes)
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


def test_a_peer_holding_the_lock_past_the_budget_still_refuses(ws: _Workspace) -> None:
    """If the peer lock directory never goes away within the wait budget, the
    wrapper must stop waiting, refuse with exit 3, and never exec the venv
    entrypoint."""
    ws.write_floor(STALE)
    ws.lock_dir.mkdir()
    t0 = time.monotonic()
    proc = ws.run("delegate", "x", wait_s=2, poll_s=1)
    elapsed = time.monotonic() - t0
    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert "still holds the lock" in proc.stderr
    assert "REFUSED" in proc.stderr
    assert not ws.argv_log.exists()
    assert elapsed >= 2, "the wrapper must actually wait out its budget"


def test_a_peer_that_finishes_without_proving_the_floor_refuses(ws: _Workspace) -> None:
    """If the peer lock disappears while the floor is still stale, the wrapper
    must stop polling immediately (well under the full budget) and refuse with
    the 'finished after about' message instead of exec'ing the entrypoint."""
    ws.write_floor(STALE)
    ws.lock_dir.mkdir()
    timer = threading.Timer(2.0, ws.lock_dir.rmdir)
    timer.start()
    try:
        t0 = time.monotonic()
        proc = ws.run("delegate", "x", wait_s=30, poll_s=1)
        elapsed = time.monotonic() - t0
    finally:
        timer.join()
    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert "finished after about" in proc.stderr
    assert not ws.argv_log.exists()
    assert elapsed < 25


def test_the_refusal_names_the_blocking_surface_and_its_clearing_command(
    ws: _Workspace,
) -> None:
    """On refusal with a reconcile receipt present, the wrapper must name each
    failing dispatch-premise surface with its verdict and the receipt timestamp,
    print the clearing remedy, demark non-blocking failures, and stay silent
    about surfaces already at target."""
    ws.write_receipt()
    proc = ws.run("delegate", "x")
    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert (
        "blocking : clone:omnimarket DID_NOT_MOVE (last reconcile 2026-09-29T21:00:00Z)"
        in proc.stderr
    )
    assert (
        "clears by: bash /x/converge-canonical-clone.sh omnimarket --execute"
        in proc.stderr
    )
    assert "not blocking delegation: clone:omnibase_core DID_NOT_MOVE" in proc.stderr
    assert "venv:omnimarket" not in proc.stderr


def test_a_refusal_with_no_receipt_says_so(ws: _Workspace) -> None:
    """When the wrapper refuses and no reconcile receipt exists on disk it must
    still refuse with exit 3 and say the floor is blocked by unknown surfaces
    rather than staying silent or crashing."""
    proc = ws.run("delegate", "x")
    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert "no reconcile receipt" in proc.stderr


def test_an_ordinary_subcommand_never_waits(ws: _Workspace) -> None:
    """Only evidence subcommands like delegate are guarded by the floor; a
    plain subcommand must exec straight through to the venv entrypoint even
    with a stale floor and a live peer lock, never printing wait messages."""
    ws.write_floor(STALE)
    ws.lock_dir.mkdir()
    t0 = time.monotonic()
    proc = ws.run("info", wait_s=30, poll_s=1)
    elapsed = time.monotonic() - t0
    assert proc.returncode == _SENTINEL_OK, proc.stderr
    assert "waiting up to" not in proc.stderr
    assert elapsed < 25
