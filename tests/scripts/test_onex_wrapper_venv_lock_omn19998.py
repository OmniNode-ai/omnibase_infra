# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``scripts/onex`` waits for a venv-only reconcile pass as it does for a host pass.

OMN-19998. The host drift fix reinstalls the dispatch venv after every
omnimarket merge with a venv-only pass that holds ``.onex-reconcile-venv.lock``
and not ``.onex-reconcile-host.lock``. The install moves the venv off the floor's
omnimarket commit and the same pass stamps the floor at its end. The wrapper
waited only on the host lock, so every evidence command refused with exit 3 for
the whole install. It now waits on either lock. Waiting never lowers the bar:
a floor still not OK when the wait ends is refused exactly as before.

Hermetic: a fake OMNI_HOME, a hand-written floor and a stub CLI entrypoint.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_WRAPPER_SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "onex"
_EXIT_BELOW_FLOOR = 3
_SENTINEL_OK = 41
GOOD = "a" * 40
STALE = "e" * 40


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        scripts = root / "omnibase_infra" / "scripts"
        scripts.mkdir(parents=True)
        self.wrapper = scripts / "onex"
        shutil.copy(_WRAPPER_SOURCE, self.wrapper)
        self.wrapper.chmod(0o755)
        site = root / ".onex-dispatch-venv" / "lib" / "python3.12" / "site-packages"
        dist_info = site / "omnimarket-0.4.11.dist-info"
        dist_info.mkdir(parents=True)
        (dist_info / "direct_url.json").write_text(
            json.dumps({"vcs_info": {"vcs": "git", "commit_id": GOOD}}) + "\n",
            encoding="utf-8",
        )
        entrypoint = root / ".onex-dispatch-venv" / "bin" / "onex"
        entrypoint.parent.mkdir(parents=True)
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
    def venv_lock(self) -> Path:
        return self.root / ".onex-reconcile-venv.lock"

    @property
    def host_lock(self) -> Path:
        return self.root / ".onex-reconcile-host.lock"

    @property
    def argv_log(self) -> Path:
        return self.root / "argv.log"

    def write_floor(self, commit: str) -> None:
        tmp = self.floor.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"distributions": {}, "omnimarket_commit": commit}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(self.floor)

    def run(self, *args: str, wait_s: int = 30) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.update(
            OMNIBASE_PATH=str(self.root),
            PATH="/usr/bin:/bin:/usr/sbin:/sbin",
            ONEX_FLOOR_LOCK_WAIT_S=str(wait_s),
            ONEX_FLOOR_LOCK_POLL_S="1",
        )
        for key in (
            "ONEX_RECONCILE_RECEIPT",
            "ONEX_DISPATCH_VENV",
            "ONEX_RECONCILE_STEP_TIMEOUT_S",
        ):
            env.pop(key, None)
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
    return _Workspace(tmp_path / "workspace")


def test_a_venv_only_pass_that_stamps_the_floor_lets_the_command_run(
    ws: _Workspace,
) -> None:
    """The recorded case: the floor reads below only while the venv pass installs."""
    ws.write_floor(STALE)
    ws.venv_lock.mkdir()

    def _venv_pass_finishes() -> None:
        ws.write_floor(GOOD)
        ws.venv_lock.rmdir()

    timer = threading.Timer(3.0, _venv_pass_finishes)
    timer.start()
    try:
        proc = ws.run("delegate", "hello")
    finally:
        timer.join()
    assert proc.returncode == _SENTINEL_OK, proc.stderr
    assert f"a peer reconcile holds {ws.venv_lock}" in proc.stderr
    assert "the peer reconcile proved the floor" in proc.stderr
    assert ws.argv_log.read_text(encoding="utf-8").strip() == "delegate hello"


def test_a_host_pass_is_still_waited_for_and_named(ws: _Workspace) -> None:
    ws.write_floor(STALE)
    ws.host_lock.mkdir()

    def _host_pass_finishes() -> None:
        ws.write_floor(GOOD)
        ws.host_lock.rmdir()

    timer = threading.Timer(2.0, _host_pass_finishes)
    timer.start()
    try:
        proc = ws.run("delegate", "hello")
    finally:
        timer.join()
    assert proc.returncode == _SENTINEL_OK, proc.stderr
    assert f"a peer reconcile holds {ws.host_lock}" in proc.stderr


def test_a_venv_pass_that_never_proves_the_floor_still_refuses(ws: _Workspace) -> None:
    ws.write_floor(STALE)
    ws.venv_lock.mkdir()
    proc = ws.run("delegate", "x", wait_s=2)
    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert "still holds the lock" in proc.stderr
    assert "REFUSED" in proc.stderr
    assert ws.venv_lock.is_dir(), "the wrapper must never reclaim its peer's lock"
    assert not ws.argv_log.exists()


def test_a_venv_pass_that_finishes_without_proving_the_floor_refuses(
    ws: _Workspace,
) -> None:
    ws.write_floor(STALE)
    ws.venv_lock.mkdir()
    timer = threading.Timer(2.0, ws.venv_lock.rmdir)
    timer.start()
    try:
        proc = ws.run("delegate", "x")
    finally:
        timer.join()
    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert "finished after about" in proc.stderr
    assert not ws.argv_log.exists()


def test_no_peer_and_a_stale_floor_refuses_without_waiting(ws: _Workspace) -> None:
    ws.write_floor(STALE)
    proc = ws.run("delegate", "x")
    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert "waiting up to" not in proc.stderr
