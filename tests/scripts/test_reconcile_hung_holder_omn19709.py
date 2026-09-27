# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for bounded reconcile-host execution (OMN-19709)."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_host_lock_holder_omn18608 import _ready
from tests.scripts.test_reconcile_host_omn17307 import Workspace, build_workspace
from tests.scripts.test_reconcile_lane_hook_arming_omn18260 import (
    _teach_dispatch_python_to_run_programs,
)
from tests.scripts.test_reconcile_workspace_venvs import _Workspace

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOST_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-host.sh"
_VENV_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"

EXIT_DECLINED = 4
EXIT_STEP_TIMEOUT = 5
EXIT_STALE_LIVE_HOLDER = 6


@pytest.fixture
def host_ws(tmp_path: Path) -> Workspace:
    return build_workspace(tmp_path)


def _host_env(ws: Workspace, **overrides: str) -> dict[str, str]:
    fixture_home = ws.root.parent / "fixture_home"
    fixture_home.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "HOME": str(fixture_home),
        "OMNI_HOME": str(ws.root),
        **overrides,
    }


def _run_host(
    ws: Workspace, *, env: dict[str, str] | None = None, timeout: float = 20
) -> subprocess.CompletedProcess[str]:
    """Run reconcile-host and clean up its process group if a regression hangs."""
    proc = subprocess.Popen(
        ["bash", str(ws.scripts / "reconcile-host.sh")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or _host_env(ws),
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


def _seed_live_lock(ws: Workspace, pid: int, *, age_seconds: int) -> Path:
    lock_dir = ws.root / ".onex-reconcile-host.lock"
    lock_dir.mkdir(parents=True)
    started_at = datetime.now(UTC) - timedelta(seconds=age_seconds)
    (lock_dir / "holder").write_text(
        f"pid={pid}\n"
        f"host={os.uname().nodename}\n"
        f"started_at={started_at.strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
        encoding="utf-8",
    )
    return lock_dir


def test_command_output_is_never_fed_to_read_through_a_here_string() -> None:
    for script in (_HOST_SCRIPT, _VENV_SCRIPT):
        source = script.read_text(encoding="utf-8")
        assert "<<<" not in source, (
            f"{script.name} still contains a here-string. Bash may fill the "
            "here-string pipe before starting its reader and deadlock when "
            "captured command output exceeds the pipe capacity."
        )


def test_lane_hook_report_handles_200_kib_status_output_within_30_seconds(
    tmp_path: Path,
) -> None:
    ws = _Workspace(tmp_path / "omni_home")
    _teach_dispatch_python_to_run_programs(ws)
    lane_identity = ws.root / "omniclaude" / "scripts" / "lane_identity.py"
    lane_identity.parent.mkdir(parents=True, exist_ok=True)
    lane_identity.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if sys.argv[1:] == ['status']:\n"
        "    sys.stdout.write(('x' * 100 + '\\n') * 2048)\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    lane_identity.chmod(0o755)
    fake_home = ws.root / "fakehome"
    fake_home.mkdir()
    env = ws.env()
    env["HOME"] = str(fake_home)
    env["ONEX_LANE_IDENTITY_SCRIPT"] = str(lane_identity)

    started = time.monotonic()
    result = subprocess.run(
        ["bash", str(_VENV_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )

    assert time.monotonic() - started < 30
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("x" * 100) == 2048


def test_venv_delegate_timeout_kills_its_group_and_releases_lock(
    host_ws: Workspace,
) -> None:
    _ready(host_ws, body="sleep 600")
    env = _host_env(host_ws, ONEX_RECONCILE_STEP_TIMEOUT_S="2")

    started = time.monotonic()
    result = _run_host(host_ws, env=env)

    assert time.monotonic() - started < 20
    assert result.returncode == EXIT_STEP_TIMEOUT, result.stderr
    assert (
        "[reconcile-host] TIMEOUT: venv delegate exceeded 2s; killed its process group"
    ) in result.stderr
    assert not (host_ws.root / ".onex-reconcile-host.lock").exists()


def test_old_live_holder_is_reported_distinctly_and_never_killed(
    host_ws: Workspace,
) -> None:
    live = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        _seed_live_lock(host_ws, live.pid, age_seconds=7_500)
        _ready(host_ws)

        result = _run_host(host_ws)

        assert result.returncode == EXIT_STALE_LIVE_HOLDER, result.stderr
        assert (
            f"[reconcile-host] STALE-LIVE-HOLDER: pid {live.pid} on "
            f"{os.uname().nodename} has held the lock for "
        ) in result.stderr
        assert "(max 7200s); this run does not kill it" in result.stderr
        assert "Remedy:" in result.stderr
        assert live.poll() is None, "the stale live lock holder was killed"
        assert not host_ws.delegate_witness.exists()
    finally:
        live.terminate()
        live.wait()


def test_young_live_holder_is_still_an_ordinary_decline(host_ws: Workspace) -> None:
    live = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        _seed_live_lock(host_ws, live.pid, age_seconds=60)
        _ready(host_ws)

        result = _run_host(
            host_ws,
            env=_host_env(host_ws, ONEX_RECONCILE_MAX_HOLDER_AGE_S="120"),
        )

        assert result.returncode == EXIT_DECLINED, result.stderr
        assert "STALE-LIVE-HOLDER" not in result.stderr
        assert "nothing to do" in result.stderr
        assert live.poll() is None
        assert not host_ws.delegate_witness.exists()
    finally:
        live.terminate()
        live.wait()
