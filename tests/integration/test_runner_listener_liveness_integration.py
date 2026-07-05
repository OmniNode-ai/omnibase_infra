# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for OMN-13915 runner listener liveness."""

from __future__ import annotations

import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
HEALTHCHECK = REPO_ROOT / "docker" / "runners" / "healthcheck.sh"


def _run_healthcheck(runner_home: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["RUNNER_HOME"] = str(runner_home)
    env["RUNNER_HEALTH_EGRESS_CHECK"] = "0"
    env["RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS"] = "900"
    return subprocess.run(
        ["bash", str(HEALTHCHECK)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


@pytest.fixture
def runner_home(tmp_path: Path) -> Path:
    home = tmp_path / "actions-runner"
    (home / "bin").mkdir(parents=True)
    (home / "_diag").mkdir()
    listener = home / "bin" / "Runner.Listener"
    listener.write_text("#!/bin/sh\nsleep 120\n", encoding="utf-8")
    listener.chmod(listener.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    (home / "_diag" / "Runner_20260704-000000-utc.log").write_text(
        "heartbeat\n", encoding="utf-8"
    )
    return home


def test_runner_healthcheck_fails_when_listener_dies(runner_home: Path) -> None:
    listener = runner_home / "bin" / "Runner.Listener"
    proc = subprocess.Popen([str(listener)])
    try:
        time.sleep(0.5)
        healthy = _run_healthcheck(runner_home)
        assert healthy.returncode == 0, healthy.stdout + healthy.stderr

        proc.terminate()
        proc.wait(timeout=10)
        time.sleep(0.5)

        unhealthy = _run_healthcheck(runner_home)
        assert unhealthy.returncode == 1
        assert "not running" in unhealthy.stdout
    finally:
        if proc.poll() is None:
            proc.kill()
