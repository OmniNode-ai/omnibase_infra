# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fail-fast + fail-loud regression tests for onex-build-loop-trigger.sh (OMN-15179).

The previous copy of this script (unversioned host state at
``~/.local/bin/onex-build-loop-trigger.sh`` on omninode-pc) hardcoded the
dev-lane container name and swallowed a failing ``docker exec`` -- the daily
systemd-timer trigger printed "Build loop triggered" even when the target
container did not exist, so the trigger silently did nothing every day.

Properties proven here (real bash execution of the actual script, not a
surrogate/rewrite -- ``docker`` and ``python3`` are shimmed on PATH so the
test is hermetic):

1. With ``ONEX_BUILD_LOOP_REDPANDA_CONTAINER`` unset, the script exits
   non-zero immediately and never invokes ``docker`` (fail-fast, no silent
   default).
2. With the env var set but ``docker exec`` failing (simulating "No such
   container"), the script exits non-zero and prints a loud error -- it must
   NOT print "Build loop triggered" (this is the exact RED this ticket exists
   to prove against the pre-fix script).
3. With ``docker exec`` succeeding, the script exits 0 and prints the success
   line, using the container name from the required env var (not a
   hardcoded default).
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "onex-build-loop-trigger.sh"


def _shim(bin_dir: Path, name: str, body: str) -> None:
    shim = bin_dir / name
    shim.write_text(f"#!/usr/bin/env bash\n{body}\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    tmp_path: Path,
    *,
    docker_exit: int,
    env_container: str | None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calllog = tmp_path / "docker_calls.log"
    calllog.write_text("")

    _shim(
        bin_dir,
        "docker",
        f'echo "docker $*" >> "{calllog}"\ncat >/dev/null\nexit {docker_exit}',
    )
    # Real python3 is fine for uuid generation; no need to shim it.

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env.pop("ONEX_BUILD_LOOP_REDPANDA_CONTAINER", None)
    if env_container is not None:
        env["ONEX_BUILD_LOOP_REDPANDA_CONTAINER"] = env_container

    return subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )


def test_missing_required_env_var_fails_fast_without_invoking_docker(
    tmp_path: Path,
) -> None:
    proc = _run(tmp_path, docker_exit=0, env_container=None)

    assert proc.returncode != 0
    assert "ONEX_BUILD_LOOP_REDPANDA_CONTAINER" in proc.stderr
    assert (
        not (tmp_path / "docker_calls.log").exists()
        or (tmp_path / "docker_calls.log").read_text() == ""
    )


def test_failing_docker_exec_exits_nonzero_and_does_not_claim_success(
    tmp_path: Path,
) -> None:
    """RED against the pre-fix script: a failing docker exec must never be
    followed by a "Build loop triggered" success line."""
    proc = _run(
        tmp_path,
        docker_exit=1,
        env_container="omnibase-infra-stability-test-redpanda",
    )

    assert proc.returncode != 0
    assert "Build loop triggered" not in proc.stdout
    assert "ERROR" in proc.stderr
    assert "omnibase-infra-stability-test-redpanda" in proc.stderr


def test_successful_docker_exec_reports_success_with_configured_container(
    tmp_path: Path,
) -> None:
    proc = _run(
        tmp_path,
        docker_exit=0,
        env_container="omnibase-infra-stability-test-redpanda",
    )

    assert proc.returncode == 0
    assert "Build loop triggered" in proc.stdout
    assert "omnibase-infra-stability-test-redpanda" in proc.stdout
