# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18630: the flock-less lock of ensure_ci_env.sh names its holder and ends.

A directory lock is not released by the kernel when its holder is SIGKILLed.
Before this change the wait polled an empty directory forever; now the holder
(pid host started) is written into the lock and the wait ends in a loud failure
that names it. These tests run the real script under a PATH with no ``flock``
against a leaked lock directory, and compare the run with a recorded one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "ensure_ci_env.sh"
DIGEST = REPO_ROOT / "scripts" / "ci" / "ci_env_digest.py"

LEAKED_HOLDER = "4242 leaked-host 2026-10-11T00:00:00Z"
INSTALL_ARGS = "--frozen --all-extras --all-groups --no-install-project"
TOOLS = ("cat", "date", "dirname", "hostname", "mkdir", "rm", "sleep")


def run_against_leaked_lock(tmp_path: Path, holder: str | None) -> dict[str, object]:
    """Run the real script, flock absent, against a lock a dead process leaked."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    # The script resolves uv before it takes the lock and never runs it on this path.
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uv.chmod(0o755)
    for tool in TOOLS:
        found = shutil.which(tool)
        assert found, f"{tool} is needed on the test PATH"
        (bin_dir / tool).symlink_to(found)
    assert not (bin_dir / "flock").exists()

    env_root = tmp_path / "envs"
    env = {
        "PATH": str(bin_dir),
        "HOME": str(tmp_path),
        "RUNNER_TEMP": str(tmp_path),
        "OMNI_CI_ENV_ROOT": str(env_root),
        "OMNI_CI_REPO": "lockdir-fixture",
        "PYTHON_VERSION": "3.12",
        "UV_VERSION": "0.6.14",
        "OMNI_CI_ENV_INSTALL_ARGS": INSTALL_ARGS,
        "OMNI_CI_ENV_LOCK_TIMEOUT_SECONDS": "0",
    }
    digest = subprocess.run(
        [
            sys.executable,
            str(DIGEST),
            "--repo-root",
            str(REPO_ROOT),
            "--python-version",
            "3.12",
            "--uv-version",
            "0.6.14",
            "--install-args",
            INSTALL_ARGS,
            "--extra",
            "",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock_path = env_root / "lockdir-fixture" / ".locks" / f"{digest}.lockdir"
    lock_path.mkdir(parents=True)
    if holder is not None:
        (lock_path / "holder").write_text(holder + "\n", encoding="utf-8")

    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stderr": result.stderr.replace(str(lock_path), "<LOCKDIR>").strip(),
        "lock_left_in_place": lock_path.is_dir(),
    }


def test_a_leaked_lock_ends_the_wait_and_names_its_holder(tmp_path: Path) -> None:
    run = run_against_leaked_lock(tmp_path, LEAKED_HOLDER)
    assert run["returncode"] == 1
    assert "Timed out waiting for shared CI env lock <LOCKDIR>" in str(run["stderr"])
    assert f"holder (pid host started): {LEAKED_HOLDER}" in str(run["stderr"])
    # The script never removes a lock it did not take.
    assert run["lock_left_in_place"] is True


def test_a_lock_with_no_holder_record_is_named_as_unrecorded(tmp_path: Path) -> None:
    """The control: the message reads the holder file rather than printing a constant."""
    run = run_against_leaked_lock(tmp_path, None)
    assert run["returncode"] == 1
    assert "holder (pid host started): unrecorded" in str(run["stderr"])
    assert LEAKED_HOLDER not in str(run["stderr"])


@pytest.mark.live_contact(
    "tests/ci/fixtures/ensure_ci_env_lockdir_timeout_omn18630.json"
)
def test_the_run_matches_the_recorded_run(
    tmp_path: Path, recorded_response: dict[str, object]
) -> None:
    run = run_against_leaked_lock(tmp_path, LEAKED_HOLDER)
    assert run == recorded_response["response"]
    assert os.access(SCRIPT, os.X_OK)
