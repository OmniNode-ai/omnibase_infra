# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: real-docker proof that a directory mount sees a redeployed hook.

Negative control and fix, side by side on a real bind mount:

* the OLD layout mounts ``runner-job-started.sh`` as a single file;
* the NEW layout mounts the ``runners`` directory.

``scripts/deploy-runners.sh`` replaces files with ``rsync -a --checksum``
(write a temp file, rename over the target, so the target gets a new inode). The
old layout's container keeps the original inode, so it keeps serving the
pre-deploy bytes (on Linux; Docker Desktop reports ``No such file or
directory`` instead, which is the failure measured on 2026-10-16). The new
layout's container resolves the name again and sees the new bytes.

Marked ``heavy``: needs a docker daemon and the ``python:3.13-slim`` image, so it
is deselected from the PR splits and run on a lab host. It fails, never skips,
when docker is absent.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.heavy]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNERS_DIR = _REPO_ROOT / "docker" / "runners"
_IMAGE = "python:3.13-slim"
_V1 = "#!/bin/sh\necho hook-v1\n"
_V2 = "#!/bin/sh\necho hook-v2\n"


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=check,
    )


def _rsync_redeploy(staging: Path, runners: Path) -> None:
    """The deploy step: ``rsync -a --checksum``, which replaces the file."""
    subprocess.run(
        ["rsync", "-a", "--checksum", f"{staging}/", f"{runners}/"],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def hook_tree(tmp_path: Path) -> Path:
    assert shutil.which("docker"), "docker is required for this proof"
    assert shutil.which("rsync"), "rsync is required for this proof"
    _docker("info", "--format", "{{.ServerVersion}}")
    runners = tmp_path / "runners"
    runners.mkdir()
    (runners / "runner-job-started.sh").write_text(_V1, encoding="utf-8")
    shutil.copy(_RUNNERS_DIR / "healthcheck.sh", runners / "healthcheck.sh")
    return runners


def _start(name: str, mount: str) -> None:
    _docker("run", "-d", "--name", name, "-v", mount, _IMAGE, "sleep", "600")


@pytest.fixture
def containers() -> Iterator[list[str]]:
    names: list[str] = []
    yield names
    for name in names:
        _docker("rm", "-f", name, check=False)


@pytest.mark.timeout(240)
def test_directory_mount_sees_a_redeployed_hook_and_single_file_mount_does_not(
    hook_tree: Path, tmp_path: Path, containers: list[str]
) -> None:
    suffix = uuid.uuid4().hex[:8]
    old, new = f"omn19998-old-{suffix}", f"omn19998-new-{suffix}"
    containers.extend([old, new])
    _start(
        old,
        f"{hook_tree / 'runner-job-started.sh'}:/usr/local/bin/runner-job-started.sh:ro",
    )
    _start(new, f"{hook_tree}:/opt/onex-runner-hooks:ro")

    assert (
        "hook-v1"
        in _docker("exec", old, "cat", "/usr/local/bin/runner-job-started.sh").stdout
    )
    assert (
        "hook-v1"
        in _docker(
            "exec", new, "cat", "/opt/onex-runner-hooks/runner-job-started.sh"
        ).stdout
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "runner-job-started.sh").write_text(_V2, encoding="utf-8")
    _rsync_redeploy(staging, hook_tree)
    assert (hook_tree / "runner-job-started.sh").read_text(encoding="utf-8") == _V2

    seen_new = _docker(
        "exec", new, "cat", "/opt/onex-runner-hooks/runner-job-started.sh"
    ).stdout
    assert "hook-v2" in seen_new, "directory mount must see the redeployed hook"

    # Negative control: the old layout is stranded on the pre-deploy inode.
    seen_old = _docker(
        "exec", old, "sh", "-c", "cat /usr/local/bin/runner-job-started.sh"
    ).stdout
    assert "hook-v1" in seen_old, (
        "negative control failed: the single-file mount followed the redeploy, "
        "so this host does not reproduce the stranded-inode layout"
    )


@pytest.mark.timeout(240)
def test_healthcheck_in_the_container_fails_when_the_hook_is_removed(
    hook_tree: Path, containers: list[str]
) -> None:
    name = f"omn19998-hc-{uuid.uuid4().hex[:8]}"
    containers.append(name)
    _start(name, f"{hook_tree}:/opt/onex-runner-hooks:ro")
    hook_env = (
        "ACTIONS_RUNNER_HOOK_JOB_STARTED=/opt/onex-runner-hooks/runner-job-started.sh"
    )

    present = _docker(
        "exec",
        "-e",
        hook_env,
        "-e",
        "RUNNER_HOME=/nonexistent",
        name,
        "bash",
        "/opt/onex-runner-hooks/healthcheck.sh",
        check=False,
    )
    # Positive control: with the hook present layer 0 is passed and the next
    # layer (no listener in this container) is what fails.
    assert present.returncode == 1
    assert "Runner.Listener not running" in present.stdout

    (hook_tree / "runner-job-started.sh").unlink()
    gone = _docker(
        "exec",
        "-e",
        hook_env,
        "-e",
        "RUNNER_HOME=/nonexistent",
        "-e",
        "RUNNER_HEALTH_HOOK_STAND_DOWN=0",
        name,
        "bash",
        "/opt/onex-runner-hooks/healthcheck.sh",
        check=False,
    )
    assert gone.returncode == 1
    assert "runner-job-started.sh does not exist" in gone.stdout
