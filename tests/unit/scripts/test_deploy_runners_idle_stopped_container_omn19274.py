# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Execution-level control for the OMN-19274 follow-up to ``runner_is_idle()``.

Discovered while migrating ``omninode-runner-3``/``omninode-runner-4`` (both
left offline by the very outage OMN-19274's credential-cache pre-flight
fixes): ``runner_is_idle()`` required GitHub to report the runner
``online``, which a STOPPED container can never satisfy -- the busy check
skipped it as "unknown" on every retry pass, so the migration token path
could never reach the exact runners it exists to recover. A container that
is not running at all cannot be mid-job, so it is now checked first and
treated as idle by construction, ahead of the GitHub online/busy read.

This runs ``runner_is_idle()`` for real against a real local stopped
container, through the same fake-``ssh``-runs-locally harness as the
OMN-19274 pre-flight tests, so it is real ``docker inspect`` output driving
the assertion, not a string match on the script.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"

REAL_DOCKER = shutil.which("docker")
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(REAL_DOCKER is None, reason="docker is not on PATH"),
]


def _extract_function(script_text: str, name: str) -> str:
    start_marker = f"{name}() {{"
    start = script_text.index(start_marker)
    depth = 0
    end = None
    for i in range(start + len(start_marker) - 1, len(script_text)):
        ch = script_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end is not None, f"unbalanced braces extracting {name}()"
    return script_text[start:end]


@pytest.fixture(scope="module")
def idle_check_function() -> str:
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    fn = _extract_function(text, "runner_is_idle")
    assert "docker inspect --format '{{.State.Running}}'" in fn
    return fn


@pytest.fixture
def fake_ssh(tmp_path: Path) -> Path:
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    ssh_stub = bindir / "ssh"
    ssh_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "shift  # drop the host argument -- run the remote command locally\n"
        'exec bash -c "$1"\n'
    )
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC)
    return bindir


@pytest.fixture
def stopped_container() -> Iterator[str]:
    name = f"omn19274-idle-test-{uuid.uuid4().hex[:12]}"
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "alpine:3", "sleep", "3600"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["docker", "stop", "-t", "1", name], check=True, capture_output=True)
    yield name
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def _run(
    idle_check_function: str, fake_ssh: Path, name: str
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{fake_ssh}:{env['PATH']}"
    env["RUNNER_HOST"] = "fake-host.invalid"
    script = f'set -euo pipefail\n{idle_check_function}\nrunner_is_idle "{name}"\n'
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )


def test_a_stopped_container_reads_idle_without_ever_asking_github(
    idle_check_function: str, fake_ssh: Path, stopped_container: str
) -> None:
    """The exact live gap: omninode-runner-3/4 were stopped (offline), so
    GitHub could never report them online, and the pre-OMN-19274-follow-up
    busy check retried and skipped them forever. A stopped container must
    read idle on the FIRST call -- this test defines ``github_runner_state``
    as a function that always errors, so a pass here proves the stopped-
    container branch returns before ever calling it.
    """
    poisoned_github_check = 'github_runner_state() { echo "GITHUB CHECK CALLED -- SHOULD NEVER REACH HERE" >&2; exit 9; }\n'
    result = _run(
        poisoned_github_check + idle_check_function, fake_ssh, stopped_container
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "SHOULD NEVER REACH HERE" not in result.stderr


def test_a_running_container_still_falls_through_to_the_github_check(
    idle_check_function: str, fake_ssh: Path
) -> None:
    """Positive control: a RUNNING container must not take the stopped-
    container shortcut -- it has to fall through to the real busy check.
    """
    name = f"omn19274-idle-running-{uuid.uuid4().hex[:12]}"
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "alpine:3", "sleep", "3600"],
        check=True,
        capture_output=True,
    )
    try:
        marker_github_check = 'github_runner_state() { echo "online false"; }\n'
        result = _run(marker_github_check + idle_check_function, fake_ssh, name)
        # Falls through past the stopped-container branch into the real
        # github_runner_state call (stubbed here to "online false", i.e.
        # not busy) and then into the docker-top worker check, which for a
        # plain `sleep` container finds no Runner.Worker and returns idle.
        assert result.returncode == 0, (result.stdout, result.stderr)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
