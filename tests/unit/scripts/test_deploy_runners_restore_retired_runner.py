# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19397: ``--rolling`` can bring back a RETIRED general-pool runner.

A retired runner (what OMN-19077's ``--retire-surplus`` leaves behind on .201
for omninode-runner-41..60) has no container, no GitHub registration, and a
kept creds volume whose cached credentials belong to the deleted
registration. Before this fix the sanctioned tooling had no way back:

- ``runner_is_idle()`` read an ABSENT container as "unknown" (``docker
  inspect`` printed nothing, GitHub had no such runner), so ``--rolling``
  skipped it as busy on every retry pass and never reached it.
- Even if it had, the OMN-19274 cache pre-flight reads the kept volume as
  "ready", so the recreate would get an EMPTY registration token, restore
  credentials for a registration GitHub deleted, and crash-loop.

The fix: an absent container runs no job, so it is idle by construction
(like a stopped one); and creating a container that does not exist always
needs ``--token-file``, whatever the volume holds. Without one it is skipped
(rc 3), never recreated into a crash.

The container-state probe runs for real against the local docker daemon
through a fake ``ssh`` that executes the "remote" command locally.
``roll_one_runner`` runs for real with its collaborators stubbed as shell
functions and a recording ``ssh``.
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
pytestmark = pytest.mark.unit
needs_docker = pytest.mark.skipif(REAL_DOCKER is None, reason="docker is not on PATH")


def _extract_function(script_text: str, name: str) -> str:
    start_marker = f"\n{name}() {{"
    assert start_marker in script_text, f"{name}() is not defined in deploy-runners.sh"
    start = script_text.index(start_marker) + 1
    depth = 0
    end = None
    for i in range(start + len(start_marker) - 2, len(script_text)):
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
def script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture
def local_ssh(tmp_path: Path) -> Path:
    bindir = tmp_path / "localssh"
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
def broken_ssh(tmp_path: Path) -> Path:
    bindir = tmp_path / "brokenssh"
    bindir.mkdir()
    ssh_stub = bindir / "ssh"
    ssh_stub.write_text("#!/usr/bin/env bash\nexit 255\n")
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC)
    return bindir


@pytest.fixture
def broken_docker(tmp_path: Path) -> Path:
    """Local ssh plus a docker whose daemon is unreachable."""
    bindir = tmp_path / "brokendocker"
    bindir.mkdir()
    ssh_stub = bindir / "ssh"
    ssh_stub.write_text('#!/usr/bin/env bash\nshift\nexec bash -c "$1"\n')
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC)
    docker_stub = bindir / "docker"
    docker_stub.write_text(
        '#!/usr/bin/env bash\necho "Cannot connect to the Docker daemon" >&2\nexit 1\n'
    )
    docker_stub.chmod(docker_stub.stat().st_mode | stat.S_IEXEC)
    return bindir


@pytest.fixture
def stopped_container() -> Iterator[str]:
    name = f"omn19397-stopped-{uuid.uuid4().hex[:12]}"
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "alpine:3", "sleep", "3600"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["docker", "stop", "-t", "1", name], check=True, capture_output=True)
    yield name
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def _bash(
    script: str, path_prefix: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}:{env['PATH']}"
    env["RUNNER_HOST"] = "fake-host.invalid"
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )


def _absent_name() -> str:
    return f"omn19397-absent-{uuid.uuid4().hex[:12]}"


# --- runner_container_state: running | stopped | absent | unknown ----------


@needs_docker
def test_an_absent_container_reads_absent(script_text: str, local_ssh: Path) -> None:
    fn = _extract_function(script_text, "runner_container_state")
    result = _bash(f'{fn}\nrunner_container_state "{_absent_name()}"', local_ssh)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "absent"


@needs_docker
def test_a_stopped_container_reads_stopped(
    script_text: str, local_ssh: Path, stopped_container: str
) -> None:
    fn = _extract_function(script_text, "runner_container_state")
    result = _bash(f'{fn}\nrunner_container_state "{stopped_container}"', local_ssh)
    assert result.stdout.strip() == "stopped", (result.stdout, result.stderr)


@needs_docker
def test_a_name_prefix_does_not_match_a_longer_container(
    script_text: str, local_ssh: Path, stopped_container: str
) -> None:
    """``omninode-runner-4`` must not read the state of ``omninode-runner-41``:
    docker's name filter is a substring regex unless anchored."""
    fn = _extract_function(script_text, "runner_container_state")
    prefix = stopped_container[:-3]
    result = _bash(f'{fn}\nrunner_container_state "{prefix}"', local_ssh)
    assert result.stdout.strip() == "absent", (result.stdout, result.stderr)


def test_an_unreachable_host_reads_unknown_not_absent(
    script_text: str, broken_ssh: Path
) -> None:
    fn = _extract_function(script_text, "runner_container_state")
    result = _bash(f'{fn}\nrunner_container_state "omninode-runner-41"', broken_ssh)
    assert result.stdout.strip() == "unknown", (result.stdout, result.stderr)


def test_an_unreachable_docker_daemon_reads_unknown_not_absent(
    script_text: str, broken_docker: Path
) -> None:
    fn = _extract_function(script_text, "runner_container_state")
    result = _bash(f'{fn}\nrunner_container_state "omninode-runner-41"', broken_docker)
    assert result.stdout.strip() == "unknown", (result.stdout, result.stderr)


# --- runner_is_idle ---------------------------------------------------------

_POISONED_GITHUB = 'github_runner_state() { echo "GITHUB CHECK CALLED" >&2; echo "unknown unknown"; }\n'


@needs_docker
def test_an_absent_container_is_idle_without_asking_github(
    script_text: str, local_ssh: Path
) -> None:
    """The live gap on .201: omninode-runner-41..60 had no container and no
    registration, so the busy check read them unknown on every pass."""
    fns = _extract_function(script_text, "runner_container_state") + "\n"
    fns += _extract_function(script_text, "runner_is_idle")
    result = _bash(
        f'set -euo pipefail\n{_POISONED_GITHUB}{fns}\nrunner_is_idle "{_absent_name()}"',
        local_ssh,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "GITHUB CHECK CALLED" not in result.stderr


def test_an_unreachable_host_is_never_idle(script_text: str, broken_ssh: Path) -> None:
    """Negative control: failing to read the host must not pass for absent."""
    fns = _extract_function(script_text, "runner_container_state") + "\n"
    fns += _extract_function(script_text, "runner_is_idle")
    result = _bash(
        f'{_POISONED_GITHUB}{fns}\nrunner_is_idle "omninode-runner-41"', broken_ssh
    )
    assert result.returncode != 0, (result.stdout, result.stderr)


# --- roll_one_runner ---------------------------------------------------------


def _roll_harness(
    script_text: str, tmp_path: Path, container_state: str, token_file: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    record = tmp_path / "ssh.log"
    bindir = tmp_path / "recssh"
    bindir.mkdir()
    ssh_stub = bindir / "ssh"
    ssh_stub.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n---\\n" "$2" >> "{record}"\nexit 0\n'
    )
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC)
    stubs = "\n".join(
        [
            'log() { echo "[log] $*"; }',
            'warn() { echo "[warn] $*" >&2; }',
            'err() { echo "[err] $*" >&2; exit 1; }',
            'encode_token() { printf "%s" "$1" | base64; }',
            "runner_is_idle() { return 0; }",
            f'runner_container_state() {{ echo "{container_state}"; }}',
            'github_runner_state() { echo "unknown unknown"; }',
            "_runner_cache_key_for_service() { echo deadbeefcafe0000; }",
            # The kept volume of a retired runner DOES hold a directory for
            # the current key -- credentials for a deleted registration.
            "_runner_cache_ready() { return 0; }",
            "wait_for_runner_online() { return 0; }",
        ]
    )
    fn = _extract_function(script_text, "roll_one_runner")
    script = (
        "set -euo pipefail\n"
        'RUNNER_HOST_DIR="/fake/runners"\n'
        "DRY_RUN=false\n"
        f'TOKEN_FILE="{token_file}"\n'
        f"{stubs}\n{fn}\n"
        'rc=0; roll_one_runner "omninode-runner-41" || rc=$?; echo "rc=${rc}"\n'
    )
    result = _bash(script, bindir)
    return result, record.read_text() if record.exists() else ""


def test_a_retired_runner_without_a_token_file_is_skipped_not_recreated(
    script_text: str, tmp_path: Path
) -> None:
    result, ssh_calls = _roll_harness(script_text, tmp_path, "absent", "")
    assert "rc=3" in result.stdout, (result.stdout, result.stderr)
    assert "up -d" not in ssh_calls, (
        "an absent runner recreated with an empty token restores credentials "
        "for a deleted registration and crash-loops"
    )


def test_a_retired_runner_with_a_token_file_is_created_with_the_token(
    script_text: str, tmp_path: Path
) -> None:
    token = tmp_path / "token"
    token.write_text("AAAFAKETOKEN")
    token.chmod(0o600)
    result, ssh_calls = _roll_harness(script_text, tmp_path, "absent", str(token))
    assert "rc=0" in result.stdout, (result.stdout, result.stderr)
    assert "up -d --force-recreate --no-deps --no-build omninode-runner-41" in ssh_calls
    assert "RUNNER_TOKEN=$(echo" in ssh_calls
    assert "AAAFAKETOKEN" not in ssh_calls, "the token must cross ssh base64-encoded"
    assert "AAAFAKETOKEN" not in result.stdout + result.stderr


def test_a_present_runner_with_a_ready_cache_still_rolls_without_a_token(
    script_text: str, tmp_path: Path
) -> None:
    """Positive control: the steady-state roll is unchanged."""
    result, ssh_calls = _roll_harness(script_text, tmp_path, "running", "")
    assert "rc=0" in result.stdout, (result.stdout, result.stderr)
    assert "export RUNNER_TOKEN=''" in ssh_calls
    assert "up -d --force-recreate --no-deps --no-build omninode-runner-41" in ssh_calls
