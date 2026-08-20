# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-execution coverage for scripts/runtime_build/compose_wait_timeout.sh
(OMN-15718).

Real defect this closes: refresh_stability_lane.sh --ref origin/dev --execute
(2026-08-05, .201) hit a forward-migration failure (OMN-15717), then its own
failure-path rollback retagged images correctly but the subsequent
``docker compose up -d --no-deps --force-recreate`` for the core services left
runtime-effects/runtime-worker stranded in ``Created`` state -- compose still
honors ``depends_on: migration-gate: condition: service_healthy`` for those
services even under ``--no-deps``, and migration-gate could never become
healthy once forward-migration had already failed. That ``up`` call had no
bounded deadline, so it hung indefinitely instead of failing fast, and never
reached the health-gate/receipt stage.

These tests exercise the two shared helpers this fix introduces --
``compose_up_bounded`` and ``reconcile_container_running_state`` -- as real
bash subprocesses (not string assertions on the script source), against a
fake ``docker``/``docker compose`` stand-in so no live Docker daemon is
required.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "compose_wait_timeout.sh"

_FAKE_DOCKER = """#!/usr/bin/env bash
# Fake `docker` for compose_wait_timeout.sh tests. State is a flat file:
# one "<name> <status>" line per container. Understands just enough of
# `docker inspect [--format FMT] NAME`, `docker start NAME`, and
# `docker rm [-f] NAME` for reconcile_container_running_state().
set -euo pipefail
STATE_FILE="${FAKE_DOCKER_STATE_FILE:?FAKE_DOCKER_STATE_FILE must be set}"

_status_of() {
    grep -E "^$1 " "${STATE_FILE}" 2>/dev/null | awk '{print $2}'
}

cmd="$1"; shift
case "${cmd}" in
    inspect)
        fmt=""
        name=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --format) fmt="$2"; shift 2 ;;
                *) name="$1"; shift ;;
            esac
        done
        status="$(_status_of "${name}")"
        if [[ -z "${status}" ]]; then
            exit 1
        fi
        if [[ -n "${fmt}" ]]; then
            printf '%s\\n' "${status}"
        fi
        exit 0
        ;;
    start)
        name="$1"
        status="$(_status_of "${name}")"
        if [[ "${status}" == "stuck" ]]; then
            # Simulates a container `docker start` cannot actually bring up
            # (e.g. it immediately re-enters Created because its own
            # depends_on condition is still unmet) -- exits 0 (docker start
            # itself does not error) but status stays non-running.
            exit 0
        fi
        sed -i.bak -E "s/^${name} .*/${name} running/" "${STATE_FILE}"
        exit 0
        ;;
    rm)
        name="${*: -1}"
        sed -i.bak -E "/^${name} /d" "${STATE_FILE}"
        exit 0
        ;;
    *)
        echo "fake docker: unsupported subcommand '${cmd}'" >&2
        exit 64
        ;;
esac
"""


@pytest.fixture
def fake_docker_env(tmp_path: Path) -> dict[str, str]:
    """PATH-prepend a fake `docker` and a state file it reads/writes."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_path = bin_dir / "docker"
    docker_path.write_text(_FAKE_DOCKER, encoding="utf-8")
    docker_path.chmod(docker_path.stat().st_mode | stat.S_IEXEC)

    state_file = tmp_path / "docker_state.txt"
    state_file.write_text("", encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_DOCKER_STATE_FILE"] = str(state_file)
    env["_STATE_FILE_PATH"] = str(state_file)  # convenience for tests
    return env


def _run_bash(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )


def _set_container_status(state_file: Path, name: str, status: str) -> None:
    lines = [
        line
        for line in state_file.read_text(encoding="utf-8").splitlines()
        if not line.startswith(f"{name} ")
    ]
    lines.append(f"{name} {status}")
    state_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# compose_up_bounded
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compose_up_bounded_kills_a_hung_command_with_exit_124(
    fake_docker_env: dict[str, str],
) -> None:
    """A command still running at the deadline is killed and returns 124.

    This is the exact regression: the rollback recreate hung indefinitely
    (no deadline at all) instead of failing fast. `sleep 30` stands in for a
    `docker compose up` stuck polling a dependency that will never become
    healthy.
    """
    start = time.monotonic()
    result = _run_bash(
        f'source "{LIB_SCRIPT}"; compose_up_bounded 1 sleep 30',
        fake_docker_env,
    )
    elapsed = time.monotonic() - start

    assert result.returncode == 124, result.stderr
    assert elapsed < 20, (
        f"compose_up_bounded did not bound the call -- took {elapsed:.1f}s "
        "against a 1s timeout (allow slack for --kill-after=15)"
    )
    assert "COMPOSE_UP_TIMEOUT" in result.stderr


@pytest.mark.unit
def test_compose_up_bounded_propagates_success_exit_code(
    fake_docker_env: dict[str, str],
) -> None:
    result = _run_bash(
        f'source "{LIB_SCRIPT}"; compose_up_bounded 5 true',
        fake_docker_env,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_compose_up_bounded_propagates_failure_exit_code_verbatim(
    fake_docker_env: dict[str, str],
) -> None:
    """A real (non-timeout) failure must NOT be remapped onto 124 or any
    other reused code -- callers need to tell a timeout apart from a genuine
    compose failure."""
    result = _run_bash(
        f'source "{LIB_SCRIPT}"; compose_up_bounded 5 bash -c "exit 3"',
        fake_docker_env,
    )
    assert result.returncode == 3, result.stderr


@pytest.mark.unit
def test_compose_up_bounded_fails_closed_when_timeout_binary_missing(
    tmp_path: Path,
) -> None:
    """Refuses to run the command unbounded if `timeout` is unavailable,
    rather than silently reintroducing the hang risk."""
    bash_bin = shutil.which("bash")
    assert bash_bin, "bash must be resolvable to run this test at all"

    empty_bin = tmp_path / "empty_bin"
    empty_bin.mkdir()
    env = {"PATH": str(empty_bin)}
    result = subprocess.run(
        [bash_bin, "-c", f'source "{LIB_SCRIPT}"; compose_up_bounded 5 true'],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert result.returncode == 64, result.stderr
    assert "timeout" in result.stderr.lower()


# ---------------------------------------------------------------------------
# reconcile_container_running_state
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reconcile_leaves_an_already_running_container_untouched(
    fake_docker_env: dict[str, str],
) -> None:
    state_file = Path(fake_docker_env["_STATE_FILE_PATH"])
    _set_container_status(state_file, "svc-a", "running")

    result = _run_bash(
        f'source "{LIB_SCRIPT}"; reconcile_container_running_state svc-a',
        fake_docker_env,
    )

    assert result.returncode == 0, result.stderr
    assert "svc-a running" in state_file.read_text(encoding="utf-8")


@pytest.mark.unit
def test_reconcile_recovers_a_created_container_via_docker_start(
    fake_docker_env: dict[str, str],
) -> None:
    """The stranded-in-Created case from the forensic log: `docker start`
    successfully recovers it to running."""
    state_file = Path(fake_docker_env["_STATE_FILE_PATH"])
    _set_container_status(state_file, "runtime-effects", "created")

    result = _run_bash(
        f'source "{LIB_SCRIPT}"; reconcile_container_running_state runtime-effects',
        fake_docker_env,
    )

    assert result.returncode == 0, result.stderr
    assert "runtime-effects running" in state_file.read_text(encoding="utf-8")


@pytest.mark.unit
def test_reconcile_tears_down_a_container_start_cannot_recover(
    fake_docker_env: dict[str, str],
) -> None:
    """AC1 (OMN-15718): a container that cannot be brought to running is
    torn down (docker rm -f), never left stranded in an ambiguous state."""
    state_file = Path(fake_docker_env["_STATE_FILE_PATH"])
    _set_container_status(state_file, "runtime-worker", "stuck")

    result = _run_bash(
        f'source "{LIB_SCRIPT}"; reconcile_container_running_state runtime-worker',
        fake_docker_env,
    )

    assert result.returncode == 1, result.stderr
    assert "STRANDED_CONTAINER" in result.stderr
    remaining = state_file.read_text(encoding="utf-8")
    assert "runtime-worker" not in remaining, (
        f"stranded container must be torn down (removed from state), got: {remaining!r}"
    )


@pytest.mark.unit
def test_reconcile_is_a_noop_for_an_absent_container(
    fake_docker_env: dict[str, str],
) -> None:
    result = _run_bash(
        f'source "{LIB_SCRIPT}"; reconcile_container_running_state never-existed',
        fake_docker_env,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Defaults / include guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_timeout_is_positive_and_overridable(
    fake_docker_env: dict[str, str],
) -> None:
    result = _run_bash(
        f'source "{LIB_SCRIPT}"; echo "${{RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}}"',
        fake_docker_env,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 0

    env = dict(fake_docker_env)
    env["RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS"] = "42"
    result = _run_bash(
        f'source "{LIB_SCRIPT}"; echo "${{RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}}"',
        env,
    )
    assert result.stdout.strip() == "42"


@pytest.mark.unit
def test_sourcing_twice_is_safe(fake_docker_env: dict[str, str]) -> None:
    """Both deploy-runtime.sh and refresh_stability_lane.sh source this file;
    a test harness sourcing both must not double-define anything badly."""
    result = _run_bash(
        f'source "{LIB_SCRIPT}"; source "{LIB_SCRIPT}"; compose_up_bounded 5 true',
        fake_docker_env,
    )
    assert result.returncode == 0, result.stderr
