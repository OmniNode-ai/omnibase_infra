# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-execution coverage for deploy-runtime.sh's partition-cap warmup
phantom tolerance (OMN-16110).

Real defect this closes: refresh_dev_lane.sh --ref origin/dev --execute
(2026-08-24, .201 dev lane) died in deploy-runtime.sh's "Broker
Topic-Provisioning Warmup" step. A stale daemon-phantom container record for
``redpanda-partition-cap`` (listed ``Dead`` by ``docker ps -a``, but "No such
container" on both ``docker inspect`` and ``docker rm -f``, with no backing
directory under the daemon's containers dir) made compose's convergence plan
try to start the phantom AFTER it had already recreated and started the real
one-shot. That trailing start failed the whole ``up`` with "No such
container" even though the cap container was up and running -- and the
unguarded ``compose_up_bounded`` call aborted the entire deploy, which then
restored the previous deployment and the dev-lane refresh rolled back
(FAILED_ROLLED_BACK receipt 20260824T151518Z-08f01068359f). The phantom
record survives ``docker rm -f`` and reproduces deterministically until a
dockerd restart, so every subsequent refresh attempt fails the same way.

The fix mirrors the OMN-13364 doctrine already applied to the broker
``up --wait`` a few lines above in the same function: the compose ``up`` exit
code is NOT the source of truth for whether the cap was applied -- the named
one-shot's own run-to-completion (``docker wait`` == 0), which the step
already checks immediately afterward, is. The ``up`` becomes best-effort
(guarded); the ``docker wait`` decision stays fail-closed and keeps the
OMN-15718 bounded deadline.

These tests execute the REAL ``warm_broker_topic_provisioning`` function
extracted from scripts/deploy-runtime.sh as a bash subprocess (not string
assertions on the script source), with the real ``compose_up_bounded`` from
scripts/runtime_build/compose_wait_timeout.sh and a fake ``docker`` stand-in,
so no live Docker daemon is required.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_RUNTIME = REPO_ROOT / "scripts" / "deploy-runtime.sh"
LIB_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "compose_wait_timeout.sh"

_FAKE_DOCKER = """#!/usr/bin/env bash
# Fake `docker` for warm_broker_topic_provisioning tests. Behavior is driven
# by env vars so each test declares its scenario:
#   FAKE_CAP_UP_EXIT     exit code for `docker compose ... up ...
#                        redpanda-partition-cap` (default 0). Non-zero also
#                        prints the daemon phantom error observed live.
#   FAKE_CAP_WAIT_OUTPUT stdout of `docker wait <cap-container>` (default 0
#                        -- the one-shot's own exit code as `docker wait`
#                        reports it).
#   FAKE_CAP_WAIT_EXIT   exit code of `docker wait` itself (default 0).
set -euo pipefail
cmd="$1"; shift
case "${cmd}" in
    compose)
        # Identify the target service: it is the trailing argv element.
        target="${*: -1}"
        if [[ "${target}" == "redpanda-partition-cap" ]]; then
            if [[ "${FAKE_CAP_UP_EXIT:-0}" != "0" ]]; then
                # Shape observed live on .201 (OMN-16110): the real one-shot
                # is recreated and started, then compose trips over the
                # phantom record and fails the whole up.
                echo " Container omnibase-infra-redpanda-partition-cap Started " >&2
                echo "Error response from daemon: No such container: e5f584a8ae2450a15141304299f6c734768ed4b9a9481ce5dfeb9d35fe4b954e" >&2
                exit "${FAKE_CAP_UP_EXIT}"
            fi
            exit 0
        fi
        # Broker readiness up (service `redpanda`) and anything else: succeed.
        exit 0
        ;;
    wait)
        if [[ -n "${FAKE_CAP_WAIT_OUTPUT+x}" ]]; then
            printf '%s\\n' "${FAKE_CAP_WAIT_OUTPUT}"
        else
            printf '0\\n'
        fi
        exit "${FAKE_CAP_WAIT_EXIT:-0}"
        ;;
    *)
        echo "fake docker: unsupported subcommand '${cmd}'" >&2
        exit 64
        ;;
esac
"""


def _extract_function(name: str) -> str:
    """Extract one top-level bash function body from deploy-runtime.sh.

    deploy-runtime.sh executes ``main "$@"`` at load time, so it cannot be
    sourced whole; extract just the function under test instead.
    """
    source = DEPLOY_RUNTIME.read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n.*?^\}}$",
        source,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"function {name}() not found in {DEPLOY_RUNTIME}"
    return match.group(0)


_HARNESS = """
set -euo pipefail
source "${LIB_SCRIPT}"
COMPOSE_PROFILE="runtime"
BROKER_READINESS_SERVICE="redpanda"
BROKER_PARTITION_CAP_SERVICE="redpanda-partition-cap"
RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS=10
log_step()  { echo "[step] $*"; }
log_info()  { echo "[info] $*"; }
log_warn()  { echo "[warn] $*"; }
log_error() { echo "[error] $*"; }
log_cmd()   { echo "[cmd] $*"; }
resolve_compose_file_args() {
    local _out_args_name="$1"
    # A single harmless compose global flag keeps the expansion non-empty.
    eval "${_out_args_name}=(--ansi never)"
}
assert_broker_reachable() { return 0; }
source "${FUNC_FILE}"
# Deliberately UNGUARDED, exactly like the live call site (deploy-runtime.sh
# main -> warm_broker_topic_provisioning under `set -euo pipefail`): guarding
# it with `|| rc=$?` here would suspend errexit inside the function body and
# mask the very unguarded-compose_up_bounded abort this fix removes.
warm_broker_topic_provisioning "/nonexistent-deploy-target" "omnibase-infra"
echo "HARNESS_RC=0"
"""


@pytest.fixture
def harness_env(tmp_path: Path) -> dict[str, str]:
    """PATH-prepend a fake `docker`, and stage the extracted function."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_path = bin_dir / "docker"
    docker_path.write_text(_FAKE_DOCKER, encoding="utf-8")
    docker_path.chmod(docker_path.stat().st_mode | stat.S_IEXEC)

    func_file = tmp_path / "warm_broker_topic_provisioning.sh"
    func_file.write_text(
        _extract_function("warm_broker_topic_provisioning"), encoding="utf-8"
    )

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["LIB_SCRIPT"] = str(LIB_SCRIPT)
    env["FUNC_FILE"] = str(func_file)
    return env


def _run_harness(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", _HARNESS],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


@pytest.mark.unit
def test_phantom_up_failure_is_tolerated_when_cap_ran_to_completion(
    harness_env: dict[str, str],
) -> None:
    """The exact live failure: cap `up` exits 1 on the phantom start, but the
    real one-shot ran to completion (`docker wait` == 0). The step must
    succeed instead of aborting the whole deploy."""
    harness_env["FAKE_CAP_UP_EXIT"] = "1"
    harness_env["FAKE_CAP_WAIT_OUTPUT"] = "0"
    result = _run_harness(harness_env)
    assert result.returncode == 0 and "HARNESS_RC=0" in result.stdout, (
        f"expected phantom-up failure to be tolerated, got rc={result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Broker partition cap applied." in result.stdout
    assert "Partition-cap compose up exited non-zero" in result.stdout


@pytest.mark.unit
def test_up_failure_with_cap_not_exiting_zero_still_aborts(
    harness_env: dict[str, str],
) -> None:
    """Fail-closed: if the one-shot did NOT run to completion with exit 0,
    the guarded `up` must not mask that -- the step still aborts."""
    harness_env["FAKE_CAP_UP_EXIT"] = "1"
    harness_env["FAKE_CAP_WAIT_OUTPUT"] = "1"
    result = _run_harness(harness_env)
    assert result.returncode != 0 and "HARNESS_RC=0" not in result.stdout, (
        f"expected abort when cap one-shot exited non-zero, got rc={result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "did not complete successfully" in result.stdout


@pytest.mark.unit
def test_docker_wait_error_fails_closed(harness_env: dict[str, str]) -> None:
    """Fail-closed: if `docker wait` itself errors with no exit-code output
    (e.g. the up truly created nothing), the step still aborts."""
    harness_env["FAKE_CAP_UP_EXIT"] = "1"
    harness_env["FAKE_CAP_WAIT_OUTPUT"] = ""
    harness_env["FAKE_CAP_WAIT_EXIT"] = "1"
    result = _run_harness(harness_env)
    assert result.returncode != 0 and "HARNESS_RC=0" not in result.stdout, (
        f"expected abort when docker wait produced no exit code, got rc={result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "did not complete successfully" in result.stdout


@pytest.mark.unit
def test_clean_path_unchanged(harness_env: dict[str, str]) -> None:
    """Regression guard: the healthy path (up 0, wait 0) still succeeds."""
    result = _run_harness(harness_env)
    assert result.returncode == 0 and "HARNESS_RC=0" in result.stdout, (
        f"expected clean path to succeed, got rc={result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Broker partition cap applied." in result.stdout
