# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RT-6 must partition its in-scope list by restart policy [OMN-16729].

Defect, measured on the .201 dev lane 2026-09-08T13:44Z:
`scripts/runtime_build/refresh_dev_lane.sh` puts `redpanda-scram-user` and
`redpanda-sasl-enable` in `REFRESH_BUILD_SERVICES` deliberately (OMN-18012
phase B: a governed refresh must RE-ASSERT the SASL flip). Both are
`restart: "no"` ONE-SHOTS. `readback_deployed_ref()` iterated the same in-scope
list demanding a RUNNING container for every member, so it failed with

    Deploy readback FAILED (RT-6): could not resolve a running container for
    in-scope service redpanda-scram-user.

for a container that had done its job perfectly (`state=exited exit=0
finished=2026-09-08T13:38:13Z`). Every governed dev refresh failed from that
point on, and the ensuing rollback restored `:latest` TAGS while the recreated
containers kept running the new image -- a tag rollback cannot recall a
recreated container, and a later `--no-build` recreate silently re-adopts the
rolled-back tags.

The fix reads `HostConfig.RestartPolicy.Name` off the container -- the compose
model as docker materialised it, not a hardcoded service-name list that goes
stale -- and asserts a one-shot as `exited` / `exit 0` / finished AFTER the
deploy started, while long-running services keep today's running-container
assertion unchanged.

Same seam-level harness as `test_deploy_runtime_rt6_scoped_readback.py`: the
real `readback_deployed_ref()` and its real dependencies are extracted from
`scripts/deploy-runtime.sh` and executed under bash with only `docker` stubbed.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
# OMN-16729: the lane -> compose-file mapping moved out of deploy-runtime.sh into
# a shared lib, because refresh_dev_lane.sh's rollback recreate needed the
# identical derivation and its hand-spelled copy had lost the dev-lane overlay.
# The harness sources the lib rather than extracting those functions by regex.
COMPOSE_FILES_SH = REPO_ROOT / "scripts" / "runtime_build" / "compose_files.sh"

GIT_SHA = "abc123def456"
VERSION = "9.9.9"
DEPLOY_STARTED_AT = "2026-09-08T13:30:00Z"

ONE_SHOT = "redpanda-scram-user"
CORE = "runtime-effects"


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        _script_text(),
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract function {name}() from deploy-runtime.sh"
    )
    return match.group(0)


def _extract_array(name: str) -> str:
    match = re.search(
        rf"^readonly {re.escape(name)}=\(.*?\n\)",
        _script_text(),
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract array {name}=() from deploy-runtime.sh"
    )
    return match.group(0)


def _write_docker_stub(bin_dir: Path) -> None:
    """A `docker` that distinguishes `ps -q` (running) from `ps -aq` (all), and
    answers `inspect -f <go-template>` per template from files on disk."""
    stub = bin_dir / "docker"
    stub.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${DOCKER_STUB_DIR}/calls.log"

if [[ "$1" == "compose" ]]; then
    shift
    args=("$@")
    n=${#args[@]}
    all=false
    for ((i = 0; i < n; i++)); do
        case "${args[$i]}" in
            -aq|-a) all=true ;;
        esac
    done
    for ((i = 0; i < n; i++)); do
        if [[ "${args[$i]}" == "ps" ]]; then
            service="${args[$((n - 1))]}"
            if [[ "${all}" == true ]]; then
                map_file="${DOCKER_STUB_DIR}/ps_all/${service}"
            else
                map_file="${DOCKER_STUB_DIR}/ps/${service}"
            fi
            if [[ -f "${map_file}" ]]; then
                cat "${map_file}"
                exit 0
            fi
            exit 0
        fi
    done
    exit 1
fi

if [[ "$1" == "inspect" ]]; then
    shift
    fmt=""
    container=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f|--format) fmt="$2"; shift 2 ;;
            *) container="$1"; shift ;;
        esac
    done
    case "${fmt}" in
        *RestartPolicy*) key="restart" ;;
        *State.Status*)  key="state" ;;
        *ExitCode*)      key="exit_code" ;;
        *FinishedAt*)    key="finished_at" ;;
        *)               key="revision" ;;
    esac
    f="${DOCKER_STUB_DIR}/inspect/${container}.${key}"
    if [[ -f "${f}" ]]; then
        cat "${f}"
    fi
    exit 0
fi

if [[ "$1" == "exec" ]]; then
    container="$2"
    package="${*: -1}"
    ver_file="${DOCKER_STUB_DIR}/version/${container}"
    if [[ -f "${ver_file}" ]]; then
        printf 'Name: %s\nVersion: %s\n' "${package}" "$(cat "${ver_file}")"
        exit 0
    fi
    exit 1
fi

exit 1
""",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_readback(
    tmp_path: Path,
    *,
    services: list[str],
    running: dict[str, str],
    all_containers: dict[str, str],
    inspect: dict[str, dict[str, str]],
    deploy_started_at: str = DEPLOY_STARTED_AT,
) -> subprocess.CompletedProcess[str]:
    stub_dir = tmp_path / "stubs"
    for sub in ("ps", "ps_all", "inspect", "version"):
        (stub_dir / sub).mkdir(parents=True, exist_ok=True)
    (stub_dir / "calls.log").write_text("", encoding="utf-8")
    _write_docker_stub(stub_dir)

    for service, container in running.items():
        (stub_dir / "ps" / service).write_text(container + "\n", encoding="utf-8")
    for service, container in all_containers.items():
        (stub_dir / "ps_all" / service).write_text(container + "\n", encoding="utf-8")
    for container, fields in inspect.items():
        for key, value in fields.items():
            (stub_dir / "inspect" / f"{container}.{key}").write_text(
                value, encoding="utf-8"
            )

    services_literal = " ".join(f'"{s}"' for s in services)
    harness = "\n".join(
        [
            "set -euo pipefail",
            "log_step() { printf 'STEP: %s\\n' \"$*\" >&2; }",
            "log_info() { printf 'INFO: %s\\n' \"$*\" >&2; }",
            "log_warn() { printf 'WARN: %s\\n' \"$*\" >&2; }",
            "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
            "log_cmd() { printf 'CMD: %s\\n' \"$*\" >&2; }",
            f'DEPLOY_STARTED_AT="{deploy_started_at}"',
            f'source "{COMPOSE_FILES_SH}"',
            _extract_function("resolve_lane_runtime_container_name"),
            _extract_array("DEV_LANE_ONLY_RUNTIME_SERVICES"),
            _extract_array("STABILITY_TEST_LANE_ONLY_RUNTIME_SERVICES"),
            _extract_function("resolve_lane_runtime_services"),
            _extract_function("service_is_one_shot"),
            _extract_function("readback_one_shot_service"),
            _extract_function("readback_deployed_ref"),
            f'RUNTIME_BUILD_SERVICES_OVERRIDE="{" ".join(services)}"',
            f"RUNTIME_BUILD_SERVICES=({services_literal})",
            (
                f'readback_deployed_ref "{GIT_SHA}" "{VERSION}" '
                f'"omnibase-infra" "{REPO_ROOT}" "/tmp/fake-deploy-target"'
            ),
        ]
    )

    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
    env["DOCKER_STUB_DIR"] = str(stub_dir)
    return subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=False, env=env
    )


def _one_shot_inspect(
    *, state: str = "exited", exit_code: str = "0", finished_at: str
) -> dict[str, str]:
    return {
        "restart": "no",
        "state": state,
        "exit_code": exit_code,
        "finished_at": finished_at,
    }


@pytest.mark.unit
def test_one_shot_exited_zero_after_deploy_start_is_accepted(tmp_path: Path) -> None:
    """AC1: the live 2026-09-08 case. A `restart: "no"` one-shot that exited 0
    during this deploy passes RT-6 instead of failing it for not running."""
    result = _run_readback(
        tmp_path,
        services=[ONE_SHOT],
        running={},  # a completed one-shot is deliberately NOT running
        all_containers={ONE_SHOT: "oneshot1"},
        inspect={"oneshot1": _one_shot_inspect(finished_at="2026-09-08T13:38:13.5Z")},
    )
    assert result.returncode == 0, (
        "a one-shot that exited 0 during this deploy must pass RT-6. "
        f"stderr={result.stderr!r}"
    )
    assert "one-shot exited 0" in result.stderr


@pytest.mark.unit
def test_one_shot_nonzero_exit_is_rejected(tmp_path: Path) -> None:
    """AC2: the relaxation is bounded -- a one-shot that FAILED still fails the
    deploy. The SASL flip's whole value is that its failure is loud."""
    result = _run_readback(
        tmp_path,
        services=[ONE_SHOT],
        running={},
        all_containers={ONE_SHOT: "oneshot1"},
        inspect={
            "oneshot1": _one_shot_inspect(
                exit_code="1", finished_at="2026-09-08T13:38:13.5Z"
            )
        },
    )
    assert result.returncode != 0
    assert "exited 1, expected 0" in result.stderr
    assert "Refusing to certify" in result.stderr


@pytest.mark.unit
def test_one_shot_that_finished_before_this_deploy_is_rejected(tmp_path: Path) -> None:
    """AC3: a green exit code from an EARLIER deploy is not evidence that this
    run re-asserted anything. The finished-at check is the half that carries the
    meaning for a re-assertion one-shot."""
    result = _run_readback(
        tmp_path,
        services=[ONE_SHOT],
        running={},
        all_containers={ONE_SHOT: "oneshot1"},
        inspect={"oneshot1": _one_shot_inspect(finished_at="2026-09-05T01:02:03.5Z")},
    )
    assert result.returncode != 0
    assert "BEFORE this deploy started" in result.stderr


@pytest.mark.unit
def test_one_shot_still_running_is_rejected(tmp_path: Path) -> None:
    """AC4: a one-shot that never reached a terminal state has proven nothing."""
    result = _run_readback(
        tmp_path,
        services=[ONE_SHOT],
        running={ONE_SHOT: "oneshot1"},
        all_containers={ONE_SHOT: "oneshot1"},
        inspect={
            "oneshot1": _one_shot_inspect(
                state="running", finished_at="0001-01-01T00:00:00Z"
            )
        },
    )
    assert result.returncode != 0
    assert "expected 'exited'" in result.stderr


@pytest.mark.unit
def test_long_running_service_missing_its_container_is_still_rejected(
    tmp_path: Path,
) -> None:
    """AC5: the pre-existing assertion for a real service is unchanged -- an
    `unless-stopped` service that exited is a failed deploy, not a one-shot."""
    result = _run_readback(
        tmp_path,
        services=[CORE],
        running={},
        all_containers={CORE: "core1"},
        inspect={
            "core1": {
                "restart": "unless-stopped",
                "state": "exited",
                "exit_code": "0",
                "finished_at": "2026-09-08T13:38:13.5Z",
                "revision": GIT_SHA,
            }
        },
    )
    assert result.returncode != 0
    assert "could not resolve a running container" in result.stderr
    assert "not a one-shot" in result.stderr


@pytest.mark.unit
def test_service_with_no_container_at_all_is_rejected(tmp_path: Path) -> None:
    """AC6: absence of any container -- running or exited -- still fails closed."""
    result = _run_readback(
        tmp_path,
        services=[CORE],
        running={},
        all_containers={},
        inspect={},
    )
    assert result.returncode != 0
    assert "NO container at all" in result.stderr


@pytest.mark.unit
def test_mixed_scope_verifies_both_halves(tmp_path: Path) -> None:
    """AC7: the real refresh scope is mixed. Both partitions are asserted in one
    pass -- the one-shot on exit status, the service on image revision."""
    result = _run_readback(
        tmp_path,
        services=[CORE, ONE_SHOT],
        running={CORE: "core1"},
        all_containers={CORE: "core1", ONE_SHOT: "oneshot1"},
        inspect={
            "core1": {
                "restart": "unless-stopped",
                "state": "running",
                "revision": GIT_SHA,
            },
            "oneshot1": _one_shot_inspect(finished_at="2026-09-08T13:38:13.5Z"),
        },
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "one-shot exited 0" in result.stderr
    assert f"{CORE} (core1) revision == {GIT_SHA}" in result.stderr


@pytest.mark.unit
def test_the_dev_refresh_scope_still_contains_the_one_shots() -> None:
    """The two halves of OMN-18012 must not drift apart again: if the one-shots
    were quietly dropped from REFRESH_BUILD_SERVICES this fix would be masking a
    regression rather than closing one."""
    text = (REPO_ROOT / "scripts" / "runtime_build" / "refresh_dev_lane.sh").read_text(
        encoding="utf-8"
    )
    scope = re.search(
        r"^readonly REFRESH_BUILD_SERVICES=\(.*?\n\)", text, re.DOTALL | re.MULTILINE
    )
    assert scope is not None
    assert "redpanda-scram-user" in scope.group(0)
    assert "redpanda-sasl-enable" in scope.group(0)
