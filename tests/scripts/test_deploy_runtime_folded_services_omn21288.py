# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Folded services are excluded and retired before the lane restarts."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/omnibase_infra/handlers/handler_runtime_deploy.sh"
POLICY_ENV = ROOT / "docker/runtime-policy.env"
pytestmark = pytest.mark.unit


def _policy(key: str) -> str:
    for line in POLICY_ENV.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    raise AssertionError(key)


# Lane projects come from the committed runtime policy, never a literal here.
FOLDING_PROJECT = _policy("STABILITY_TEST_COMPOSE_PROJECT")
DEV_PROJECT = _policy("DEV_COMPOSE_PROJECT")
DOGFOOD_PROJECT = _policy("DOGFOOD_COMPOSE_PROJECT")


def _function(name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{.*?^\}}", SCRIPT.read_text(), re.M | re.S)
    assert match
    return match.group(0)


def _harness(
    profiles: str, project: str, override: bool = False, *, dev_profiles: str = ""
) -> str:
    text = SCRIPT.read_text()
    declarations = []
    for name in (
        "RUNTIME_SERVICES",
        "DEV_LANE_ONLY_RUNTIME_SERVICES",
        "DEV_LANE_EXTRA_BROKER_CLIENTS",
        "STABILITY_TEST_LANE_ONLY_RUNTIME_SERVICES",
    ):
        match = re.search(rf"^readonly {name}=\(.*?^\)", text, re.M | re.S)
        assert match
        declarations.append(match.group(0))
    return "\n".join(
        [
            "set -euo pipefail",
            f"source {shlex.quote(str(ROOT / 'scripts/runtime_build/compose_files.sh'))}",
            'log_error() { printf "%s\\n" "$*" >&2; }',
            *declarations,
            _function("resolve_lane_folded_services"),
            _function("resolve_lane_runtime_services"),
            f"DEV_COMPOSE_PROJECT={shlex.quote(DEV_PROJECT)}",
            f"DEV_RUNTIME_MAIN_FOLDED_PROFILES={shlex.quote(dev_profiles)}",
            f"STABILITY_TEST_COMPOSE_PROJECT={shlex.quote(FOLDING_PROJECT)}",
            f"STABILITY_TEST_RUNTIME_MAIN_FOLDED_PROFILES={shlex.quote(profiles)}",
            f"PROJECT={shlex.quote(project)}",
            "RUNTIME_BUILD_SERVICES_OVERRIDE=runtime-effects"
            if override
            else "unset RUNTIME_BUILD_SERVICES_OVERRIDE",
            "RUNTIME_BUILD_SERVICES=(runtime-effects)"
            if override
            else 'RUNTIME_BUILD_SERVICES=("${RUNTIME_SERVICES[@]}")',
        ]
    )


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("profiles", ["effects", " , effects , ,"])
def test_folded_effects_are_excluded(profiles: str, override: bool) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            _harness(profiles, FOLDING_PROJECT, override)
            + '\nresolve_lane_runtime_services services "$PROJECT"\ndeclare -p services',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "runtime-effects" not in result.stdout


@pytest.mark.parametrize(
    "project",
    [FOLDING_PROJECT, DEV_PROJECT, DOGFOOD_PROJECT],
)
def test_unfolded_lanes_keep_effects(project: str) -> None:
    profiles = "" if project == FOLDING_PROJECT else "effects"
    result = subprocess.run(
        [
            "bash",
            "-c",
            _harness(profiles, project)
            + '\nresolve_lane_runtime_services services "$PROJECT"\ndeclare -p services',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "runtime-effects" in result.stdout


def test_unknown_profile_fails() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            _harness("unknown-role", FOLDING_PROJECT)
            + '\nresolve_lane_runtime_services services "$PROJECT"',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Unknown folded runtime profile: unknown-role" in result.stderr


@pytest.mark.parametrize("override", [False, True])
def test_folded_dev_tenant_carrier_is_excluded_and_retired(override: bool) -> None:
    harness = (
        _harness("effects", DEV_PROJECT, override, dev_profiles="tenant-projection")
        + "\n"
        + _function("restart_services")
        + r"""
if [[ -n "${RUNTIME_BUILD_SERVICES_OVERRIDE:-}" ]]; then
    RUNTIME_BUILD_SERVICES=(tenant-projection-writer)
    RUNTIME_BUILD_SERVICES_OVERRIDE=tenant-projection-writer
fi
log_step() { :; }
log_info() { :; }
log_cmd() { :; }
resolve_compose_file_args() { local -n out="$1"; out=(); }
exclude_cloud_database_services() { :; }
COMPOSE_PROFILE=runtime
RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS=60
docker() {
    case "$1" in
        ps)
            [[ "$*" == *"label=com.docker.compose.project=$PROJECT"* ]] || return 9
            [[ "$*" == *"service=tenant-projection-writer"* ]] || return 9
            printf 'tenant-a\ntenant-b\n' ;;
        stop | rm) printf '%s\n' "$*" ;;
        *) return 9 ;;
    esac
}
compose_up_bounded() { shift; printf 'UP %s\n' "$*"; }
restart_services /tmp "$PROJECT"
"""
    )
    result = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=True
    )
    lines = result.stdout.splitlines()
    if override:
        assert not any(line.startswith(("UP ", "stop ", "rm ")) for line in lines)
        return
    up = next(i for i, line in enumerate(lines) if line.startswith("UP "))
    assert (
        lines.index("stop --time 30 tenant-a tenant-b")
        < lines.index("rm tenant-a tenant-b")
        < up
    )
    assert "tenant-projection-writer" not in lines[up]
    assert "runtime-effects" in lines[up]
    assert " -v" not in result.stdout


@pytest.mark.parametrize("profiles", ["effects", "effects,workers"])
@pytest.mark.parametrize("override", [False, True])
def test_retirement_precedes_recreate(profiles: str, override: bool) -> None:
    harness = (
        _harness(profiles, FOLDING_PROJECT, override)
        + "\n"
        + _function("restart_services")
        + r"""
log_step() { :; }
log_info() { printf 'INFO %s\n' "$*"; }
log_cmd() { :; }
resolve_compose_file_args() { local -n out="$1"; out=(); }
exclude_cloud_database_services() { :; }
COMPOSE_PROFILE=runtime
RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS=60
docker() {
    case "$1" in
        ps)
            [[ "$*" == *"label=com.docker.compose.project=$PROJECT"* ]] || return 9
            if [[ "$*" == *"service=runtime-effects"* ]]; then
                printf 'effects-id\neffects-old-id\n'
            elif [[ "$*" == *"service=runtime-worker"* ]]; then
                printf 'worker-id\n'
            else return 9; fi ;;
        stop | rm) printf '%s\n' "$*" ;;
        *) return 9 ;;
    esac
}
compose_up_bounded() { shift; printf 'UP %s\n' "$*"; }
restart_services /tmp "$PROJECT"
"""
    )
    result = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=True
    )
    lines = result.stdout.splitlines()
    if override:
        assert not any(line.startswith(("stop ", "rm ")) for line in lines)
        assert not any(line.startswith("UP ") for line in lines)
        return
    up = next(i for i, line in enumerate(lines) if line.startswith("UP "))
    assert (
        lines.index("stop --time 30 effects-id effects-old-id")
        < lines.index("rm effects-id effects-old-id")
        < up
    )
    assert "runtime-effects" not in lines[up]
    if "workers" in profiles:
        assert (
            lines.index("stop --time 30 worker-id") < lines.index("rm worker-id") < up
        )
        assert "runtime-worker" not in lines[up]
    assert " -v" not in result.stdout


def test_empty_folded_build_scope_never_invokes_compose() -> None:
    harness = (
        _harness("effects", FOLDING_PROJECT, override=True)
        + "\n"
        + _function("build_images")
        + r"""
log_step() { :; }
log_info() { printf '%s\n' "$*"; }
resolve_compose_file_args() { local -n out="$1"; out=(); }
docker() { printf 'unexpected docker invocation\n' >&2; return 9; }
build_images /tmp "$PROJECT" unused
"""
    )
    result = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=True
    )
    assert "No runtime services remain to build." in result.stdout
    assert not result.stderr


def test_scoped_worker_restart_preserves_effects() -> None:
    harness = (
        _harness("effects", FOLDING_PROJECT, override=True)
        + "\n"
        + _function("restart_services")
        + r"""
RUNTIME_BUILD_SERVICES=(runtime-worker)
RUNTIME_BUILD_SERVICES_OVERRIDE=runtime-worker
log_step() { :; }
log_info() { :; }
log_cmd() { :; }
resolve_compose_file_args() { local -n out="$1"; out=(); }
exclude_cloud_database_services() { :; }
COMPOSE_PROFILE=runtime
RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS=60
docker() { printf 'unexpected effects retirement\n' >&2; return 9; }
compose_up_bounded() { shift; printf 'UP %s\n' "$*"; }
restart_services /tmp "$PROJECT"
"""
    )
    result = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=True
    )
    assert "runtime-worker" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize("override", [False, True])
def test_unknown_profile_propagates_when_caller_handles_failure(override: bool) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            _harness("unknown-role", FOLDING_PROJECT, override)
            + '\nif resolve_lane_runtime_services services "$PROJECT"; then exit 9; fi',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Unknown folded runtime profile: unknown-role" in result.stderr


def test_dev_worker_fold_filters_the_scoped_build_and_restart() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            _harness("effects", DEV_PROJECT)
            + "\nDEV_RUNTIME_MAIN_FOLDED_PROFILES=workers"
            + '\nresolve_lane_runtime_services services "$PROJECT"\ndeclare -p services',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "runtime-worker" not in result.stdout
    assert "runtime-effects" in result.stdout


@pytest.mark.parametrize("profile", ["worker", "workers"])
def test_worker_profile_spellings_remain_accepted(profile: str) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            _harness(profile, FOLDING_PROJECT)
            + '\nresolve_lane_runtime_services services "$PROJECT"\ndeclare -p services',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "runtime-worker" not in result.stdout
    assert "omninode-runtime" in result.stdout
