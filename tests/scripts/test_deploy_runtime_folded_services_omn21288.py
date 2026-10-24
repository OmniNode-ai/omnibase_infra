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


def _harness(profiles: str, project: str, override: bool = False) -> str:
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


@pytest.mark.parametrize("profiles", ["effects", "effects,worker"])
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
        assert "stop effects-id effects-old-id" in lines
        assert "rm effects-id effects-old-id" in lines
        assert not any(line.startswith("UP ") for line in lines)
        return
    up = next(i for i, line in enumerate(lines) if line.startswith("UP "))
    assert (
        lines.index("stop effects-id effects-old-id")
        < lines.index("rm effects-id effects-old-id")
        < up
    )
    assert "runtime-effects" not in lines[up]
    if "worker" in profiles:
        assert lines.index("stop worker-id") < lines.index("rm worker-id") < up
        assert "runtime-worker" not in lines[up]
    assert " -v" not in result.stdout
