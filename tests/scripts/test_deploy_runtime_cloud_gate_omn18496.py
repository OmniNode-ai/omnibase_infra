# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18496: cloud schema failures gate their consumers, not companions.

Execute the deployment functions under the production shell options with only
the Docker boundary replaced. These are scoping regressions, not live mint proof.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/deploy-runtime.sh"
CLOUD_CONSUMERS = ("onex-api",)
pytestmark = pytest.mark.unit


def _function(source: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{.*?^\}}", source, re.M | re.S)
    assert match is not None, name
    return match[0]


def _array(source: str, name: str) -> str:
    match = re.search(rf"^readonly {name}=\(.*?^\)", source, re.M | re.S)
    assert match is not None, name
    return match[0]


def _invoke(
    *,
    failed_service: str = "",
    failure_phase: str = "wait",
    project: str = "omnibase-infra",
    override: str = "",
    stop_fails: bool = False,
    full_stack: bool = False,
    config_fails: bool = False,
    config_empty: bool = False,
) -> subprocess.CompletedProcess[str]:
    source = SCRIPT.read_text()
    arrays = (
        "RUNTIME_SERVICES",
        "DEV_LANE_ONLY_RUNTIME_SERVICES",
        "DEV_LANE_EXTRA_BROKER_CLIENTS",
        "STABILITY_TEST_LANE_ONLY_RUNTIME_SERVICES",
        "RUNTIME_MIGRATION_SERVICES",
        "RUNTIME_MIGRATION_ONESHOTS",
        "DEV_LANE_ONLY_MIGRATION_SERVICES",
        "DEV_LANE_ONLY_MIGRATION_ONESHOTS",
        "REQUIRED_PROJECTION_TABLES",
    )
    functions = (
        "resolve_lane_runtime_services",
        "refresh_one_migration_service",
        "run_runtime_migration_preflight",
        "restart_services",
        "bringup_full_stack",
    )
    # Load the scope helper when present so the pre-fix revision exercises the
    # old failure path, rather than failing merely because a symbol is absent.
    extras = ""
    if "exclude_cloud_database_services() {" in source:
        extras = _array(source, "DEV_LANE_CLOUD_DB_SERVICES") + "\n"
        extras += _function(source, "exclude_cloud_database_services")
    shell = "\n".join(
        [
            "set -euo pipefail",
            "COMPOSE_PROFILE=runtime; RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS=10",
            "DEV_LANE_CLOUD_MIGRATIONS_READY=true",
            'log_error() { echo "ERROR:$*" >&2; }',
            'log_warn() { echo "WARN:$*" >&2; }',
            "log_info() { :; }; log_step() { :; }; log_cmd() { :; }",
            'source "$COMPOSE_HELPER"',
            *(_array(source, name) for name in arrays),
            'RUNTIME_BUILD_SERVICES=("${RUNTIME_SERVICES[@]}")',
            'if [[ -n "$RUNTIME_BUILD_SERVICES_OVERRIDE" ]]; then read -ra RUNTIME_BUILD_SERVICES <<< "$RUNTIME_BUILD_SERVICES_OVERRIDE"; fi',
            extras,
            *(_function(source, name) for name in functions),
            # A function boundary records argv without executing any containers.
            r"""docker() {
    { printf 'DOCKER'; printf ': %s' "$@"; printf '\n'; } >&2
    if [[ "$1" == wait ]]; then
        if [[ "$FAILURE_PHASE" == wait && "$2" == "$PROJECT-$FAILED_SERVICE" ]]; then echo 7; else echo 0; fi
    elif [[ "$1" == exec ]]; then
        if [[ "$FAILED_SERVICE" == projection-table ]]; then echo f; else echo t; fi
    elif [[ " $* " == *" config --services "* ]]; then
        [[ "$CONFIG_FAILS" == false ]] || return 1
        if [[ "$CONFIG_EMPTY" == false ]]; then
            printf '%s\n' omninode-runtime runtime-effects cloud-migration-files cloud-migration onex-api
        fi
    elif [[ " $* " == *" stop "* && "$STOP_FAILS" == true ]]; then
        return 1
    elif [[ "$FAILURE_PHASE" == up && " $* " == *" up "* && "${@: -1}" == "$FAILED_SERVICE" ]]; then
        return 1
    fi
}""",
            'timeout() { shift 2; "$@"; }',
            'compose_up_bounded() { shift; "$@"; }',
            'run_runtime_migration_preflight /proof "$PROJECT"',
            'if [[ "$FULL_STACK" == true ]]; then bringup_full_stack /proof "$PROJECT"; else restart_services /proof "$PROJECT"; fi',
            "echo RUNTIME_REFRESH_COMPLETED",
        ]
    )
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME"}}
    env.update(
        COMPOSE_HELPER=str(ROOT / "scripts/runtime_build/compose_files.sh"),
        PROJECT=project,
        FAILED_SERVICE=failed_service,
        FAILURE_PHASE=failure_phase,
        RUNTIME_BUILD_SERVICES_OVERRIDE=override,
        STOP_FAILS=str(stop_fails).lower(),
        FULL_STACK=str(full_stack).lower(),
        CONFIG_FAILS=str(config_fails).lower(),
        CONFIG_EMPTY=str(config_empty).lower(),
    )
    return subprocess.run(
        ["bash", "-c", shell],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("service", ["cloud-migration-files", "cloud-migration"])
@pytest.mark.parametrize("phase", ["wait", "up"])
def test_cloud_failure_starts_effects_and_stops_only_cloud_consumers(
    service: str, phase: str
) -> None:
    result = _invoke(failed_service=service, failure_phase=phase)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RUNTIME_REFRESH_COMPLETED" in result.stdout
    assert ": stop: --timeout: 30: onex-api" in result.stderr
    restart = result.stderr.split(": up: -d: --no-deps: --force-recreate")[-1]
    assert ": runtime-effects" in restart
    assert ": onex-api" not in restart
    assert "cloud" in result.stderr.lower()
    if service == "cloud-migration-files":
        assert ": wait: omnibase-infra-cloud-migration\n" not in result.stderr


def test_successful_cloud_migration_restores_api_to_refresh_set() -> None:
    result = _invoke()
    assert result.returncode == 0, result.stdout + result.stderr
    assert ": runtime-effects" in result.stderr
    assert ": onex-api" in result.stderr
    assert ": stop:" not in result.stderr


def test_full_profile_failure_still_starts_effects_without_retrying_cloud() -> None:
    result = _invoke(failed_service="cloud-migration", full_stack=True)
    assert result.returncode == 0, result.stdout + result.stderr
    startup = result.stderr.split(": up: -d:")[-1]
    assert ": runtime-effects" in startup
    assert ": onex-api" not in startup
    assert "cloud-migration" not in startup


def test_unreadable_full_profile_cannot_fall_back_to_unscoped_up() -> None:
    result = _invoke(
        failed_service="cloud-migration", full_stack=True, config_fails=True
    )
    assert result.returncode != 0
    assert ": up: -d: omninode-runtime" not in result.stderr


def test_empty_full_profile_cannot_fall_back_to_unscoped_up() -> None:
    result = _invoke(
        failed_service="cloud-migration", full_stack=True, config_empty=True
    )
    assert result.returncode != 0
    assert "Refusing an empty service scope" in result.stderr


@pytest.mark.parametrize("service", ["forward-migration", "intelligence-migration"])
def test_required_migration_failure_still_aborts_runtime_refresh(service: str) -> None:
    result = _invoke(failed_service=service)
    assert result.returncode != 0
    assert "RUNTIME_REFRESH_COMPLETED" not in result.stdout
    assert ": runtime-effects" not in result.stderr


def test_missing_required_projection_table_still_aborts_refresh() -> None:
    result = _invoke(failed_service="projection-table")
    assert result.returncode != 0
    assert "Missing projection table" in result.stderr
    assert ": runtime-effects" not in result.stderr


def test_unstoppable_cloud_consumer_refuses_refresh() -> None:
    result = _invoke(failed_service="cloud-migration", stop_fails=True)
    assert result.returncode != 0
    assert "RUNTIME_REFRESH_COMPLETED" not in result.stdout


def test_cloud_failure_cannot_turn_an_api_only_override_into_full_project_up() -> None:
    result = _invoke(failed_service="cloud-migration", override="onex-api")
    assert result.returncode != 0
    assert "RUNTIME_REFRESH_COMPLETED" not in result.stdout


def test_effects_only_override_never_migrates_or_stops_an_unrequested_api() -> None:
    result = _invoke(failed_service="cloud-migration", override="runtime-effects")
    assert result.returncode == 0, result.stdout + result.stderr
    assert ": runtime-effects" in result.stderr
    assert "cloud-migration" not in result.stderr
    assert ": onex-api" not in result.stderr


@pytest.mark.parametrize(
    "project",
    ["omnibase-infra-stability-test", "omnibase-infra-prod", "omnibase-infra-judge"],
)
def test_other_lanes_never_attempt_cloud_migrations(project: str) -> None:
    result = _invoke(failed_service="cloud-migration", project=project)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cloud-migration" not in result.stderr
    assert ": runtime-effects" in result.stderr


def test_declared_cloud_consumer_set_excludes_companion_runtime() -> None:
    compose = yaml.safe_load(
        (ROOT / "docker/docker-compose.dev-lane.yml")
        .read_text()
        .replace("!override", "")
    )
    consumers = tuple(
        name
        for name, service in compose["services"].items()
        if "cloud-migration" in service.get("depends_on", {})
    )
    assert consumers == CLOUD_CONSUMERS
    api_dependency = compose["services"]["onex-api"]["depends_on"]["cloud-migration"]
    assert api_dependency["condition"] == "service_completed_successfully"
    gate = _array(SCRIPT.read_text(), "DEV_LANE_CLOUD_DB_SERVICES")
    assert tuple(re.findall(r"^    ([a-z][a-z0-9-]*)$", gate, re.M)) == CLOUD_CONSUMERS


def test_degraded_readback_uses_the_same_cloud_scope_as_restart() -> None:
    body = _function(SCRIPT.read_text(), "readback_deployed_ref")
    assert "exclude_cloud_database_services readback_scope" in body


def test_effects_keep_their_declared_companion_dependencies() -> None:
    compose = yaml.safe_load((ROOT / "docker/docker-compose.infra.yml").read_text())
    assert compose["services"]["runtime-effects"]["depends_on"] == {
        "omninode-runtime": {"condition": "service_started"},
        "redpanda": {"condition": "service_healthy"},
    }
