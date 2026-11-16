# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability follow retires the folded effects container that holds :18086 (OMN-24562).

Main publishes the effects port once the effects role is folded, so a runtime-effects
container left by an older compose file collides with main on the host port. The
stability lane retires exactly the profile-mapped services, and nothing else.
"""

from __future__ import annotations

import subprocess
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Phase, Scope
from deploy_agent.executor import (
    DeployExecutor,
    _requested_services_for_up,
    lane_config_for,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
STABILITY_PROJECT = "label=com.docker.compose.project=" + (
    lane_config_for(EnumRuntimeLane.STABILITY_TEST).compose_project
)


@pytest.fixture(autouse=True)
def _committed_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read this checkout's runtime policy, not the deploy host's clone path."""
    monkeypatch.setattr(
        "deploy_agent.executor.RUNTIME_POLICY_ENV_FILE",
        REPO_ROOT / "docker/runtime-policy.env",
    )


def test_stability_disables_only_the_folded_effects_service() -> None:
    config = lane_config_for(EnumRuntimeLane.STABILITY_TEST)
    assert config.disabled_services == frozenset({"runtime-effects"})
    requested = _requested_services_for_up(
        Scope.RUNTIME, [], lane=EnumRuntimeLane.STABILITY_TEST
    )
    assert "runtime-effects" not in requested
    # Positive control: the services the lane still runs by name stay requested.
    assert {"omninode-runtime", "runtime-worker"} <= set(requested)


def test_stability_retires_the_old_effects_container_and_keeps_volumes() -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="935bf70c5d47\n" if cmd[1] == "ps" else "", stderr=""
        )

    with patch("deploy_agent.executor._run", side_effect=run):
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.STABILITY_TEST)

    assert calls == [
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            STABILITY_PROJECT,
            "--filter",
            "label=com.docker.compose.service=runtime-effects",
        ],
        ["docker", "stop", "--time", "30", "935bf70c5d47"],
        ["docker", "rm", "935bf70c5d47"],
    ]
    assert all("-v" not in c and "--volumes" not in c for c in calls)


def test_stability_retires_only_containers_of_its_own_compose_project() -> None:
    """Two lanes' runtime-effects containers sit in one docker census; one is retired."""
    own = STABILITY_PROJECT.split("=", 2)[2]
    census = {
        "own-effects": {"project": own, "service": "runtime-effects"},
        "own-main": {"project": own, "service": "omninode-runtime"},
        "other-effects": {"project": "other-lane", "service": "runtime-effects"},
        "dev-effects": {"project": "omnibase-infra-dev", "service": "runtime-effects"},
    }
    assert own not in {"other-lane", "omnibase-infra-dev"}
    calls: list[list[str]] = []

    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[1] != "ps":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        wanted = {
            key: value
            for flag, spec in pairwise(cmd)
            if flag == "--filter"
            for key, value in [
                spec.removeprefix("label=com.docker.compose.").split("=", 1)
            ]
        }
        ids = [
            name
            for name, labels in census.items()
            if all(labels[key] == value for key, value in wanted.items())
        ]
        return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(ids), stderr="")

    with patch("deploy_agent.executor._run", side_effect=run):
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.STABILITY_TEST)

    assert [c for c in calls if c[1] != "ps"] == [
        ["docker", "stop", "--time", "30", "own-effects"],
        ["docker", "rm", "own-effects"],
    ]
    touched = {name for c in calls if c[1] != "ps" for name in c[2:]}
    assert touched.isdisjoint({"own-main", "other-effects", "dev-effects"})


def test_stability_with_no_old_container_removes_nothing() -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("deploy_agent.executor._run", side_effect=run):
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.STABILITY_TEST)

    assert [c[1] for c in calls] == ["ps"]


class _StopAfterRetirementError(Exception):
    """Raised by the fake `_run` once the runtime phase reaches `docker compose`."""


def _drive_runtime_phase(
    lane: EnumRuntimeLane, services: list[str], events: list[str]
) -> None:
    """Run `_compose_up` for the runtime scope, recording what it does in order.

    The retirement is replaced by a recorder so the test pins the call site (the
    lane it passes and where it sits before compose) without needing docker.
    """
    executor = DeployExecutor()
    budget = SimpleNamespace(timeout_seconds=600, describe=lambda: "test budget")

    def retire(retired_lane: EnumRuntimeLane, *_: object) -> None:
        events.append(f"retire:{retired_lane.value}")

    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        events.append("compose")
        raise _StopAfterRetirementError

    with (
        patch.object(executor_mod, "_run", side_effect=run),
        patch.object(executor_mod, "_compose_env", lambda *a, **k: {}),
        patch.object(executor_mod, "finish_recreate_renames", lambda lane: []),
        patch.object(executor_mod, "preflight_dev_images", lambda *a, **k: None),
        patch.object(executor_mod, "runtime_compose_up_budget", lambda *a: budget),
        patch.object(executor_mod, "runtime_health_probe_budget", lambda *a: 900),
        patch.object(executor, "_ensure_runtime_migrations_ready"),
        patch.object(executor, "rerun_changed_oneshots", return_value=[]),
        patch.object(executor, "_retire_folded_dev_services", side_effect=retire),
        pytest.raises(_StopAfterRetirementError),
    ):
        executor._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            services,
            lambda phase, status: None,
            lane=lane,
        )


def test_stability_runtime_phase_retires_before_compose_starts_main() -> None:
    events: list[str] = []
    _drive_runtime_phase(EnumRuntimeLane.STABILITY_TEST, [], events)
    assert events == [f"retire:{EnumRuntimeLane.STABILITY_TEST.value}", "compose"]


def test_runtime_phase_without_main_runtime_retires_nothing() -> None:
    """Positive control for the call's guard: main is what takes the port over."""
    events: list[str] = []
    _drive_runtime_phase(EnumRuntimeLane.STABILITY_TEST, ["runtime-worker"], events)
    assert events == ["compose"]


def test_stability_census_failure_fails_the_retirement_loudly() -> None:
    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="docker down")

    with (
        patch.object(executor_mod, "_run", side_effect=run),
        pytest.raises(RuntimeError, match="runtime-effects container census"),
    ):
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.STABILITY_TEST)


def test_stability_stop_failure_fails_before_the_remove() -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[1] == "ps":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="935bf70c5d47\n", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="stop refused")

    with (
        patch.object(executor_mod, "_run", side_effect=run),
        pytest.raises(RuntimeError, match="Cannot retire folded dev containers"),
    ):
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.STABILITY_TEST)

    assert [c[1] for c in calls] == ["ps", "stop"]


def test_prod_runtime_phase_retires_nothing() -> None:
    assert not lane_config_for(EnumRuntimeLane.PROD).disabled_services
    with patch.object(executor_mod, "_run") as run:
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.PROD)
    run.assert_not_called()
