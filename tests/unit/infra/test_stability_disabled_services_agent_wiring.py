# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability lane's disabled services never reach the dev lane's containers.

The stability lane lists ``autoheal`` and ``context-audit-consumer`` in its
``disabled_services``. Two readers of that field decide whether that is safe:

* the retire path (``_retire_folded_dev_services``) stops and removes containers, and
  must only ever do so for the lane's own compose project, never by the base file's
  fixed ``container_name`` (which the dev lane's containers hold);
* the compose-up call site (``_compose_up``), where an explicit service request the lane
  filters to nothing must never become an unnamed ``compose up`` that starts the whole
  profile.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import (
    DeployExecutor,
    _requested_services_for_up,
    lane_config_for,
)

from tests.unit.infra.test_lane_container_name_disjointness import (
    _container_names,
    _load_services,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE = REPO_ROOT / "docker/docker-compose.infra.yml"
STABILITY = EnumRuntimeLane.STABILITY_TEST
DISABLED_BY_OVERLAY = {"autoheal", "context-audit-consumer"}
STABILITY_PROJECT = lane_config_for(STABILITY).compose_project
DEV_PROJECT = lane_config_for(EnumRuntimeLane.DEV).compose_project


def _base_container_names() -> dict[str, str]:
    return _container_names(_load_services(BASE_COMPOSE))


@pytest.fixture(autouse=True)
def _committed_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read this checkout's runtime policy, not the deploy host's clone path."""
    monkeypatch.setattr(
        "deploy_agent.executor.RUNTIME_POLICY_ENV_FILE",
        REPO_ROOT / "docker/runtime-policy.env",
    )


def _docker_census(
    calls: list[list[str]],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A fake ``docker`` over both lanes' containers; ``ps`` honours label filters."""
    census = {
        "dev-autoheal": {"project": DEV_PROJECT, "service": "autoheal"},
        "dev-audit": {"project": DEV_PROJECT, "service": "context-audit-consumer"},
        "dev-effects": {"project": DEV_PROJECT, "service": "runtime-effects"},
        "stab-effects": {"project": STABILITY_PROJECT, "service": "runtime-effects"},
        "stab-main": {"project": STABILITY_PROJECT, "service": "omninode-runtime"},
        "stab-autoheal": {"project": STABILITY_PROJECT, "service": "autoheal"},
        "stab-audit": {
            "project": STABILITY_PROJECT,
            "service": "context-audit-consumer",
        },
    }

    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:3] != ["docker", "ps", "-aq"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        wanted: dict[str, str] = {}
        for flag, spec in pairwise(cmd):
            if flag == "--filter" and spec.startswith("label=com.docker.compose."):
                key, _sep, value = spec.removeprefix(
                    "label=com.docker.compose."
                ).partition("=")
                wanted[key] = value
        ids = [
            name
            for name, labels in census.items()
            if all(labels[key] == value for key, value in wanted.items())
        ]
        return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(ids), stderr="")

    return run


def _assert_project_scoped_only(calls: list[list[str]]) -> None:
    base_names = set(_base_container_names().values())
    assert {"omnibase-infra-autoheal", "omninode-context-audit-consumer"} <= base_names
    for cmd in calls:
        assert not base_names & set(cmd), f"{cmd} names a dev lane container"
        if cmd[:3] == ["docker", "ps", "-aq"]:
            assert f"label=com.docker.compose.project={STABILITY_PROJECT}" in cmd
            assert not any(part.rpartition("=")[2] == DEV_PROJECT for part in cmd)
        else:
            touched = set(cmd[2:])
            assert not touched & {"dev-autoheal", "dev-audit", "dev-effects"}


def test_stability_disabled_services_retire_path_is_project_scoped() -> None:
    """Retiring what the lane disables is a census inside the lane's own project."""
    config = lane_config_for(STABILITY)
    assert config.disabled_services >= DISABLED_BY_OVERLAY

    calls: list[list[str]] = []
    with patch.object(executor_mod, "_run", side_effect=_docker_census(calls)):
        DeployExecutor()._retire_folded_dev_services(STABILITY)

    # Positive control: the retire path did act, on the lane's own folded service.
    assert any(cmd[:2] == ["docker", "stop"] for cmd in calls)
    assert [c[-1] for c in calls if c[1] == "rm"] == ["stab-effects"]
    _assert_project_scoped_only(calls)


def test_stability_disabled_services_retire_even_when_policy_folds_them() -> None:
    """If a fold policy ever retired the two, the census still stays in-project.

    The retire path reads ``_folded_dev_services & disabled_services``. Forcing the
    policy to fold both services makes the retire path walk them; the dev lane's
    containers of the same service names must still never be listed, stopped or
    removed, and no command names the base file's fixed ``container_name``.
    """
    calls: list[list[str]] = []
    folded = {"runtime-effects", *DISABLED_BY_OVERLAY}
    with (
        patch.object(executor_mod, "_run", side_effect=_docker_census(calls)),
        patch.object(executor_mod, "_folded_dev_services", return_value=folded),
    ):
        DeployExecutor()._retire_folded_dev_services(STABILITY)

    removed = {name for cmd in calls if cmd[1] == "rm" for name in cmd[2:]}
    # Positive control: the lane's own containers of these services are the ones retired.
    assert {"stab-autoheal", "stab-audit", "stab-effects"} <= removed
    assert not removed & {"dev-autoheal", "dev-audit", "dev-effects", "stab-main"}
    _assert_project_scoped_only(calls)


class _StopAtComposeError(Exception):
    """Raised by the fake `_run` once the runtime phase reaches a compose command."""


def _drive_runtime_phase(
    lane: EnumRuntimeLane, services: list[str]
) -> tuple[list[list[str]], list[PhaseStatus]]:
    """Run ``_compose_up`` for the runtime scope; return the commands and statuses.

    The fake ``_run`` stops the phase at its first command, so only what the phase
    issues first is recorded.
    """
    executor = DeployExecutor()
    budget = SimpleNamespace(timeout_seconds=600, describe=lambda: "test budget")
    commands: list[list[str]] = []
    statuses: list[PhaseStatus] = []

    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        raise _StopAtComposeError

    with (
        patch.object(executor_mod, "_run", side_effect=run),
        patch.object(executor_mod, "_compose_env", lambda *a, **k: {}),
        patch.object(executor_mod, "finish_recreate_renames", lambda lane: []),
        patch.object(executor_mod, "preflight_dev_images", lambda *a, **k: None),
        patch.object(executor_mod, "runtime_compose_up_budget", lambda *a: budget),
        patch.object(executor_mod, "runtime_health_probe_budget", lambda *a: 900),
        patch.object(executor, "_ensure_runtime_migrations_ready"),
        patch.object(executor, "rerun_changed_oneshots", return_value=[]),
        patch.object(executor, "_retire_folded_dev_services"),
        contextlib.suppress(_StopAtComposeError),
    ):
        executor._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            services,
            lambda phase, status: statuses.append(status),
            lane=lane,
        )
    return commands, statuses


DISABLED_ONLY = ["autoheal"]


def test_requested_services_for_up_filters_a_disabled_only_request_to_empty() -> None:
    assert _requested_services_for_up(Scope.RUNTIME, ["autoheal"], lane=STABILITY) == []
    assert (
        _requested_services_for_up(
            Scope.RUNTIME, ["autoheal", "context-audit-consumer"], lane=STABILITY
        )
        == []
    )
    # Positive controls: the lane-agnostic request keeps the name, and an empty request
    # on the lane still expands to the lane's full runtime list.
    assert _requested_services_for_up(Scope.RUNTIME, ["autoheal"], lane=None) == [
        "autoheal"
    ]
    assert "omninode-runtime" in _requested_services_for_up(
        Scope.RUNTIME, [], lane=STABILITY
    )


def test_requested_services_for_up_empty_result_never_becomes_unnamed_compose_up() -> (
    None
):
    """A one-service request the lane filters out is skipped, not widened."""
    commands, statuses = _drive_runtime_phase(STABILITY, DISABLED_ONLY)
    assert commands == [], f"the phase ran {commands} for a disabled-only request"
    assert statuses[-1] is PhaseStatus.SUCCESS


def test_requested_services_for_up_positive_control_reaches_a_named_compose_up() -> (
    None
):
    """A request the lane keeps does reach compose, with the name on the argv."""
    commands, _ = _drive_runtime_phase(STABILITY, ["runtime-worker"])
    up = [cmd for cmd in commands if "up" in cmd]
    assert up, f"no compose up reached for a kept service: {commands}"
    assert up[0][-1] == "runtime-worker"
