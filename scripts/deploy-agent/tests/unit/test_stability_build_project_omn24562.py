# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability job builds the images its own project runs (OMN-24562).

Job 7179dcca built under the default project, so the images landed under the dev
lane's names while the stability up runs ``--no-build`` against its own project's
images: it found the October images, or none (``No such image``). It also named
``autoheal``, which the stability overlay disables and which would take the dev
lane's container name.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import (
    DeployExecutor,
    _requested_services_for_up,
    lane_config_for,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SHA = "a" * 40


@pytest.fixture(autouse=True)
def _committed_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deploy_agent.executor.RUNTIME_POLICY_ENV_FILE",
        REPO_ROOT / "docker/runtime-policy.env",
    )


def _noop(phase: Phase, status: PhaseStatus) -> None:
    return None


def _build_project_of(lane: EnumRuntimeLane) -> str:
    captured: list[list[str]] = []

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        DeployExecutor()._compose_build(Scope.RUNTIME, SHA, _noop, runtime_lane=lane)
    builds = [cmd for cmd in captured if "build" in cmd]
    assert len(builds) == 1
    return builds[0][builds[0].index("-p") + 1]


def test_stability_builds_under_the_project_it_runs() -> None:
    lane = EnumRuntimeLane.STABILITY_TEST
    project = _build_project_of(lane)
    assert project == lane_config_for(lane).compose_project
    assert project != lane_config_for(EnumRuntimeLane.DEV).compose_project


def test_dev_still_builds_under_its_own_project() -> None:
    lane = EnumRuntimeLane.DEV
    assert _build_project_of(lane) == lane_config_for(lane).compose_project


def test_stability_does_not_request_autoheal_and_keeps_the_rest() -> None:
    requested = _requested_services_for_up(
        Scope.RUNTIME, [], lane=EnumRuntimeLane.STABILITY_TEST
    )
    assert "autoheal" not in requested
    # Positive control: the services the lane runs by name stay requested.
    assert {"omninode-runtime", "runtime-worker", "projection-api"} <= set(requested)


def test_autoheal_is_disabled_by_profile_in_the_stability_overlay() -> None:
    lane = EnumRuntimeLane.STABILITY_TEST.value
    overlay = (REPO_ROOT / f"docker/docker-compose.{lane}.yml").read_text()
    block = overlay.split("\n  autoheal:\n", 1)[1].split("\n  #", 1)[0]
    assert f'profiles: !override ["{lane}-disabled"]' in block
    disabled = lane_config_for(EnumRuntimeLane.STABILITY_TEST).disabled_services
    assert "autoheal" in disabled
