# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24562: the stability build names the overlay and the composed services; dev is unchanged."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from deploy_agent.events import (
    DEV_LANE_BUILD_SERVICES,
    SCOPE_SERVICES,
    BuildSource,
    EnumRuntimeLane,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor, lane_config_for

from omnibase_infra.nodes.node_setup_local_provision_effect.runtime_cost_estimator_build import (
    STABILITY_TEST_COMPOSED_RUNTIME_SERVICES,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
_OVERLAY = f"docker-compose.{EnumRuntimeLane.STABILITY_TEST.value}.yml"
_COMPOSE_TAG = re.compile(r"(?m)(:\s+|-\s+)!(?:override|reset)\b[ \t]*")


def _stability_composed_services() -> set[str]:
    text = (REPO_ROOT / "docker" / _OVERLAY).read_text(encoding="utf-8")
    overlay = yaml.safe_load(_COMPOSE_TAG.sub(r"\1", text))
    return {
        name
        for name, service in overlay["services"].items()
        if isinstance(service.get("build"), Mapping)
        and str(service["build"].get("context", "")).endswith(
            "sibling-repos/omnimarket"
        )
    }


def _builds_for(lane: EnumRuntimeLane, overlay: str) -> list[list[str]]:
    executor = DeployExecutor()
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured.append(list(cmd))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def _noop(phase: Phase, status: PhaseStatus) -> None:
        return None

    with (
        patch("deploy_agent.executor._run", side_effect=_fake_run),
        patch(
            "deploy_agent.executor._hold_build_context",
            side_effect=lambda purpose: nullcontext(),
        ),
        patch.object(DeployExecutor, "_compose_up", return_value=None),
    ):
        executor.rebuild_scope(
            Scope.RUNTIME,
            [],
            _noop,
            git_sha="0" * 40,
            git_ref="origin/dev",
            build_source=BuildSource.RELEASE,
            lane=lane,
        )
    return [
        cmd
        for cmd in captured
        if "build" in cmd and any(overlay in token for token in cmd)
    ]


def test_stability_build_set_is_exactly_the_composed_overlay_services() -> None:
    composed = _stability_composed_services()
    assert composed == set(STABILITY_TEST_COMPOSED_RUNTIME_SERVICES)
    assert len(STABILITY_TEST_COMPOSED_RUNTIME_SERVICES) == len(
        set(STABILITY_TEST_COMPOSED_RUNTIME_SERVICES)
    )


def test_stability_composed_base_services_belong_to_the_runtime_scope() -> None:
    base = {"omninode-runtime", "runtime-worker", "projection-api"}
    assert base <= set(SCOPE_SERVICES[Scope.RUNTIME])


def test_stability_renders_the_composed_build_under_its_own_project() -> None:
    builds = _builds_for(EnumRuntimeLane.STABILITY_TEST, _OVERLAY)
    assert builds, "the stability build did not pass the stability overlay"
    for cmd in builds:
        project = cmd[cmd.index("-p") + 1]
        assert (
            project == lane_config_for(EnumRuntimeLane.STABILITY_TEST).compose_project
        )
        assert project != lane_config_for(EnumRuntimeLane.DEV).compose_project
        assert set(STABILITY_TEST_COMPOSED_RUNTIME_SERVICES) <= set(cmd)


def test_dev_build_is_unchanged_by_the_stability_build() -> None:
    dev_builds = _builds_for(EnumRuntimeLane.DEV, "docker-compose.dev-lane.yml")
    assert dev_builds
    for cmd in dev_builds:
        assert tuple(cmd[-len(DEV_LANE_BUILD_SERVICES) :]) == DEV_LANE_BUILD_SERVICES
        assert "composed-runtime-base-image" not in cmd
        assert not any(_OVERLAY in token for token in cmd)
    assert not _builds_for(EnumRuntimeLane.DEV, _OVERLAY)
