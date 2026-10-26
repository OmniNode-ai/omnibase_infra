# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21279: the overlay build must rebuild every composed runtime service."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from deploy_agent.events import (
    DEV_LANE_BUILD_SERVICES,
    DEV_LANE_COMPOSED_RUNTIME_SERVICES,
    DEV_LANE_ONLY_BUILDABLE_SERVICES,
    SCOPE_SERVICES,
    BuildSource,
    EnumRuntimeLane,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE_TAG = re.compile(r"(?m)(:\s+|-\s+)!(?:override|reset)\b[ \t]*")


def _composed_overlay_services() -> set[str]:
    text = (REPO_ROOT / "docker" / "docker-compose.dev-lane.yml").read_text(
        encoding="utf-8"
    )
    overlay = yaml.safe_load(_COMPOSE_TAG.sub(r"\1", text))
    return {
        name
        for name, service in overlay["services"].items()
        if isinstance(service.get("build"), Mapping)
        and str(service["build"].get("context", "")).endswith(
            "sibling-repos/omnimarket"
        )
    }


def test_dev_build_set_covers_every_composed_overlay_service() -> None:
    composed = _composed_overlay_services()
    assert composed, "no composed runtime services found"
    assert composed == set(DEV_LANE_COMPOSED_RUNTIME_SERVICES) | (
        composed & set(DEV_LANE_ONLY_BUILDABLE_SERVICES)
    )
    assert composed <= set(DEV_LANE_BUILD_SERVICES)


def test_dev_build_set_preserves_the_lane_only_members_without_duplicates() -> None:
    assert len(DEV_LANE_BUILD_SERVICES) == len(set(DEV_LANE_BUILD_SERVICES))
    assert set(DEV_LANE_ONLY_BUILDABLE_SERVICES) <= set(DEV_LANE_BUILD_SERVICES)
    assert DEV_LANE_BUILD_SERVICES[: len(DEV_LANE_ONLY_BUILDABLE_SERVICES)] == (
        DEV_LANE_ONLY_BUILDABLE_SERVICES
    )
    assert DEV_LANE_BUILD_SERVICES[len(DEV_LANE_ONLY_BUILDABLE_SERVICES) :] == tuple(
        service
        for service in DEV_LANE_COMPOSED_RUNTIME_SERVICES
        if service not in DEV_LANE_ONLY_BUILDABLE_SERVICES
    )


def test_composed_base_services_belong_to_the_runtime_scope() -> None:
    assert set(DEV_LANE_COMPOSED_RUNTIME_SERVICES) <= set(SCOPE_SERVICES[Scope.RUNTIME])


def test_dev_overlay_build_argv_names_every_composed_base_service() -> None:
    executor = DeployExecutor()
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured.append(list(cmd))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
        return None

    with (
        patch("deploy_agent.executor._run", side_effect=_fake_run),
        patch.object(DeployExecutor, "_compose_up", return_value=None),
    ):
        executor.rebuild_scope(
            Scope.RUNTIME,
            [],
            _noop_phase_update,
            git_sha="0" * 40,
            git_ref="origin/dev",
            build_source=BuildSource.RELEASE,
            lane=EnumRuntimeLane.DEV,
        )
    builds = [
        cmd
        for cmd in captured
        if "build" in cmd and any("docker-compose.dev-lane.yml" in tok for tok in cmd)
    ]
    assert builds, "the build phase did not pass the dev-lane overlay"
    for cmd in builds:
        assert set(DEV_LANE_COMPOSED_RUNTIME_SERVICES) <= set(cmd)
