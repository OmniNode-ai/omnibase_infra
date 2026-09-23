# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One BUILD_DATE per rebuild, shared by every compose build in it (OMN-19220).

``Dockerfile.runtime`` declares ``ARG BUILD_DATE`` at the top of both stages, so
a changed value misses the cache for every RUN after it. The dev-lane-only
build (OMN-18108) is a second ``docker compose build`` of the same Dockerfile
and args; when it stamped its own clock it re-ran the whole image from scratch
(2% cached on .201, 2026-09-22) and two consecutive dev-lane rebuilds died at
the 3600s hard bound AFTER their first build had succeeded.

The fake runner sleeps past a second boundary on every build, so a
per-call clock would produce two different values here and the shared one
cannot.
"""

from __future__ import annotations

import subprocess
import time
from unittest.mock import patch

import pytest
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


def _build_dates(executor: DeployExecutor, scope: Scope) -> list[str]:
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured.append(list(cmd))
        if "build" in cmd:
            time.sleep(1.1)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with (
        patch("deploy_agent.executor._run", side_effect=_fake_run),
        patch.object(DeployExecutor, "_compose_up", return_value=None),
        patch.object(DeployExecutor, "_deploy_gateway_lane", return_value=None),
    ):
        executor.rebuild_scope(
            scope,
            [],
            _noop_phase_update,
            git_sha="0" * 40,
            git_ref="origin/dev",
            build_source=BuildSource.RELEASE,
            lane=EnumRuntimeLane.DEV,
        )
    return [
        tok.split("=", 1)[1]
        for cmd in captured
        if "build" in cmd
        for tok in cmd
        if tok.startswith("BUILD_DATE=")
    ]


@pytest.mark.unit
class TestOneBuildDatePerDeploy:
    def test_the_dev_lane_only_build_reuses_the_runtime_builds_date(self) -> None:
        dates = _build_dates(DeployExecutor(), Scope.RUNTIME)
        assert len(dates) == 2, dates
        assert len(set(dates)) == 1, (
            "the dev-lane-only build carried its own BUILD_DATE, which misses "
            f"the whole image's cache and rebuilds it from scratch: {dates}"
        )

    def test_every_build_in_a_full_rebuild_shares_one_date(self) -> None:
        dates = _build_dates(DeployExecutor(), Scope.FULL)
        assert len(dates) >= 2, dates
        assert len(set(dates)) == 1, dates

    def test_the_next_rebuild_gets_a_fresh_date(self) -> None:
        executor = DeployExecutor()
        first = _build_dates(executor, Scope.RUNTIME)
        second = _build_dates(executor, Scope.RUNTIME)
        assert set(first).isdisjoint(second), (
            "a date carried across rebuilds would stamp a new image with the "
            f"previous deploy's build time: {first} then {second}"
        )
