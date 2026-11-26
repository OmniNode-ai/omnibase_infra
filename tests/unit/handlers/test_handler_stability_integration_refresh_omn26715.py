# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability integration refresh (OMN-26715): order, rollback target and health-gated rollback."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnibase_infra.handlers.handler_stability_integration_refresh import (
    HandlerStabilityIntegrationRefresh,
)
from omnibase_infra.models.model_stability_integration_refresh_request import (
    ModelStabilityIntegrationRefreshRequest,
)

pytestmark = pytest.mark.unit

_HEADS = {"omnibase_core": "c" * 40, "omnibase_infra": "i" * 40, "omnimarket": "m" * 40}


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Host:
    def __init__(
        self,
        *,
        health_after: float | None,
        build_error: str = "",
        heads: dict[str, str] | None = None,
        clock: _Clock,
    ) -> None:
        self.calls: list[str] = []
        self.recreated: list[tuple[tuple[str, str], ...]] = []
        self._health_after = health_after
        self._build_error = build_error
        self._heads = _HEADS if heads is None else heads
        self._clock = clock
        self._live = {
            "omninode-runtime": "sha256:old-rt",
            "runtime-worker": "sha256:old-wk",
            "projection-api": "sha256:old-pa",
        }

    def resolve_head(self, repository: str, branch: str) -> str:
        assert branch == "integration/launch-20261126"
        return self._heads.get(repository, "")

    def cut_deploy_tags(
        self, heads: Sequence[tuple[str, str]]
    ) -> tuple[tuple[str, str], ...]:
        self.calls.append("tag")
        return tuple((repo, f"deploy-20261126213000-{sha[:12]}") for repo, sha in heads)

    def build_image(
        self, heads: Sequence[tuple[str, str]], tags: Sequence[tuple[str, str]]
    ) -> str:
        self.calls.append("build")
        self.built_with = tuple(tags)
        if self._build_error:
            raise RuntimeError(self._build_error)
        return "sha256:new"

    def service_image_id(self, compose_project: str, service: str) -> str:
        return self._live[service]

    def recreate(
        self, compose_project: str, services: Sequence[tuple[str, str]]
    ) -> None:
        self.calls.append("recreate")
        self.recreated.append(tuple(services))
        self._live.update(dict(services))
        self._recreated_at = self._clock.now

    def health_status(self, url: str) -> int:
        if self._health_after is None:
            return 503
        serving = all(image == "sha256:new" for image in self._live.values())
        return (
            200
            if serving and self._clock.now - self._recreated_at >= self._health_after
            else 503
        )


def _request(tmp_path: Path) -> ModelStabilityIntegrationRefreshRequest:
    return ModelStabilityIntegrationRefreshRequest(
        integration_date="20261126",
        repositories=tuple(_HEADS),
        compose_project="stability-project",
        services=("omninode-runtime", "runtime-worker", "projection-api"),
        health_url="http://stability.invalid/health",
        health_window_seconds=300,
        record_dir=str(tmp_path),
    )


def _run(host: _Host, clock: _Clock, tmp_path: Path):
    container = MagicMock()
    container.get_service.return_value = host
    handler = HandlerStabilityIntegrationRefresh(container)
    handler.clock = clock
    handler.sleep = clock.sleep
    return handler.handle(_request(tmp_path))


def test_healthy_build_runs_on_the_new_image_and_records_the_rollback_target(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    host = _Host(health_after=60.0, clock=clock)
    result = _run(host, clock, tmp_path)
    assert result.outcome == "refreshed"
    assert result.built_image_id == "sha256:new"
    assert result.seconds_to_healthy == pytest.approx(60.0, abs=5.0)
    assert dict(result.rollback_target)["omninode-runtime"] == "sha256:old-rt"
    record = json.loads(next(tmp_path.glob("rollback-*.json")).read_text())
    assert record["rollback_target"]["projection-api"] == "sha256:old-pa"
    assert host.calls == ["tag", "build", "recreate"]
    assert dict(result.deploy_tags) == {
        repo: f"deploy-20261126213000-{sha[:12]}" for repo, sha in _HEADS.items()
    }
    assert host.built_with == result.deploy_tags


def test_health_past_five_minutes_restores_the_recorded_images(tmp_path: Path) -> None:
    clock = _Clock()
    host = _Host(health_after=None, clock=clock)
    result = _run(host, clock, tmp_path)
    assert result.outcome == "rolled_back"
    assert host.recreated[-1] == (
        ("omninode-runtime", "sha256:old-rt"),
        ("runtime-worker", "sha256:old-wk"),
        ("projection-api", "sha256:old-pa"),
    )
    assert clock.now >= 300.0


def test_a_failed_build_leaves_the_lane_untouched(tmp_path: Path) -> None:
    clock = _Clock()
    host = _Host(
        health_after=0.0,
        build_error="FAIL omnibase-core: no dev-merge tag",
        clock=clock,
    )
    result = _run(host, clock, tmp_path)
    assert result.outcome == "build_failed"
    assert "no dev-merge tag" in result.detail
    assert host.recreated == []
    assert dict(result.rollback_target)["runtime-worker"] == "sha256:old-wk"


def test_a_missing_integration_branch_refuses_before_any_build(tmp_path: Path) -> None:
    clock = _Clock()
    host = _Host(
        health_after=0.0,
        heads={"omnibase_core": "c" * 40, "omnibase_infra": "i" * 40},
        clock=clock,
    )
    result = _run(host, clock, tmp_path)
    assert result.outcome == "refused"
    assert "omnimarket" in result.detail
    assert host.calls == []
    assert not list(tmp_path.glob("rollback-*.json"))
