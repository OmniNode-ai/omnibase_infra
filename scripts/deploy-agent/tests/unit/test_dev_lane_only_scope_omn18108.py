# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18108 -- a dev-lane deploy must reach the dev-lane-only services.

``SCOPE_SERVICES[Scope.RUNTIME]`` is lane-agnostic and lists ten services.
``docker/docker-compose.dev-lane.yml`` declares eight more that exist on no
other lane, and ``scripts/deploy-runtime.sh`` carries them in
``DEV_LANE_ONLY_RUNTIME_SERVICES``. That array is expanded only inside that
script, so it is reachable only through ``deploy-runtime.sh`` /
``refresh_dev_lane.sh`` -- never through the deploy agent.

The consequence is not intermittent. EVERY agent-path deploy to the dev lane
advanced the runtime family and left those eight behind. Measured on the .201
dev lane 2026-09-10T00:45Z: the runtime family carried image revision
``4598a4358bd9f59528875b8b320b6cde54383fb1`` (the deploy agent's own build,
2026-09-09T23:52:50Z) while all eight carried ``3461e4b0aeae`` from the day
before, roughly 35 infra commits behind. ``restart: unless-stopped`` keeps a
stale image running and healthy, which is why nothing reported it --
``refresh_dev_lane.sh`` names that same hazard in its own comment.

The build half matters as much as the up half. ``_compose_build`` passes only
the BASE compose file, while ``_compose_up`` passes the lane's overlay too, so
adding the eight to the up list alone would recreate them from their existing
stale images and report success.

``onex-api`` is in the array for a different reason and this file says so
rather than leaving it to be discovered: it is TAG-REFERENCED
(``${ONEX_API_IMAGE}``), not lane-built. A governed deploy can RECREATE it so
it reads a new environment; it cannot ADVANCE its tag, because no sanctioned
script builds that image or repoints the operator env file.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import (
    DEV_LANE_ONLY_BUILDABLE_SERVICES,
    DEV_LANE_ONLY_RUNTIME_SERVICES,
    DEV_LANE_ONLY_TAG_REFERENCED_SERVICES,
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
    services_for_scope,
)
from deploy_agent.executor import DeployExecutor, _requested_services_for_up

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
DEV_LANE_OVERLAY_BASENAME = "docker-compose.dev-lane.yml"


def _bash_array(name: str) -> list[str]:
    """Return the entries of a ``readonly NAME=( ... )`` array in the script.

    Parsed rather than duplicated: this is the whole point of the anti-drift
    test below. A comment line inside the array is ignored; every other
    non-empty token is an entry.
    """
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        rf"^readonly\s+{re.escape(name)}=\((?P<body>.*?)^\)", text, re.M | re.S
    )
    if match is None:  # pragma: no cover - defended by its own test below
        raise AssertionError(f"{name} array not found in {DEPLOY_SCRIPT}")
    entries: list[str] = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            entries.extend(line.split())
    return entries


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


def _ok() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


class TestTheArrayIsFoundAtAll:
    """A positive control: an empty parse must never read as agreement."""

    def test_dev_lane_only_array_is_non_empty(self) -> None:
        assert len(_bash_array("DEV_LANE_ONLY_RUNTIME_SERVICES")) >= 8


class TestDevLaneRuntimeScope:
    def test_dev_runtime_scope_covers_every_dev_lane_only_service(self) -> None:
        resolved = services_for_scope(Scope.RUNTIME, lane=EnumRuntimeLane.DEV)
        missing = [s for s in DEV_LANE_ONLY_RUNTIME_SERVICES if s not in resolved]
        assert not missing, (
            "a DEV-lane runtime deploy does not reach these services, so every "
            f"agent rebuild leaves them stale: {missing}"
        )

    def test_dev_runtime_scope_still_covers_the_base_runtime_family(self) -> None:
        base = services_for_scope(Scope.RUNTIME)
        resolved = services_for_scope(Scope.RUNTIME, lane=EnumRuntimeLane.DEV)
        assert resolved[: len(base)] == base

    def test_dev_full_scope_covers_every_dev_lane_only_service(self) -> None:
        resolved = services_for_scope(Scope.FULL, lane=EnumRuntimeLane.DEV)
        missing = [s for s in DEV_LANE_ONLY_RUNTIME_SERVICES if s not in resolved]
        assert not missing, missing

    def test_dev_core_scope_is_not_widened(self) -> None:
        # The eight are runtime-profile services. Core scope is postgres /
        # redpanda / valkey and must stay exactly that on every lane.
        assert services_for_scope(Scope.CORE, lane=EnumRuntimeLane.DEV) == (
            services_for_scope(Scope.CORE)
        )

    def test_requested_services_for_up_on_dev_carries_the_dev_lane_only_set(
        self,
    ) -> None:
        resolved = _requested_services_for_up(
            Scope.RUNTIME, [], lane=EnumRuntimeLane.DEV
        )
        missing = [s for s in DEV_LANE_ONLY_RUNTIME_SERVICES if s not in resolved]
        assert not missing, missing

    def test_an_explicit_service_list_is_still_honoured_verbatim_on_dev(self) -> None:
        requested = ["omninode-runtime"]
        assert (
            _requested_services_for_up(
                Scope.RUNTIME, requested, lane=EnumRuntimeLane.DEV
            )
            == requested
        )


class TestProdAndStabilityScopeUnchanged:
    """AC3. The eight services are declared on no other lane at all."""

    @pytest.mark.parametrize(
        "lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST]
    )
    def test_runtime_scope_is_the_base_list(self, lane: EnumRuntimeLane) -> None:
        assert services_for_scope(Scope.RUNTIME, lane=lane) == services_for_scope(
            Scope.RUNTIME
        )

    @pytest.mark.parametrize(
        "lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST]
    )
    def test_full_scope_is_the_base_list(self, lane: EnumRuntimeLane) -> None:
        assert services_for_scope(Scope.FULL, lane=lane) == services_for_scope(
            Scope.FULL
        )

    @pytest.mark.parametrize(
        "lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST]
    )
    def test_no_dev_lane_only_service_leaks_onto_another_lane(
        self, lane: EnumRuntimeLane
    ) -> None:
        resolved = set(services_for_scope(Scope.FULL, lane=lane))
        assert resolved.isdisjoint(set(DEV_LANE_ONLY_RUNTIME_SERVICES))

    def test_omitting_the_lane_returns_the_lane_agnostic_base_list(self) -> None:
        # An unlanded caller must not silently acquire dev-lane services.
        assert services_for_scope(Scope.RUNTIME) == services_for_scope(
            Scope.RUNTIME, lane=None
        )


class TestTheTwoListsCannotDrift:
    """AC4. Bidirectional, so editing either side alone is a red test."""

    def test_python_declaration_equals_the_bash_array(self) -> None:
        from_bash = _bash_array("DEV_LANE_ONLY_RUNTIME_SERVICES")
        assert set(DEV_LANE_ONLY_RUNTIME_SERVICES) == set(from_bash), (
            "scripts/deploy-runtime.sh DEV_LANE_ONLY_RUNTIME_SERVICES and "
            "deploy_agent.events.DEV_LANE_ONLY_RUNTIME_SERVICES have drifted. "
            f"only in python: {sorted(set(DEV_LANE_ONLY_RUNTIME_SERVICES) - set(from_bash))}; "
            f"only in bash: {sorted(set(from_bash) - set(DEV_LANE_ONLY_RUNTIME_SERVICES))}"
        )

    def test_no_duplicate_entries_on_either_side(self) -> None:
        from_bash = _bash_array("DEV_LANE_ONLY_RUNTIME_SERVICES")
        assert len(from_bash) == len(set(from_bash))
        assert len(DEV_LANE_ONLY_RUNTIME_SERVICES) == len(
            set(DEV_LANE_ONLY_RUNTIME_SERVICES)
        )

    def test_dev_lane_only_services_are_disjoint_from_the_base_runtime_list(
        self,
    ) -> None:
        # Membership in the base list would deploy them to prod, stability-test
        # and judge, where the service name does not exist in the merged
        # compose -- a fatal `no such service`.
        assert set(DEV_LANE_ONLY_RUNTIME_SERVICES).isdisjoint(
            set(services_for_scope(Scope.RUNTIME))
        )


class TestOnexApiIsRecreateOnlyNotBuild:
    """AC5. Stated in a test, not left to be discovered on the lane."""

    def test_onex_api_is_declared_tag_referenced(self) -> None:
        assert "onex-api" in DEV_LANE_ONLY_TAG_REFERENCED_SERVICES

    def test_a_tag_referenced_service_is_excluded_from_the_build_set(self) -> None:
        assert "onex-api" not in DEV_LANE_ONLY_BUILDABLE_SERVICES

    def test_it_is_still_in_the_recreate_set(self) -> None:
        # The RECREATE is the point: a container that is never recreated never
        # reads a new environment, and onex-api carries the lane's broker
        # credentials and eleven fail-closed variables.
        assert "onex-api" in services_for_scope(Scope.RUNTIME, lane=EnumRuntimeLane.DEV)

    def test_the_buildable_set_is_the_array_minus_the_tag_referenced_ones(
        self,
    ) -> None:
        assert set(DEV_LANE_ONLY_BUILDABLE_SERVICES) == (
            set(DEV_LANE_ONLY_RUNTIME_SERVICES) - DEV_LANE_ONLY_TAG_REFERENCED_SERVICES
        )


class TestTheBuildPhaseReachesThem:
    """AC2's build half: recreating from a stale image advances nothing."""

    def _captured_build_commands(self, lane: EnumRuntimeLane) -> list[list[str]]:
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            captured.append(list(cmd))
            return _ok()

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
                lane=lane,
            )
        return [c for c in captured if "build" in c]

    def test_a_dev_build_names_the_dev_lane_only_buildable_services(self) -> None:
        builds = self._captured_build_commands(EnumRuntimeLane.DEV)
        flat = [tok for cmd in builds for tok in cmd]
        missing = [s for s in DEV_LANE_ONLY_BUILDABLE_SERVICES if s not in flat]
        assert not missing, (
            "the build phase never reaches these services, so the up phase "
            f"recreates them from their existing stale images: {missing}"
        )

    def test_a_dev_build_passes_the_dev_lane_overlay(self) -> None:
        builds = self._captured_build_commands(EnumRuntimeLane.DEV)
        flat = [tok for cmd in builds for tok in cmd]
        assert any(DEV_LANE_OVERLAY_BASENAME in tok for tok in flat), (
            "without the lane overlay compose cannot resolve these services at all"
        )

    def test_a_stability_build_names_no_dev_lane_only_service(self) -> None:
        builds = self._captured_build_commands(EnumRuntimeLane.STABILITY_TEST)
        flat = {tok for cmd in builds for tok in cmd}
        assert flat.isdisjoint(set(DEV_LANE_ONLY_RUNTIME_SERVICES))

    def test_a_stability_build_does_not_pass_the_dev_lane_overlay(self) -> None:
        builds = self._captured_build_commands(EnumRuntimeLane.STABILITY_TEST)
        flat = [tok for cmd in builds for tok in cmd]
        assert not any(DEV_LANE_OVERLAY_BASENAME in tok for tok in flat)


class TestTheCommandModelAcceptsThem:
    """A dev command naming one of the eight must validate, and only on dev."""

    def test_dev_command_may_name_a_dev_lane_only_service(self) -> None:
        cmd = ModelRebuildRequested(
            correlation_id="11111111-1111-4111-8111-111111111111",  # type: ignore[arg-type]
            requested_by="test",
            scope=Scope.RUNTIME,
            runtime_lane=EnumRuntimeLane.DEV,
            git_ref="origin/dev",
            services=["projection-live-events-writer"],
        )
        assert cmd.services == ["projection-live-events-writer"]

    def test_stability_command_may_not(self) -> None:
        with pytest.raises(ValueError):
            ModelRebuildRequested(
                correlation_id="11111111-1111-4111-8111-111111111111",  # type: ignore[arg-type]
                requested_by="test",
                scope=Scope.RUNTIME,
                runtime_lane=EnumRuntimeLane.STABILITY_TEST,
                git_ref="origin/dev",
                services=["projection-live-events-writer"],
            )
