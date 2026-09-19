# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A FULL rebuild converges the deps; it does not recreate the broker (OMN-18640).

The deploy agent's ``_compose_up`` carried ``--force-recreate`` as a CONSTANT
from ``505c6e78d`` (OMN-7409, 2026-04-03), with no recorded reason. OMN-9455
later found the RUNTIME half of that constant destroying core infra and fixed
it by appending ``--no-deps`` to the runtime argv. The CORE half was never
revisited, so the deps leg of every ``scope=full`` rebuild kept issuing

    docker compose ... --profile core up -d --force-recreate --pull always

with no service list -- and ``SCOPE_SERVICES[Scope.CORE]`` is
``[postgres, redpanda, valkey]``. The lane's BROKER was therefore destroyed and
rebuilt on every runtime rebuild.

Measured on the ``.201`` dev lane on 2026-09-18: nine ``scope=full`` commands
were accepted between 16:14 and 21:04 EDT, and the agent's own journal records
a completed ``phase core deps recreate`` for each. Every one of them dropped
the group coordinator for every consumer on that broker -- it wedged
``omninode-runtime-effects`` for 97 minutes that night -- and left partition
leadership reconciling behind it.

It bought nothing. No service in the ``core`` profile declares a ``build:``
section, so a rebuild never produces a new image for the recreate to adopt, and
the agent logs exactly that sentence every time it derives the core image-build
ceiling. Proven against the live lane the same night: ``docker compose
--profile core up -d --dry-run`` with the agent's own environment reported
``omnibase-infra-redpanda Running``, and the identical dry-run with
``--force-recreate --pull always`` reported it ``Recreated``. There was no
config drift for the flag to be responding to. The flag WAS the cause.

What these tests pin, in the terms of the defect:

1. The deps leg of a DEV ``Scope.FULL`` rebuild carries no ``--force-recreate``.
2. The same holds on PROD, whose FULL path is a separate call site.
3. No argv a FULL rebuild issues force-recreates anything in the core scope --
   asserted over EVERY captured command, not just the one this module expects
   to be the deps leg, so a new call site cannot reintroduce the churn behind a
   test that only ever looks at one index.
4. The runtime leg is UNCHANGED: still ``--force-recreate``, still
   ``--no-deps``. Convergence is right for a dependency and wrong for a service
   whose image this deploy just rebuilt.
5. A direct ``scope=core`` command still force-recreates. That one IS a request
   to act on the deps, and the fix is deliberately scoped to the side effect.
6. The deps argv now agrees with ``converge_deps``' own recovery argv, which
   has documented the same conclusion since OMN-18692 ("--force-recreate is
   deliberately ABSENT, because convergence must start what is missing and
   leave what is already running alone"). The two no longer contradict.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent.events import (
    SCOPE_SERVICES,
    BuildSource,
    EnumRuntimeLane,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor

pytestmark = pytest.mark.unit

#: postgres, redpanda, valkey -- the services a deps recreate destroys.
CORE_SERVICES: tuple[str, ...] = tuple(SCOPE_SERVICES[Scope.CORE])
BROKER = "redpanda"


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


def _ok(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=list(cmd), returncode=0, stdout="", stderr=""
    )


def _is_compose_up(cmd: Sequence[str]) -> bool:
    return "compose" in cmd and "up" in cmd


def _profile_of(cmd: Sequence[str]) -> str:
    tokens = list(cmd)
    return tokens[tokens.index("--profile") + 1] if "--profile" in tokens else ""


def _capture_full_rebuild(lane: EnumRuntimeLane, **kwargs: Any) -> list[list[str]]:
    """Drive the real ``rebuild_scope`` for ``Scope.FULL`` and return its argv.

    Stubs only the seams that reach the machine. ``_compose_up`` itself is
    LIVE, and so is every call site's decision about what to pass it -- which
    is the whole point: the defect was at the call site, so a test that stubbed
    ``_compose_up`` would have kept reporting green through it.
    """
    executor = DeployExecutor()
    captured: list[list[str]] = []

    def fake_run(
        cmd: list[str], timeout: int, **_: object
    ) -> subprocess.CompletedProcess[str]:
        captured.append(list(cmd))
        return _ok(cmd)

    with (
        patch("deploy_agent.executor._run", side_effect=fake_run),
        patch.object(DeployExecutor, "_compose_build", return_value=None),
        patch.object(
            DeployExecutor, "_build_dev_lane_only_services", return_value=None
        ),
        patch.object(DeployExecutor, "_deploy_gateway_lane", return_value=None),
        patch.object(
            DeployExecutor, "_ensure_runtime_migrations_ready", return_value=None
        ),
        patch.object(DeployExecutor, "_pull_pinned_image", return_value=None),
        patch.object(DeployExecutor, "_resolve_prod_image_env", return_value={}),
        patch("deploy_agent.executor.verify_containers_up", return_value=(True, [])),
        patch(
            "deploy_agent.executor.verify_oneshots_completed", return_value=(True, [])
        ),
    ):
        executor.rebuild_scope(
            Scope.FULL,
            [],
            _noop_phase_update,
            git_sha="0" * 40,
            git_ref="origin/dev",
            build_source=BuildSource.RELEASE,
            lane=lane,
            **kwargs,
        )
    return [c for c in captured if _is_compose_up(c)]


def _deps_leg(commands: Sequence[Sequence[str]]) -> list[str]:
    legs = [list(c) for c in commands if _profile_of(c) == "core"]
    assert len(legs) == 1, f"expected exactly one core-profile compose up, got {legs}"
    return legs[0]


class TestTheDepsLegOfAFullRebuild:
    """The leg that destroyed the broker nine times on 2026-09-18."""

    def test_a_dev_full_rebuild_does_not_force_recreate_the_deps(self) -> None:
        leg = _deps_leg(_capture_full_rebuild(EnumRuntimeLane.DEV))
        assert "--force-recreate" not in leg, (
            "the deps leg of a FULL rebuild force-recreates the lane's broker, "
            "dropping every consumer's group coordinator for a service whose "
            f"image this deploy never rebuilt: {leg}"
        )

    def test_a_prod_full_rebuild_does_not_force_recreate_the_deps(self) -> None:
        leg = _deps_leg(
            _capture_full_rebuild(
                EnumRuntimeLane.PROD, image_digest="sha256:" + "a" * 64
            )
        )
        assert "--force-recreate" not in leg, (
            "prod's FULL path is a SECOND call site; a promotion advances a "
            f"pinned runtime image and is not a request to replace its deps: {leg}"
        )

    def test_the_deps_leg_still_converges_the_whole_core_profile(self) -> None:
        """Dropping the flag must not narrow what the leg is responsible for."""
        leg = _deps_leg(_capture_full_rebuild(EnumRuntimeLane.DEV))
        assert _profile_of(leg) == "core"
        assert "up" in leg and "-d" in leg
        assert "--pull" in leg and leg[leg.index("--pull") + 1] == "always", (
            "convergence is what replaces the flag, so the image pull that "
            f"makes convergence able to notice a new image must survive: {leg}"
        )
        assert not set(leg) & set(CORE_SERVICES), (
            "core scope leaves service selection to the active profile; naming "
            f"them here would silently narrow the leg: {leg}"
        )

    @pytest.mark.parametrize("lane", [EnumRuntimeLane.DEV, EnumRuntimeLane.PROD])
    def test_no_command_a_full_rebuild_issues_force_recreates_the_broker(
        self, lane: EnumRuntimeLane
    ) -> None:
        """Stated over EVERY argv, so a new call site cannot slip past index 0."""
        kwargs: dict[str, Any] = (
            {"image_digest": "sha256:" + "a" * 64}
            if lane == EnumRuntimeLane.PROD
            else {}
        )
        offenders = [
            cmd
            for cmd in _capture_full_rebuild(lane, **kwargs)
            if "--force-recreate" in cmd
            and (_profile_of(cmd) == "core" or BROKER in cmd)
        ]
        assert not offenders, (
            "a FULL rebuild may not force-recreate any core-scope service on "
            f"lane {lane.value}; these commands do: {offenders}"
        )


class TestTheRuntimeLegIsUnchanged:
    """Convergence is right for a dependency and wrong for a rebuilt image."""

    def test_the_runtime_leg_still_force_recreates_and_still_passes_no_deps(
        self,
    ) -> None:
        runtime_legs = [
            cmd
            for cmd in _capture_full_rebuild(EnumRuntimeLane.DEV)
            if _profile_of(cmd) == "runtime" and "--no-deps" in cmd
        ]
        assert runtime_legs, "expected at least one runtime-profile compose up"
        family = runtime_legs[-1]
        assert "--force-recreate" in family, (
            "the runtime family's images were just rebuilt under the same tag, "
            "so convergence would re-adopt the stale containers -- this leg "
            f"must keep forcing: {family}"
        )
        assert "--no-deps" in family, (
            f"OMN-9455: the runtime leg must never walk depends_on: {family}"
        )


class TestADirectCoreCommandIsUntouched:
    """The fix removes a SIDE EFFECT, not an operator affordance."""

    def test_an_explicit_core_scope_compose_up_still_force_recreates(self) -> None:
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **_: object
        ) -> subprocess.CompletedProcess[str]:
            captured.append(list(cmd))
            return _ok(cmd)

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch(
                "deploy_agent.executor.verify_containers_up", return_value=(True, [])
            ),
        ):
            executor._compose_up(Phase.CORE, Scope.CORE, [], _noop_phase_update)

        assert "--force-recreate" in captured[0], (
            "a command whose scope IS core is a request to act on the deps; "
            f"only the FULL rebuild's deps LEG was changed: {captured[0]}"
        )


class TestTheAgentNoLongerContradictsItself:
    """``converge_deps`` reached this conclusion first; the two now agree."""

    def test_the_recovery_argv_and_the_deps_leg_agree_on_the_flag(self) -> None:
        executor = DeployExecutor()
        recovery = executor._deps_compose_argv(EnumRuntimeLane.DEV, list(CORE_SERVICES))
        leg = _deps_leg(_capture_full_rebuild(EnumRuntimeLane.DEV))
        assert "--force-recreate" not in recovery
        assert "--force-recreate" not in leg, (
            "the agent's recovery path converges while its deploy path "
            "recreates -- one of the two is wrong about the same lane"
        )
