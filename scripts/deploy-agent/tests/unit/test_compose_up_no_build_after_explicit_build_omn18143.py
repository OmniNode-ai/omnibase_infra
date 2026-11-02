# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A compose up that follows the explicit image builds must not build (OMN-18143).

Dev deploy job 42587fe2 (ref 5084c5ea) was killed by the 600s runtime-phase
compose-up ceiling; the next job, 1047c46b, spent 5 min 49 s inside the same
compose up before it created a single container. The agent had already built
every image one step earlier (``_compose_build`` for CORE and RUNTIME, then
``_build_dev_lane_only_services``), yet the up command omitted ``--no-build``,
so compose was free to rebuild the services whose ``build:`` section carries a
``service:`` additional context (``dev-lane-infra-runtime-image``) inside the
ceiling that only accounts for health-gated start periods.

These tests read the command line the agent builds. They use no wall clock.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import DeployExecutor

pytestmark = pytest.mark.unit

SHA = "a" * 40


def _noop(phase: Phase, status: PhaseStatus) -> None:
    return None


class _UpSeenError(Exception):
    """Raised by the fake runner once it has recorded the compose-up argv."""


def _up_command(scope: Scope, **kwargs: Any) -> list[str]:
    """Run the real ``_compose_up`` and return the ``docker compose ... up`` argv."""
    seen: list[list[str]] = []

    def _fake_run(cmd: list[str], timeout: int = 0, **_: Any) -> Any:
        if "up" in cmd:
            seen.append(list(cmd))
            raise _UpSeenError
        return subprocess.CompletedProcess(cmd, 0, "", "")

    budget = type(
        "Budget",
        (),
        {"timeout_seconds": 600, "describe": staticmethod(lambda: "test budget")},
    )
    executor = DeployExecutor()
    with (
        patch.object(executor_mod, "_run", _fake_run),
        patch.object(executor_mod, "_compose_env", lambda *a, **k: {}),
        patch.object(executor_mod, "finish_recreate_renames", lambda lane: []),
        patch.object(executor_mod, "preflight_dev_images", lambda *a, **k: None),
        patch.object(executor_mod, "runtime_compose_up_budget", lambda *a: budget),
        patch.object(executor, "_ensure_runtime_migrations_ready"),
        patch.object(executor, "rerun_changed_oneshots", return_value=[]),
        patch.object(executor, "_retire_folded_dev_tenant_projections"),
        patch.object(executor, "observe_deps_convergence"),
        patch.object(executor, "_record_compose_invocation"),
        pytest.raises(_UpSeenError),
    ):
        executor._compose_up(
            Phase.RUNTIME if scope is Scope.RUNTIME else Phase.CORE,
            scope,
            [],
            _noop,
            lane=EnumRuntimeLane.DEV,
            **kwargs,
        )
    (cmd,) = seen
    return cmd


class TestTheCommandLineTheAgentBuilds:
    def test_a_default_runtime_up_may_still_build(self) -> None:
        """The default is unchanged: a caller that did not build leaves compose free to."""
        assert "--no-build" not in _up_command(Scope.RUNTIME)

    def test_a_runtime_up_after_the_explicit_build_forbids_a_second_build(
        self,
    ) -> None:
        cmd = _up_command(Scope.RUNTIME, compose_may_build=False)
        assert "--no-build" in cmd
        # The flag belongs to `up`, before the service list and the pull policy.
        assert cmd.index("up") < cmd.index("--no-build")
        assert cmd[cmd.index("--pull") + 1] == "never"

    def test_forbidding_the_build_leaves_the_core_pull_policy_alone(self) -> None:
        """Only compose's build is forbidden; CORE still pulls its pinned images."""
        cmd = _up_command(Scope.CORE, compose_may_build=False, force_recreate=False)
        assert "--no-build" in cmd
        assert cmd[cmd.index("--pull") + 1] == "always"


class TestAFullRebuildBringsTheRuntimeUpWithoutBuilding:
    def test_the_runtime_phase_up_of_a_full_rebuild_forbids_a_second_build(
        self, tmp_path: Path
    ) -> None:
        executor = DeployExecutor()
        with (
            patch.object(executor, "_compose_build"),
            patch.object(executor, "_build_dev_lane_only_services"),
            patch.object(executor, "_preflight_runtime_boot"),
            patch.object(executor, "_deploy_gateway_lane"),
            patch.object(executor, "_compose_up") as compose_up,
            patch.object(
                executor_mod, "_reconcile_lock_roots", return_value=[str(tmp_path)]
            ),
        ):
            executor.rebuild_scope(
                Scope.FULL, [], _noop, git_sha=SHA, lane=EnumRuntimeLane.DEV
            )
        runtime_ups = [
            call
            for call in compose_up.call_args_list
            if call.args[:2] == (Phase.RUNTIME, Scope.RUNTIME)
        ]
        assert len(runtime_ups) == 1
        assert runtime_ups[0].kwargs.get("compose_may_build") is False


class TestASingleScopeRebuildBringsServicesUpWithoutBuilding:
    @pytest.mark.parametrize(
        ("scope", "phase", "services"),
        [
            (Scope.CORE, Phase.CORE, []),
            (Scope.CORE, Phase.CORE, ["postgres"]),
            (Scope.RUNTIME, Phase.RUNTIME, []),
            (Scope.RUNTIME, Phase.RUNTIME, ["omninode-runtime"]),
        ],
    )
    def test_single_scope_up_after_explicit_build_forbids_a_second_build(
        self, tmp_path: Path, scope: Scope, phase: Phase, services: list[str]
    ) -> None:
        executor = DeployExecutor()
        with (
            patch.object(executor, "_compose_build") as compose_build,
            patch.object(executor, "_build_dev_lane_only_services"),
            patch.object(executor, "_preflight_runtime_boot"),
            patch.object(executor, "_deploy_gateway_lane"),
            patch.object(executor, "_compose_up") as compose_up,
            patch.object(
                executor_mod, "_reconcile_lock_roots", return_value=[str(tmp_path)]
            ),
        ):
            executor.rebuild_scope(
                scope, services, _noop, git_sha=SHA, lane=EnumRuntimeLane.DEV
            )

        compose_build.assert_called_once()
        assert compose_build.call_args.args[:2] == (scope, SHA)
        compose_up.assert_called_once_with(
            phase,
            scope,
            services,
            _noop,
            lane=EnumRuntimeLane.DEV,
            compose_may_build=False,
        )
