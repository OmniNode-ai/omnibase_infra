# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-scope compose-up must not recreate core infra (OMN-9455).

Regression tests for the 2026-04-22 incident where a runtime rebuild ran
``docker compose ... --profile runtime up -d --force-recreate --pull always``
without ``--no-deps`` and without an explicit service list. Compose walked the
``depends_on`` graph and collided with the live ``omnibase-infra-infisical``
container, breaking Redpanda/Postgres/Valkey/Phoenix.

These tests pin:

1. ``Scope.RUNTIME`` includes ``--no-deps`` on the compose-up command.
2. ``Scope.RUNTIME`` targets the runtime service list explicitly even when the
   caller passed no services.
3. Runtime verification polls only the runtime service set (never core infra).
4. Explicit runtime service subsets are honored verbatim and verification is
   bounded by that subset.
5. ``Scope.CORE`` and ``Scope.FULL`` behavior is unchanged (no ``--no-deps`` and
   no forced runtime service list).
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
from deploy_agent.events import Phase, PhaseStatus, Scope, services_for_scope
from deploy_agent.executor import (
    DeployExecutor,
    _requested_services_for_up,
)


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


def _ok() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


@pytest.mark.unit
class TestRequestedServicesForUp:
    """The helper deciding which services compose-up should target per-scope."""

    def test_runtime_scope_with_empty_services_returns_runtime_service_list(
        self,
    ) -> None:
        assert _requested_services_for_up(Scope.RUNTIME, []) == services_for_scope(
            Scope.RUNTIME
        )

    def test_runtime_scope_with_explicit_services_returns_those_services(self) -> None:
        requested = ["omninode-runtime", "runtime-effects"]
        assert _requested_services_for_up(Scope.RUNTIME, requested) == requested

    def test_core_scope_with_empty_services_returns_empty_list(self) -> None:
        # Core behavior unchanged: compose infers services from the active profile.
        assert _requested_services_for_up(Scope.CORE, []) == []

    def test_core_scope_with_explicit_services_returns_those_services(self) -> None:
        requested = ["postgres"]
        assert _requested_services_for_up(Scope.CORE, requested) == requested

    def test_full_scope_with_empty_services_returns_empty_list(self) -> None:
        # Full-scope behavior unchanged: compose brings up both profiles.
        assert _requested_services_for_up(Scope.FULL, []) == []


@pytest.mark.unit
class TestRuntimeScopeComposeUp:
    """Regression tests for the OMN-9455 runtime rebuild incident."""

    def test_runtime_scope_uses_no_deps_and_explicit_runtime_service_list(
        self,
    ) -> None:
        """Runtime rebuild with no explicit services must still target runtime
        services directly and pass --no-deps so compose cannot walk depends_on
        into core infra."""
        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch(
                "deploy_agent.executor.verify_containers_up",
                return_value=(True, []),
            ) as mock_verify,
        ):
            executor._compose_up(
                Phase.RUNTIME,
                Scope.RUNTIME,
                [],
                _noop_phase_update,
            )

        assert captured_cmds, "Expected a docker compose up invocation"
        compose_cmd = captured_cmds[0]
        runtime_services = services_for_scope(Scope.RUNTIME)

        assert "--no-deps" in compose_cmd, (
            "Runtime scope compose up must include --no-deps to prevent "
            "recreating core infra dependencies (OMN-9455)"
        )
        # Core infra must never appear as a targeted service on a runtime rebuild.
        for core_service in ("postgres", "redpanda", "valkey", "infisical"):
            assert core_service not in compose_cmd, (
                f"Core infra service {core_service!r} must not appear in a "
                "runtime-scope compose up command (OMN-9455)"
            )
        # The tail of the command must be exactly the runtime service list.
        assert compose_cmd[-len(runtime_services) :] == runtime_services, (
            "Runtime scope compose up must target the explicit runtime service "
            f"list; got {compose_cmd}"
        )
        # Verification must be scoped to the requested runtime services.
        mock_verify.assert_called_once_with(runtime_services, timeout_s=120)

    def test_runtime_scope_subset_preserves_requested_services_only(self) -> None:
        """An explicit runtime subset must be targeted verbatim; verification
        must only wait for that subset."""
        executor = DeployExecutor()
        requested = ["omninode-runtime", "runtime-effects"]
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch(
                "deploy_agent.executor.verify_containers_up",
                return_value=(True, []),
            ) as mock_verify,
        ):
            executor._compose_up(
                Phase.RUNTIME,
                Scope.RUNTIME,
                requested,
                _noop_phase_update,
            )

        compose_cmd = captured_cmds[0]

        assert "--no-deps" in compose_cmd
        assert compose_cmd[-len(requested) :] == requested
        mock_verify.assert_called_once_with(requested, timeout_s=120)


@pytest.mark.unit
class TestCoreAndFullScopeComposeUp:
    """Core and full-scope compose-up behavior must remain unchanged."""

    def test_core_scope_does_not_use_no_deps(self) -> None:
        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch(
                "deploy_agent.executor.verify_containers_up",
                return_value=(True, []),
            ) as mock_verify,
        ):
            executor._compose_up(
                Phase.CORE,
                Scope.CORE,
                [],
                _noop_phase_update,
            )

        compose_cmd = captured_cmds[0]
        core_services = services_for_scope(Scope.CORE)

        assert "--no-deps" not in compose_cmd, (
            "Core scope compose up must NOT include --no-deps (OMN-9455 is "
            "scoped to runtime only)"
        )
        # Core scope with empty services must leave service selection to the
        # active profile — no explicit services should be appended.
        assert "--profile" in compose_cmd
        profile_idx = compose_cmd.index("--profile")
        assert compose_cmd[profile_idx + 1] == "core"
        # Last positional tokens must be the compose flags, not service names.
        assert compose_cmd[-1] == "always", (
            "Core scope with empty services must not append explicit service "
            f"names to compose up; got {compose_cmd}"
        )
        # Verification uses the scope default list when no explicit services
        # were requested.
        mock_verify.assert_called_once_with(core_services, timeout_s=120)

    def test_core_scope_honors_explicit_service_subset(self) -> None:
        executor = DeployExecutor()
        requested = ["postgres"]
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch(
                "deploy_agent.executor.verify_containers_up",
                return_value=(True, []),
            ) as mock_verify,
        ):
            executor._compose_up(
                Phase.CORE,
                Scope.CORE,
                requested,
                _noop_phase_update,
            )

        compose_cmd = captured_cmds[0]

        assert "--no-deps" not in compose_cmd
        assert compose_cmd[-len(requested) :] == requested
        mock_verify.assert_called_once_with(requested, timeout_s=120)
