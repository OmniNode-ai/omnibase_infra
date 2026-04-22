# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for Scope.RUNTIME compose behavior (OMN-9455).

The deploy-agent on .201 accepted a runtime-only rebuild on 2026-04-22 but ran
``docker compose ... --profile runtime up -d --force-recreate --pull always``
without ``--no-deps``. Compose recreated core infra dependencies (postgres,
redpanda, valkey, infisical, phoenix) and collided with the running
``omnibase-infra-infisical`` container, leaving the cluster partially broken.

These tests pin the runtime compose command shape and verification scope so the
regression cannot recur:

1. ``Scope.RUNTIME`` compose-up includes ``--no-deps``.
2. ``Scope.RUNTIME`` always emits an explicit service list (never a bare profile
   fan-out).
3. ``Scope.RUNTIME`` verification inspects only the services it just asked
   compose to touch.
4. ``Scope.CORE`` and ``Scope.FULL`` remain unchanged (no ``--no-deps``).
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from deploy_agent.events import (
    SCOPE_SERVICES,
    Phase,
    PhaseStatus,
    Scope,
    services_for_scope,
)
from deploy_agent.executor import DeployExecutor


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    pass


def _make_ok_result(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class TestRuntimeScopeComposeShape:
    """Runtime scope must use --no-deps and an explicit service list."""

    def test_runtime_scope_includes_no_deps_flag(self) -> None:
        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _make_ok_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            with patch(
                "deploy_agent.executor.verify_containers_up",
                return_value=(True, []),
            ):
                executor._compose_up(
                    Phase.RUNTIME,
                    Scope.RUNTIME,
                    ["omninode-runtime", "runtime-effects"],
                    _noop_phase_update,
                )

        assert captured_cmds, "expected at least one compose call"
        compose_cmd = captured_cmds[0]
        assert "--no-deps" in compose_cmd, (
            f"Scope.RUNTIME compose up must include --no-deps, got: {compose_cmd}"
        )

    def test_runtime_scope_targets_only_requested_services(self) -> None:
        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []
        requested = ["omninode-runtime", "runtime-effects"]

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _make_ok_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            with patch(
                "deploy_agent.executor.verify_containers_up",
                return_value=(True, []),
            ):
                executor._compose_up(
                    Phase.RUNTIME,
                    Scope.RUNTIME,
                    requested,
                    _noop_phase_update,
                )

        compose_cmd = captured_cmds[0]
        # The compose cmd must explicitly name the requested services after 'up'
        # and must NOT include any core services.
        for svc in requested:
            assert svc in compose_cmd, f"expected service {svc!r} in compose command"
        for core_svc in SCOPE_SERVICES[Scope.CORE]:
            assert core_svc not in compose_cmd, (
                f"runtime scope must not include core service {core_svc!r}; "
                f"compose cmd was: {compose_cmd}"
            )

    def test_runtime_scope_falls_back_to_canonical_runtime_services(self) -> None:
        """When no services are passed, runtime scope must still explicitly
        enumerate the canonical runtime service list instead of fanning out to
        the whole profile."""
        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _make_ok_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            with patch(
                "deploy_agent.executor.verify_containers_up",
                return_value=(True, []),
            ):
                executor._compose_up(
                    Phase.RUNTIME,
                    Scope.RUNTIME,
                    [],
                    _noop_phase_update,
                )

        compose_cmd = captured_cmds[0]
        assert "--no-deps" in compose_cmd
        for svc in services_for_scope(Scope.RUNTIME):
            assert svc in compose_cmd, (
                f"runtime fallback must include canonical service {svc!r}, "
                f"got {compose_cmd}"
            )
        for core_svc in SCOPE_SERVICES[Scope.CORE]:
            assert core_svc not in compose_cmd


class TestRuntimeScopeVerification:
    """Runtime verification must only wait on the services it actually touched."""

    def test_runtime_verify_ignores_core_services(self) -> None:
        executor = DeployExecutor()
        requested = ["omninode-runtime", "runtime-effects"]
        verify_calls: list[list[str]] = []

        def fake_verify(
            expected: list[str], timeout_s: int = 120
        ) -> tuple[bool, list[str]]:
            verify_calls.append(list(expected))
            return True, []

        with patch("deploy_agent.executor._run", return_value=_make_ok_result()):
            with patch(
                "deploy_agent.executor.verify_containers_up", side_effect=fake_verify
            ):
                executor._compose_up(
                    Phase.RUNTIME,
                    Scope.RUNTIME,
                    requested,
                    _noop_phase_update,
                )

        assert verify_calls, "expected verify_containers_up to be called"
        verified = verify_calls[0]
        assert verified == requested, (
            f"runtime verify must target exactly the requested services; "
            f"got {verified}, expected {requested}"
        )
        for core_svc in SCOPE_SERVICES[Scope.CORE]:
            assert core_svc not in verified

    def test_runtime_verify_uses_canonical_list_when_no_services_passed(self) -> None:
        executor = DeployExecutor()
        verify_calls: list[list[str]] = []

        def fake_verify(
            expected: list[str], timeout_s: int = 120
        ) -> tuple[bool, list[str]]:
            verify_calls.append(list(expected))
            return True, []

        with patch("deploy_agent.executor._run", return_value=_make_ok_result()):
            with patch(
                "deploy_agent.executor.verify_containers_up", side_effect=fake_verify
            ):
                executor._compose_up(
                    Phase.RUNTIME,
                    Scope.RUNTIME,
                    [],
                    _noop_phase_update,
                )

        assert verify_calls
        verified = verify_calls[0]
        assert verified == services_for_scope(Scope.RUNTIME)
        for core_svc in SCOPE_SERVICES[Scope.CORE]:
            assert core_svc not in verified


class TestNonRuntimeScopesUnchanged:
    """Scope.CORE and Scope.FULL preserve pre-OMN-9455 compose-up semantics."""

    def test_core_scope_does_not_use_no_deps(self) -> None:
        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _make_ok_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            with patch(
                "deploy_agent.executor.verify_containers_up",
                return_value=(True, []),
            ):
                executor._compose_up(
                    Phase.CORE,
                    Scope.CORE,
                    [],
                    _noop_phase_update,
                )

        compose_cmd = captured_cmds[0]
        assert "--no-deps" not in compose_cmd, (
            f"Scope.CORE must not carry --no-deps; got: {compose_cmd}"
        )
        # Profile-wide fan-out is preserved: no explicit service list needed.
        # We only assert that core services are not filtered OUT. The absence of
        # a trailing service list is the existing semantic.

    def test_core_scope_verify_uses_scope_service_list(self) -> None:
        executor = DeployExecutor()
        verify_calls: list[list[str]] = []

        def fake_verify(
            expected: list[str], timeout_s: int = 120
        ) -> tuple[bool, list[str]]:
            verify_calls.append(list(expected))
            return True, []

        with patch("deploy_agent.executor._run", return_value=_make_ok_result()):
            with patch(
                "deploy_agent.executor.verify_containers_up", side_effect=fake_verify
            ):
                executor._compose_up(
                    Phase.CORE,
                    Scope.CORE,
                    [],
                    _noop_phase_update,
                )

        assert verify_calls
        assert verify_calls[0] == services_for_scope(Scope.CORE)
