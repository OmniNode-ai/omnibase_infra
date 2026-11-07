# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed roll recovers stopped containers without advancing their images."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import DeployExecutor


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["running", "health", "timeout"])
def test_runtime_roll_recovery_preserves_failure_and_peer_images(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    services = ["runtime-effects", "runtime-worker"]
    executor = DeployExecutor()
    phases: list[tuple[Phase, PhaseStatus]] = []
    success = subprocess.CompletedProcess([], 0, "", "")
    roll = Mock(return_value=success)
    if failure == "health":
        roll.return_value = subprocess.CompletedProcess([], 1, "", "unhealthy")
    elif failure == "timeout":
        roll.side_effect = subprocess.TimeoutExpired("compose up", 900)
    recovery = Mock(return_value=success)
    monkeypatch.setattr(executor_mod, "_run", roll)
    monkeypatch.setattr(executor_mod.subprocess, "run", recovery)
    monkeypatch.setattr(executor_mod, "_compose_env", lambda *a, **k: {})
    monkeypatch.setattr(
        executor_mod,
        "runtime_compose_up_budget",
        lambda *a: SimpleNamespace(timeout_seconds=600, describe=lambda: "test"),
    )
    monkeypatch.setattr(executor_mod, "runtime_health_probe_budget", lambda *a: 900)
    monkeypatch.setattr(executor, "_ensure_runtime_migrations_ready", Mock())
    monkeypatch.setattr(executor, "rerun_changed_oneshots", Mock(return_value=[]))
    monkeypatch.setattr(
        executor_mod,
        "_compose_service_states",
        lambda *a: dict.fromkeys(services, ("created", None)),
    )
    # The first step fails, then the project census sees both stopped services.
    # Starting them restores running state, but cannot finish the partial roll.
    verifications = [(False, services), (True, [])]
    if failure == "running":
        verifications.insert(0, (False, services[:1]))
    monkeypatch.setattr(
        executor_mod, "verify_containers_up", Mock(side_effect=verifications)
    )
    convergence = Mock()
    monkeypatch.setattr(executor, "_require_compose_up_converged", convergence)
    error_type = executor_mod.load_dev_boot_guard().DevBootFailedError

    with pytest.raises(error_type, match="Runtime roll stopped at runtime-effects"):
        executor._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            services,
            lambda phase, status: phases.append((phase, status)),
            lane=EnumRuntimeLane.DEV,
        )

    assert roll.call_count == 1
    assert roll.call_args.args[0][-1] == "runtime-effects"
    assert recovery.call_count == 2
    assert [call.args[0][-1] for call in recovery.call_args_list] == services
    for call in recovery.call_args_list:
        command = call.args[0]
        assert "--no-recreate" in command
        assert "--no-build" in command
        assert command[command.index("--pull") + 1] == "never"
        assert "--no-deps" in command
    assert {item.service for item in executor.container_residue} == set(services)
    assert all(item.recovered for item in executor.container_residue)
    convergence.assert_not_called()
    assert (Phase.RUNTIME, PhaseStatus.SUCCESS) not in phases
