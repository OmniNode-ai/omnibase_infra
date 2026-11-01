# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev fold retires tenant-projection and cannot widen an empty restart scope."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from deploy_agent.compose_budget import _read_compose_services
from deploy_agent.events import (
    DEV_LANE_GATEWAY_SERVICES,
    EnumRuntimeLane,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import (
    DEFAULT_DEV_INSTANCE,
    DeployExecutor,
    lane_config_for,
    lane_folded_profiles,
)

pytestmark = pytest.mark.unit


def test_primary_dev_disables_the_tenant_carrier_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deploy_agent.executor._load_runtime_policy_env",
        lambda: {
            "DEV_RUNTIME_MAIN_FOLDED_PROFILES": "tenant-projection",
        },
    )
    config = lane_config_for(EnumRuntimeLane.DEV)
    # The gateway project's services are off too: the shipped table declares
    # gateway-deploy off for dev-201 (OMN-21629).
    assert config.disabled_services == frozenset(
        {"tenant-projection-writer", *DEV_LANE_GATEWAY_SERVICES}
    )
    assert config.compose_files[-1].endswith("docker-compose.dev-tenant-fold.yml")
    assert "tenant-projection" in lane_folded_profiles(EnumRuntimeLane.DEV)
    monkeypatch.setattr(
        "deploy_agent.executor.active_dev_instance",
        lambda: f"{DEFAULT_DEV_INSTANCE}-other",
    )
    assert not lane_folded_profiles(EnumRuntimeLane.DEV)


@pytest.mark.parametrize("failure", [None, "ps", "stop", "rm"])
def test_retirement_is_scoped_and_keeps_volumes(failure: str | None) -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            1 if cmd[1] == failure else 0,
            stdout="tenant-projection-writer-a\ntenant-projection-writer-b\n"
            if cmd[1] == "ps"
            else "",
            stderr="",
        )

    with (
        patch(
            "deploy_agent.executor.lane_folded_profiles",
            return_value={"tenant-projection"},
        ),
        patch("deploy_agent.executor._run", side_effect=run),
    ):
        if failure:
            with pytest.raises(RuntimeError):
                DeployExecutor()._retire_folded_dev_tenant_projections(
                    EnumRuntimeLane.DEV
                )
        else:
            DeployExecutor()._retire_folded_dev_tenant_projections(EnumRuntimeLane.DEV)
    assert "label=com.docker.compose.service=tenant-projection-writer" in calls[0]
    assert "label=com.docker.compose.project=omnibase-infra" in calls[0]
    assert all("-v" not in c and "--volumes" not in c for c in calls)
    if failure is None:
        assert calls[1] == [
            "docker",
            "stop",
            "--time",
            "30",
            "tenant-projection-writer-a",
            "tenant-projection-writer-b",
        ]
        assert calls[2] == [
            "docker",
            "rm",
            "tenant-projection-writer-a",
            "tenant-projection-writer-b",
        ]


def test_retirement_is_inert_on_other_lanes() -> None:
    with patch("deploy_agent.executor._run") as run:
        DeployExecutor()._retire_folded_dev_tenant_projections(
            EnumRuntimeLane.STABILITY_TEST
        )
    run.assert_not_called()


def test_only_folded_tenant_carrier_request_does_not_start_the_whole_profile() -> None:
    updates: list[tuple[Phase, PhaseStatus]] = []
    with (
        patch("deploy_agent.executor._requested_services_for_up", return_value=[]),
        patch("deploy_agent.executor._run") as run,
    ):
        DeployExecutor()._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            ["tenant-projection-writer"],
            lambda phase, status: updates.append((phase, status)),
            lane=EnumRuntimeLane.DEV,
        )
    run.assert_not_called()
    assert updates[-1] == (Phase.RUNTIME, PhaseStatus.SUCCESS)


def test_budget_reader_honors_service_reset(tmp_path) -> None:
    base = tmp_path / "base.yml"
    overlay = tmp_path / "fold.yml"
    base.write_text(
        "services:\n  main:\n    image: example/main\n  tenant-projection-writer:\n    image: example/tenant-projection-writer\n"
    )
    overlay.write_text("services:\n  tenant-projection-writer: !reset null\n")
    assert set(_read_compose_services([str(base), str(overlay)])) == {"main"}


def test_real_folded_deploy_retires_tenant_carrier_before_starting_main() -> None:
    calls: list[list[str]] = []

    def run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="tenant-projection-writer-a\n" if cmd[1] == "ps" else "",
            stderr="",
        )

    executor = DeployExecutor()
    policy = Path(__file__).resolve().parents[4] / "docker/runtime-policy.env"
    with (
        patch("deploy_agent.executor.RUNTIME_POLICY_ENV_FILE", policy),
        patch("deploy_agent.executor._run", side_effect=run),
        patch(
            "deploy_agent.executor.runtime_compose_up_budget",
            return_value=SimpleNamespace(timeout_seconds=120, describe=lambda: "test"),
        ),
        patch("deploy_agent.executor.finish_recreate_renames"),
        patch(
            "deploy_agent.executor.verify_containers_up", return_value=(True, [])
        ) as verify,
        patch.object(executor, "_ensure_runtime_migrations_ready"),
        patch.object(executor, "rerun_changed_oneshots"),
    ):
        executor._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            [],
            lambda *args: None,
            lane=EnumRuntimeLane.DEV,
        )
    assert [c[1] for c in calls] == ["ps", "stop", "rm", "compose"]
    assert "omninode-runtime" in calls[-1]
    assert "tenant-projection-writer" not in calls[-1]
    assert "tenant-projection-writer" not in verify.call_args.args[0]
