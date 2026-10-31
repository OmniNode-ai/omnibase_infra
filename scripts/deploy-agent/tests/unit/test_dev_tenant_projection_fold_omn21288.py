# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev fold retires every folded role's container and cannot widen an empty restart scope."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from deploy_agent.compose_budget import _read_compose_services
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import (
    DEFAULT_DEV_INSTANCE,
    DeployExecutor,
    lane_config_for,
    lane_folded_profiles,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def _committed_fold_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read this checkout's fold overlay, not the deploy host's clone path."""
    monkeypatch.setattr(
        "deploy_agent.executor._DEV_TENANT_FOLD_OVERLAY",
        str(REPO_ROOT / "docker/docker-compose.dev-tenant-fold.yml"),
    )


RUNTIME_WRITERS = {
    "projection-writer-live-events": "projection-live-events-writer",
    "projection-writer-registration": "projection-registration-writer",
    "projection-writer-savings": "projection-savings-writer",
    "projection-writer-tenant-credentials": "projection-tenant-credentials-writer",
    "projection-writer-tenant-registry": "projection-tenant-registry-writer",
}


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
    assert config.disabled_services == frozenset({"tenant-projection-writer"})
    assert config.compose_files[-1].endswith("docker-compose.dev-tenant-fold.yml")
    assert "tenant-projection" in lane_folded_profiles(EnumRuntimeLane.DEV)
    monkeypatch.setattr(
        "deploy_agent.executor.active_dev_instance",
        lambda: f"{DEFAULT_DEV_INSTANCE}-other",
    )
    assert not lane_folded_profiles(EnumRuntimeLane.DEV)


def test_committed_policy_disables_every_folded_writer() -> None:
    """The committed policy folds the five in-process writers; each container is disabled."""
    policy = Path(__file__).resolve().parents[4] / "docker/runtime-policy.env"
    with patch("deploy_agent.executor.RUNTIME_POLICY_ENV_FILE", policy):
        folded = lane_folded_profiles(EnumRuntimeLane.DEV)
        config = lane_config_for(EnumRuntimeLane.DEV)
    assert set(RUNTIME_WRITERS) <= folded
    assert config.disabled_services == frozenset(
        {"tenant-projection-writer", *RUNTIME_WRITERS.values()}
    )
    # The two writers that still run their own consume loop keep their containers.
    assert "projection-delegation-writer" not in config.disabled_services
    assert "projection-claude-hook-events-writer" not in config.disabled_services


def test_folded_profile_without_a_gated_service_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deploy_agent.executor._load_runtime_policy_env",
        lambda: {"DEV_RUNTIME_MAIN_FOLDED_PROFILES": "workers"},
    )
    with pytest.raises(
        RuntimeError, match=r"gates no service under \['folded-workers'\]"
    ):
        lane_config_for(EnumRuntimeLane.DEV)


@pytest.mark.parametrize("failure", [None, "ps", "stop", "rm"])
def test_retirement_is_scoped_and_keeps_volumes(failure: str | None) -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        service = next(
            (
                arg.rsplit("=", 1)[1]
                for arg in cmd
                if arg.startswith("label=com.docker.compose.service=")
            ),
            "",
        )
        return subprocess.CompletedProcess(
            cmd,
            1 if cmd[1] == failure else 0,
            stdout=f"{service}-a\n{service}-b\n" if cmd[1] == "ps" else "",
            stderr="",
        )

    profiles = frozenset({"tenant-projection", "projection-writer-savings"})
    with (
        patch(
            "deploy_agent.executor.lane_folded_profiles",
            return_value=profiles,
        ),
        patch("deploy_agent.executor._run", side_effect=run),
    ):
        if failure:
            with pytest.raises(RuntimeError):
                DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.DEV)
        else:
            DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.DEV)
    assert all(
        "label=com.docker.compose.project=omnibase-infra" in c
        for c in calls
        if c[1] == "ps"
    )
    assert all("-v" not in c and "--volumes" not in c for c in calls)
    if failure in ("ps",):
        assert len(calls) == 1
        return
    censused = [c[-1] for c in calls if c[1] == "ps"]
    assert censused == [
        "label=com.docker.compose.service=projection-savings-writer",
        "label=com.docker.compose.service=tenant-projection-writer",
    ]
    if failure is None:
        ids = [
            "projection-savings-writer-a",
            "projection-savings-writer-b",
            "tenant-projection-writer-a",
            "tenant-projection-writer-b",
        ]
        assert calls[2] == ["docker", "stop", "--time", "30", *ids]
        assert calls[3] == ["docker", "rm", *ids]


def test_retirement_is_inert_on_other_lanes() -> None:
    with patch("deploy_agent.executor._run") as run:
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.STABILITY_TEST)
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
            stdout=f"{cmd[-1].rsplit('=', 1)[1]}-a\n" if cmd[1] == "ps" else "",
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
    retired = {"tenant-projection-writer", *RUNTIME_WRITERS.values()}
    assert [c[1] for c in calls] == [*["ps"] * len(retired), "stop", "rm", "compose"]
    assert "omninode-runtime" in calls[-1]
    for service in retired:
        assert service not in calls[-1]
        assert service not in verify.call_args.args[0]
        assert f"{service}-a" in calls[len(retired)]
