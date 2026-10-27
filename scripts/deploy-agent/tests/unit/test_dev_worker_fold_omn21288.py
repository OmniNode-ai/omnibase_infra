# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev fold retires workers and cannot widen an empty restart scope."""

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


def test_primary_dev_disables_the_worker_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deploy_agent.executor._load_runtime_policy_env",
        lambda: {
            "DEV_RUNTIME_MAIN_FOLDED_PROFILES": "workers",
        },
    )
    config = lane_config_for(EnumRuntimeLane.DEV)
    assert config.disabled_services == frozenset({"runtime-worker"})
    assert config.compose_files[-2].endswith("docker-compose.dev-fold.yml")
    assert "workers" in lane_folded_profiles(EnumRuntimeLane.DEV)
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
            stdout="worker-a\nworker-b\n" if cmd[1] == "ps" else "",
            stderr="",
        )

    with (
        patch("deploy_agent.executor.lane_folded_profiles", return_value={"workers"}),
        patch("deploy_agent.executor._run", side_effect=run),
    ):
        executor = DeployExecutor()
        if failure:
            with pytest.raises(RuntimeError):
                with executor.folded_dev_worker_handoff(EnumRuntimeLane.DEV):
                    executor._retire_folded_dev_workers(EnumRuntimeLane.DEV)
        else:
            with executor.folded_dev_worker_handoff(EnumRuntimeLane.DEV):
                executor._retire_folded_dev_workers(EnumRuntimeLane.DEV)
                assert all(c[1] != "rm" for c in calls)
    assert "label=com.docker.compose.service=runtime-worker" in calls[0]
    assert "label=com.docker.compose.project=omnibase-infra" in calls[0]
    assert all("-v" not in c and "--volumes" not in c for c in calls)
    if failure is None:
        assert calls[1] == ["docker", "stop", "--time", "30", "worker-a", "worker-b"]
        assert calls[2] == ["docker", "rm", "worker-a", "worker-b"]


def test_retirement_is_inert_on_other_lanes() -> None:
    with patch("deploy_agent.executor._run") as run:
        DeployExecutor()._retire_folded_dev_workers(EnumRuntimeLane.STABILITY_TEST)
    run.assert_not_called()


def test_only_folded_worker_request_does_not_start_the_whole_profile() -> None:
    updates: list[tuple[Phase, PhaseStatus]] = []
    with (
        patch("deploy_agent.executor.lane_folded_profiles", return_value={"workers"}),
        patch("deploy_agent.executor._run") as run,
    ):
        DeployExecutor()._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            ["runtime-worker"],
            lambda phase, status: updates.append((phase, status)),
            lane=EnumRuntimeLane.DEV,
        )
    run.assert_not_called()
    assert updates[-1] == (Phase.RUNTIME, PhaseStatus.SUCCESS)


def test_budget_reader_honors_service_reset(tmp_path) -> None:
    base = tmp_path / "base.yml"
    overlay = tmp_path / "fold.yml"
    base.write_text(
        "services:\n  main:\n    image: example/main\n  runtime-worker:\n    image: example/worker\n"
    )
    overlay.write_text("services:\n  runtime-worker: !reset null\n")
    assert set(_read_compose_services([str(base), str(overlay)])) == {"main"}


def test_real_folded_deploy_retires_worker_before_starting_main() -> None:
    calls: list[list[str]] = []

    def run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="worker-a\n"
            if cmd[1] == "ps"
            and "label=com.docker.compose.service=runtime-worker" in cmd
            else "",
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
        with executor.folded_dev_worker_handoff(EnumRuntimeLane.DEV):
            executor._compose_up(
                Phase.RUNTIME,
                Scope.RUNTIME,
                [],
                lambda *args: None,
                lane=EnumRuntimeLane.DEV,
            )
            assert [c[1] for c in calls] == ["ps", "stop", "ps", "compose"]
    assert [c[1] for c in calls] == ["ps", "stop", "ps", "compose", "rm"]
    assert "omninode-runtime" in calls[-2]
    assert "runtime-worker" not in calls[-2]
    assert "runtime-worker" not in verify.call_args.args[0]


@pytest.mark.parametrize("folded", [frozenset(), frozenset({"workers"})])
def test_other_empty_filtered_requests_are_refused(folded: frozenset[str]) -> None:
    with (
        patch("deploy_agent.executor.lane_folded_profiles", return_value=folded),
        patch("deploy_agent.executor._requested_services_for_up", return_value=[]),
        patch("deploy_agent.executor._run") as run,
        pytest.raises(RuntimeError, match="Every requested compose service"),
    ):
        DeployExecutor()._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            ["runtime-effects"],
            lambda *args: None,
        )
    run.assert_not_called()


def test_failed_handoff_stops_candidate_before_restoring_worker() -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        output = ""
        if cmd[1] == "ps":
            output = (
                "main-id"
                if "label=com.docker.compose.service=omninode-runtime" in cmd
                else "worker-id"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    executor = DeployExecutor()
    with (
        patch("deploy_agent.executor.lane_folded_profiles", return_value={"workers"}),
        patch("deploy_agent.executor._run", side_effect=run),
        pytest.raises(RuntimeError, match="unhealthy candidate"),
    ):
        with executor.folded_dev_worker_handoff(EnumRuntimeLane.DEV):
            executor._retire_folded_dev_workers(EnumRuntimeLane.DEV)
            raise RuntimeError("unhealthy candidate")
    assert [cmd[1] for cmd in calls] == ["ps", "stop", "ps", "stop", "start"]
    assert calls[-2][-1] == "main-id"
    assert calls[-1][-1] == "worker-id"
    assert not executor._folded_worker_containers


def test_failed_candidate_stop_does_not_start_a_second_worker_owner() -> None:
    executor = DeployExecutor()
    executor._folded_worker_containers = ["worker-id"]
    with (
        patch(
            "deploy_agent.executor._run",
            side_effect=[
                subprocess.CompletedProcess([], 0, stdout="main-id", stderr=""),
                subprocess.CompletedProcess([], 1, stdout="", stderr="stop failed"),
            ],
        ) as run,
        pytest.raises(RuntimeError, match="Cannot stop candidate main"),
    ):
        with executor.folded_dev_worker_handoff(EnumRuntimeLane.DEV):
            raise RuntimeError("unhealthy candidate")
    assert all(call.args[0][1] != "start" for call in run.call_args_list)
    assert executor._folded_worker_containers == ["worker-id"]


def test_rebuild_keeps_worker_until_verification_returns() -> None:
    executor = DeployExecutor()
    calls: list[list[str]] = []

    def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="worker-id" if cmd[1] == "ps" else "", stderr=""
        )

    def rebuild(*args: object, **kwargs: object) -> list[str]:
        executor._retire_folded_dev_workers(EnumRuntimeLane.DEV)
        return ["omninode-runtime"]

    with (
        patch("deploy_agent.executor.lane_folded_profiles", return_value={"workers"}),
        patch("deploy_agent.executor._run", side_effect=run),
        patch.object(executor, "_rebuild_scope", side_effect=rebuild),
        patch.object(executor, "_verify", return_value=[]),
    ):
        assert executor.rebuild_scope(Scope.RUNTIME, [], lambda *args: None) == [
            "omninode-runtime"
        ]
        assert [cmd[1] for cmd in calls] == ["ps", "stop"]
        executor.verify(lambda *args: None)
        assert [cmd[1] for cmd in calls] == ["ps", "stop", "rm"]


@pytest.mark.parametrize("method", ["rebuild", "verify"])
def test_public_failure_path_restores_retained_worker(method: str) -> None:
    executor = DeployExecutor()
    executor._folded_worker_containers = ["worker-id"]
    executor._folded_worker_project = "owned-test-project"
    with (
        patch(
            "deploy_agent.executor._run",
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ) as run,
        patch.object(
            executor, "_rebuild_scope", side_effect=RuntimeError("candidate refused")
        ),
        patch.object(
            executor, "_verify", side_effect=RuntimeError("candidate refused")
        ),
        pytest.raises(RuntimeError, match="candidate refused"),
    ):
        if method == "rebuild":
            executor.rebuild_scope(Scope.RUNTIME, [], lambda *args: None)
        else:
            executor.verify(lambda *args: None)
    assert (
        "label=com.docker.compose.project=owned-test-project"
        in run.call_args_list[0].args[0]
    )
    assert run.call_args_list[-1].args[0] == ["docker", "start", "worker-id"]


def test_primary_dev_keeps_both_folds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deploy_agent.executor._load_runtime_policy_env",
        lambda: {"DEV_RUNTIME_MAIN_FOLDED_PROFILES": "workers,tenant-projection"},
    )
    config = lane_config_for(EnumRuntimeLane.DEV)
    assert {"runtime-worker", "tenant-projection-writer"} <= config.disabled_services
    assert [Path(f).name for f in config.compose_files][-2:] == [
        "docker-compose.dev-fold.yml",
        "docker-compose.dev-tenant-fold.yml",
    ]
