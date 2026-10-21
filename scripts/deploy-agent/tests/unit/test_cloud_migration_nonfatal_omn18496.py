# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18496: cloud migrations gate their consumers while runtime keeps serving."""

from __future__ import annotations

import re
import subprocess
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import events
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Phase, Scope
from deploy_agent.executor import DeployExecutor, DevLaneMigrationPreflightError

from .test_lab_overlay_build_order_omn18545 import (
    SHA,
    _dev_cmd,
    _FakeApplier,
    _make_agent,
)
from .test_lab_overlay_build_order_omn18545 import (
    _FakeExecutor as _BaseFakeExecutor,
)

pytestmark = pytest.mark.unit

FAILURE = (
    "Dev-lane migration preflight failed for cloud-migration: "
    "permission denied to create database"
)
DEPLOY_SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/handlers/handler_runtime_deploy.sh"
)


def _bash_array(name: str) -> list[str]:
    match = re.search(
        rf"^readonly\s+{re.escape(name)}=\((?P<body>.*?)^\)",
        DEPLOY_SCRIPT.read_text(encoding="utf-8"),
        re.M | re.S,
    )
    assert match is not None, f"{name} array not found in {DEPLOY_SCRIPT}"
    return [
        token
        for line in match.group("body").splitlines()
        for token in line.split("#", 1)[0].split()
    ]


@pytest.fixture
def failing_migration(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Exercise real preflight decisions without Docker or environment reads."""
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if "up" in cmd and cmd[-1] == "cloud-migration":
            return subprocess.CompletedProcess(
                cmd, 1, "", "permission denied to create database"
            )
        return subprocess.CompletedProcess(cmd, 0, "t\n" if "psql" in cmd else "", "")

    monkeypatch.setattr(executor_mod, "_run", run)
    monkeypatch.setattr(executor_mod, "_compose_env", lambda *args: {})
    monkeypatch.setattr(
        executor_mod, "verify_containers_up", Mock(return_value=(True, []))
    )
    monkeypatch.setattr(
        executor_mod, "verify_oneshots_completed", Mock(return_value=(True, []))
    )
    monkeypatch.setattr(
        executor_mod, "read_checkout_migration_fingerprint", lambda: "tree:runner"
    )
    return calls


def test_cloud_consumers_match_nonempty_bash_array() -> None:
    bash = _bash_array("DEV_LANE_CLOUD_DB_SERVICES")
    assert bash == ["onex-api"]  # Positive control: an empty parse cannot pass.
    assert tuple(bash) == events.DEV_LANE_CLOUD_DB_SERVICES


@pytest.mark.parametrize("failure_mode", ["recreate", "exit-zero"])
def test_cloud_failure_stops_consumer_and_preserves_applied_fingerprint(
    tmp_path: Path,
    failing_migration: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    if failure_mode == "exit-zero":
        run = executor_mod._run

        def successful_up(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            result = run(cmd, **kwargs)
            return subprocess.CompletedProcess(cmd, 0, result.stdout, "")

        monkeypatch.setattr(executor_mod, "_run", successful_up)
        monkeypatch.setattr(
            executor_mod,
            "verify_oneshots_completed",
            lambda services, **kwargs: (False, ["cloud-migration"])
            if services == ["cloud-migration"]
            else (True, []),
        )
    executor_mod.write_applied_migration_fingerprint(
        tmp_path, EnumRuntimeLane.DEV, "previous"
    )
    executor = DeployExecutor(migration_record_dir=tmp_path)
    executor._ensure_runtime_migrations_ready(lane=EnumRuntimeLane.DEV)
    assert "cloud-migration" in executor.cloud_migration_failure
    if failure_mode == "recreate":
        assert executor.cloud_migration_failure == FAILURE
    stops = [cmd for cmd in failing_migration if "stop" in cmd]
    assert len(stops) == 1
    assert stops[0] == [
        "docker",
        "compose",
        *executor_mod._compose_file_args(EnumRuntimeLane.DEV),
        "-p",
        executor_mod.lane_config_for(EnumRuntimeLane.DEV).compose_project,
        "--profile",
        "runtime",
        "stop",
        "--timeout",
        "30",
        "onex-api",
    ]
    assert (
        executor_mod.read_applied_migration_fingerprint(tmp_path, EnumRuntimeLane.DEV)
        == "previous"
    )


def test_failed_copier_never_launches_applier(
    failing_migration: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = executor_mod._run

    def fail_copier(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = run(cmd, **kwargs)
        if "up" in cmd and cmd[-1] == "cloud-migration-files":
            return subprocess.CompletedProcess(cmd, 1, "", "copy failed")
        return result

    monkeypatch.setattr(executor_mod, "_run", fail_copier)
    executor = DeployExecutor()
    executor._ensure_runtime_migrations_ready(lane=EnumRuntimeLane.DEV)
    assert "cloud-migration-files" in executor.cloud_migration_failure
    assert not any(
        "up" in cmd and cmd[-1] == "cloud-migration" for cmd in failing_migration
    )


def test_runtime_migration_failure_remains_fatal(
    failing_migration: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = executor_mod._run

    def fail_forward(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = run(cmd, **kwargs)
        if "up" in cmd and cmd[-1] == "forward-migration":
            return subprocess.CompletedProcess(cmd, 1, "", "DDL refused")
        return result

    monkeypatch.setattr(executor_mod, "_run", fail_forward)
    with pytest.raises(
        RuntimeError, match="Runtime migration preflight failed for forward-migration"
    ):
        DeployExecutor()._ensure_runtime_migrations_ready(lane=EnumRuntimeLane.DEV)


def _stub_runtime_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor_mod,
        "runtime_compose_up_budget",
        Mock(return_value=Mock(timeout_seconds=300)),
    )
    monkeypatch.setattr(DeployExecutor, "rerun_changed_oneshots", Mock())


def test_runtime_up_and_readback_exclude_cloud_consumer(
    failing_migration: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_runtime_budget(monkeypatch)
    DeployExecutor()._compose_up(
        Phase.RUNTIME, Scope.RUNTIME, [], lambda *args: None, lane=EnumRuntimeLane.DEV
    )
    runtime_up = [
        cmd for cmd in failing_migration if "up" in cmd and "omninode-runtime" in cmd
    ]
    assert len(runtime_up) == 1
    assert "onex-api" not in runtime_up[0]
    expected = executor_mod.verify_containers_up.call_args.args[0]
    assert "omninode-runtime" in expected
    assert "onex-api" not in expected


def test_cloud_consumer_only_request_refuses_empty_scope(
    failing_migration: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_runtime_budget(monkeypatch)
    with pytest.raises(
        DevLaneMigrationPreflightError, match="refusing an empty service scope"
    ):
        DeployExecutor()._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            ["onex-api"],
            lambda *args: None,
            lane=EnumRuntimeLane.DEV,
        )


def test_idle_converge_keeps_cloud_failure_fatal(
    failing_migration: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        executor_mod, "lane_lock", lambda *args, **kwargs: nullcontext()
    )
    with pytest.raises(DevLaneMigrationPreflightError, match="cloud-migration"):
        DeployExecutor().converge_forward_migration(lane=EnumRuntimeLane.DEV)


def test_observations_reset_cloud_failure() -> None:
    executor = DeployExecutor()
    assert executor.cloud_migration_failure == ""
    executor.cloud_migration_failure = FAILURE
    executor.reset_deploy_observations()
    assert executor.cloud_migration_failure == ""


class _FakeExecutor(_BaseFakeExecutor):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.cloud_migration_failure: str = ""
        self._failure = failure

    def rebuild_scope(self, *args: object, **kwargs: object) -> list[str]:
        services = super().rebuild_scope(*args, **kwargs)
        self.cloud_migration_failure = self._failure
        return services


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [FAILURE, ""], ids=["cloud-failure", "healthy-control"]
)
async def test_agent_verifies_runtime_before_cloud_failure_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    _FakeApplier.reset()
    monkeypatch.setattr(agent_mod, "LabOverlayApplier", _FakeApplier)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", True)
    cmd = _dev_cmd()
    executor = _FakeExecutor(failure)
    agent, store = _make_agent(tmp_path, monkeypatch, cmd, executor)
    try:
        agent._run_deploy(cmd)
        job = store.load(cmd.correlation_id)
        assert job is not None
        assert "rebuild_scope" in executor.calls
        assert "verify" in executor.calls
        assert executor.calls.index("rebuild_scope") < executor.calls.index("verify")
        assert job.status == ("failed" if failure else "success")
        assert [(entry["entry"], entry["sha"]) for entry in _FakeApplier.calls] == [
            ("repair" if failure else "apply", SHA)
        ]
        if failure:
            assert any(failure in error for error in job.errors)
    finally:
        agent._job_pool.shutdown(wait=True)
        agent._lag_pool.shutdown(wait=True)
        agent._settle_pool.shutdown(wait=True)


def test_failed_consumer_stop_is_recorded_without_aborting_runtime(
    failing_migration: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = executor_mod._run

    def fail_stop(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = run(cmd, **kwargs)
        if "stop" in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "daemon refused stop")
        return result

    monkeypatch.setattr(executor_mod, "_run", fail_stop)
    _stub_runtime_budget(monkeypatch)
    executor = DeployExecutor()
    executor._compose_up(
        Phase.RUNTIME, Scope.RUNTIME, [], lambda *args: None, lane=EnumRuntimeLane.DEV
    )
    assert (
        executor.cloud_migration_failure
        == FAILURE + "; stopping onex-api also failed: daemon refused stop"
    )
    assert any("up" in cmd and "omninode-runtime" in cmd for cmd in failing_migration)


def test_missing_runtime_projection_remains_fatal_after_cloud_failure(
    failing_migration: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = executor_mod._run

    def missing_projection(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = run(cmd, **kwargs)
        if "psql" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "f\n", "")
        return result

    monkeypatch.setattr(executor_mod, "_run", missing_projection)
    with pytest.raises(
        RuntimeError, match="Runtime migration preflight failed: missing"
    ):
        DeployExecutor()._ensure_runtime_migrations_ready(lane=EnumRuntimeLane.DEV)


def test_idle_converge_clears_previous_cloud_failure(
    failing_migration: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = executor_mod._run

    def successful_up(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = run(cmd, **kwargs)
        return subprocess.CompletedProcess(cmd, 0, result.stdout, "")

    monkeypatch.setattr(executor_mod, "_run", successful_up)
    monkeypatch.setattr(
        executor_mod, "lane_lock", lambda *args, **kwargs: nullcontext()
    )
    executor = DeployExecutor()
    executor.cloud_migration_failure = FAILURE
    assert (
        executor.converge_forward_migration(lane=EnumRuntimeLane.DEV) == "tree:runner"
    )
    assert executor.cloud_migration_failure == ""
    assert not any("stop" in cmd for cmd in failing_migration)


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
@pytest.mark.parametrize(
    "failure", [FAILURE, ""], ids=["cloud-failure", "healthy-control"]
)
def test_restarted_services_report_only_ungated_consumers(
    monkeypatch: pytest.MonkeyPatch,
    scope: Scope,
    failure: str,
) -> None:
    executor = DeployExecutor()
    monkeypatch.setattr(
        executor_mod, "_hold_build_context", lambda *args: nullcontext()
    )
    for method in (
        "_compose_build",
        "_build_dev_lane_only_services",
        "_deploy_gateway_lane",
    ):
        monkeypatch.setattr(executor, method, Mock())

    def compose_up(phase: Phase, *args: object, **kwargs: object) -> None:
        if phase == Phase.RUNTIME:
            executor.cloud_migration_failure = failure

    monkeypatch.setattr(executor, "_compose_up", compose_up)
    restarted = executor.rebuild_scope(
        scope, [], lambda *args: None, lane=EnumRuntimeLane.DEV
    )
    assert "omninode-runtime" in restarted
    assert ("onex-api" in restarted) == (not failure)
