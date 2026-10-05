# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Automatic image rollback preserves a failed deploy's verdict."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelHealthCheck,
    ModelRebuildRequested,
    Phase,
    Scope,
)
from deploy_agent.executor import DeployExecutor, VerificationFailedError
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit
_IMAGE_ID = "sha256:" + "a" * 64
_OTHER_ID = "sha256:" + "b" * 64


def _check(service: str, *, healthy: bool = True) -> ModelHealthCheck:
    return ModelHealthCheck(
        service=service,
        endpoint="http://localhost/health",
        status="pass" if healthy else "fail",
        detail="" if healthy else "degraded",
    )


def _point(
    lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
) -> executor_mod.ModelRollbackPoint:
    targets = executor_mod.lane_config_for(lane).runtime_health_targets
    return executor_mod.ModelRollbackPoint(
        lane=lane,
        captured_at=datetime.now(UTC),
        images=tuple(
            executor_mod.ModelRetainedImage(
                container=container,
                service=service,
                image_ref="runtime:latest",
                image_id=_IMAGE_ID,
            )
            for (container, _), service in zip(
                targets, ("omninode-runtime", "runtime-effects"), strict=True
            )
        ),
    )


@pytest.fixture
def command_runner(monkeypatch: pytest.MonkeyPatch) -> Mock:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = (
            (_IMAGE_ID if argv[3] == "{{.Image}}" else "runtime:latest")
            if argv[:2] == ["docker", "inspect"]
            else ""
        )
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    runner = Mock(side_effect=run)
    monkeypatch.setattr(executor_mod, "_run", runner)
    return runner


@pytest.mark.parametrize("lane", [EnumRuntimeLane.DEV, EnumRuntimeLane.STABILITY_TEST])
def test_capture_probes_targets_and_retains_image(
    monkeypatch: pytest.MonkeyPatch, command_runner: Mock, lane: EnumRuntimeLane
) -> None:
    executor = DeployExecutor()
    probe = Mock(side_effect=lambda **kw: (_check(kw["service"]), None))
    monkeypatch.setattr(executor, "_probe_runtime_health", probe)
    before = datetime.now(UTC)
    point = executor_mod.capture_rollback_point(executor, lane)
    assert point is not None
    assert before <= point.captured_at <= datetime.now(UTC)
    assert point.lane == lane
    assert point.images == _point(lane).images
    for container, port in executor_mod.lane_config_for(lane).runtime_health_targets:
        probe.assert_any_call(service=container, port=port)
        assert any(
            call.args[0]
            == ["docker", "inspect", "--format", "{{.Config.Image}}", container]
            for call in command_runner.call_args_list
        )
    assert any(
        call.args[0] == ["docker", "tag", _IMAGE_ID, f"runtime:rollback-{lane.value}"]
        for call in command_runner.call_args_list
    )
    with pytest.raises(ValueError, match="frozen"):
        point.lane = EnumRuntimeLane.PROD


@pytest.mark.parametrize("bad_target", [0, 1])
def test_capture_unhealthy_lane_has_no_point(
    monkeypatch: pytest.MonkeyPatch,
    command_runner: Mock,
    caplog: pytest.LogCaptureFixture,
    bad_target: int,
) -> None:
    executor = DeployExecutor()
    checks = [
        (_check("target", healthy=index != bad_target), None) for index in range(2)
    ]
    monkeypatch.setattr(executor, "_probe_runtime_health", Mock(side_effect=checks))
    assert executor_mod.capture_rollback_point(executor, EnumRuntimeLane.DEV) is None
    assert not any(
        call.args[0][:2] == ["docker", "tag"] for call in command_runner.call_args_list
    )
    assert "degraded" in caplog.text


@pytest.mark.parametrize("failure", ["id", "ref", "retain", "timeout"])
def test_capture_unreadable_or_unretained_image_has_no_point(
    monkeypatch: pytest.MonkeyPatch, command_runner: Mock, failure: str
) -> None:
    executor = DeployExecutor()
    monkeypatch.setattr(
        executor, "_probe_runtime_health", Mock(return_value=(_check("runtime"), None))
    )
    original = command_runner.side_effect

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 10)
        if (
            (failure == "id" and "{{.Image}}" in argv)
            or (failure == "ref" and "{{.Config.Image}}" in argv)
            or (failure == "retain" and argv[:2] == ["docker", "tag"])
        ):
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="docker unavailable"
            )
        return original(argv, **kwargs)

    command_runner.side_effect = run
    assert executor_mod.capture_rollback_point(executor, EnumRuntimeLane.DEV) is None


def test_capture_keeps_registry_port_in_retention_reference(
    monkeypatch: pytest.MonkeyPatch, command_runner: Mock
) -> None:
    executor = DeployExecutor()
    monkeypatch.setattr(
        executor, "_probe_runtime_health", Mock(return_value=(_check("runtime"), None))
    )
    original = command_runner.side_effect

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "{{.Config.Image}}" in argv:
            return subprocess.CompletedProcess(
                argv, 0, stdout="localhost:5000/team/runtime:latest", stderr=""
            )
        return original(argv, **kwargs)

    command_runner.side_effect = run
    assert (
        executor_mod.capture_rollback_point(executor, EnumRuntimeLane.DEV) is not None
    )
    assert any(
        call.args[0][-1] == "localhost:5000/team/runtime:rollback-dev"
        for call in command_runner.call_args_list
    )


@pytest.mark.parametrize("verify_fails", [False, True])
def test_restore_retags_distinct_refs_then_recreates_and_verifies(
    monkeypatch: pytest.MonkeyPatch, command_runner: Mock, verify_fails: bool
) -> None:
    executor = DeployExecutor()
    point = _point(EnumRuntimeLane.STABILITY_TEST)
    order = Mock()
    order.attach_mock(command_runner, "run")
    compose = Mock()
    verify = Mock(
        side_effect=VerificationFailedError([_check("runtime", healthy=False)])
        if verify_fails
        else None
    )
    order.attach_mock(compose, "compose")
    order.attach_mock(verify, "verify")
    monkeypatch.setattr(executor, "_compose_up", compose)
    monkeypatch.setattr(executor, "verify", verify)
    callback = Mock()
    outcome = executor_mod.restore_rollback_point(executor, point, callback)
    assert outcome.restored is True
    assert outcome.verified is (not verify_fails)
    assert command_runner.call_count == 1
    assert command_runner.call_args.args[0] == [
        "docker",
        "tag",
        _IMAGE_ID,
        "runtime:latest",
    ]
    compose.assert_called_once_with(
        Phase.RUNTIME,
        Scope.RUNTIME,
        executor_mod._requested_services_for_up(Scope.RUNTIME, [], lane=point.lane),
        callback,
        lane=point.lane,
        force_recreate=True,
        build=False,
    )
    verify.assert_called_once_with(on_phase_update=callback, lane=point.lane)
    assert [call[0] for call in order.mock_calls] == ["run", "compose", "verify"]
    if verify_fails:
        assert "degraded" in outcome.detail


@pytest.mark.parametrize("failure", ["tag", "timeout", "compose"])
def test_restore_docker_failure_returns_outcome(
    monkeypatch: pytest.MonkeyPatch, command_runner: Mock, failure: str
) -> None:
    executor = DeployExecutor()
    compose = Mock(
        side_effect=RuntimeError("compose failed") if failure == "compose" else None
    )
    verify = Mock()
    monkeypatch.setattr(executor, "_compose_up", compose)
    monkeypatch.setattr(executor, "verify", verify)
    if failure == "tag":
        command_runner.side_effect = None
        command_runner.return_value = subprocess.CompletedProcess(
            [], 1, stdout="", stderr="tag failed"
        )
    elif failure == "timeout":
        command_runner.side_effect = subprocess.TimeoutExpired("docker tag", 10)
    outcome = executor_mod.restore_rollback_point(executor, _point(), Mock())
    assert outcome.restored is False
    assert outcome.verified is False
    assert outcome.detail
    verify.assert_not_called()
    if failure != "compose":
        compose.assert_not_called()


def test_restore_restores_each_separate_image_reference(
    monkeypatch: pytest.MonkeyPatch, command_runner: Mock
) -> None:
    executor = DeployExecutor()
    monkeypatch.setattr(executor, "_compose_up", Mock())
    monkeypatch.setattr(executor, "verify", Mock())
    point = _point()
    point = point.model_copy(
        update={
            "images": (
                point.images[0],
                point.images[1].model_copy(
                    update={"image_ref": "effects:latest", "image_id": _OTHER_ID}
                ),
            )
        }
    )
    assert executor_mod.restore_rollback_point(executor, point, Mock()).verified
    assert [call.args[0] for call in command_runner.call_args_list] == [
        ["docker", "tag", _IMAGE_ID, "runtime:latest"],
        ["docker", "tag", _OTHER_ID, "effects:latest"],
    ]


def test_compose_restore_never_builds_or_pulls_including_migrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = DeployExecutor()
    commands: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="t", stderr="")

    monkeypatch.setattr(executor_mod, "_run", run)
    monkeypatch.setattr(
        executor_mod, "verify_containers_up", Mock(return_value=(True, []))
    )
    monkeypatch.setattr(
        executor_mod, "verify_oneshots_completed", Mock(return_value=(True, []))
    )
    monkeypatch.setattr(
        executor_mod, "read_checkout_migration_fingerprint", lambda: None
    )
    executor._compose_up(
        Phase.RUNTIME, Scope.RUNTIME, ["omninode-runtime"], Mock(), build=False
    )
    ups = [
        argv for argv in commands if argv[:2] == ["docker", "compose"] and "up" in argv
    ]
    assert len(ups) > 1
    for argv in ups:
        assert "--no-build" in argv
        assert argv[argv.index("--pull") + 1] == "never"
    assert ups[-1][-1] == "omninode-runtime"


def _agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cmd: ModelRebuildRequested
) -> tuple[DeployAgent, Mock]:
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(tmp_path / "lane-locks"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod.socket, "gethostname", lambda: "rollback-test")
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = JobStore(tmp_path)
    agent.job_store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    executor = Mock(spec=DeployExecutor)
    executor.git_pull.return_value = "a" * 40
    executor.rebuild_scope.return_value = ["omninode-runtime"]
    executor.verify.return_value = []
    executor.deploy_and_verify.return_value = []
    executor.resolve_stability_ready_digest.return_value = cmd.image_digest
    for name in (
        "container_residue",
        "recreate_supervision",
        "verify_recreate",
        "deps_convergence",
        "compose_invocations",
        "health_checks",
    ):
        setattr(executor, name, [])
    executor.sibling_source_refs = {}
    agent.executor = executor
    monkeypatch.setattr(agent, "_resolve_lab_overlay_sha", lambda *args, **kwargs: None)
    return agent, executor


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
@pytest.mark.parametrize("verified", [False, True])
def test_agent_failed_verify_restores_and_job_stays_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: Scope, verified: bool
) -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=scope,
        runtime_lane=EnumRuntimeLane.DEV,
    )
    agent, executor = _agent(tmp_path, monkeypatch, cmd)
    point = _point()
    capture = Mock(return_value=point)
    restore = Mock(
        return_value=executor_mod.ModelRollbackOutcome(
            restored=True, verified=verified, detail="restored"
        )
    )
    monkeypatch.setattr(agent_mod, "capture_rollback_point", capture)
    monkeypatch.setattr(agent_mod, "restore_rollback_point", restore)
    failure = VerificationFailedError([_check("runtime-effects", healthy=False)])
    executor.verify.side_effect = failure
    order = Mock()
    order.attach_mock(capture, "capture")
    order.attach_mock(executor.rebuild_scope, "rebuild")
    order.attach_mock(executor.verify, "verify")
    order.attach_mock(restore, "restore")
    agent._run_deploy(cmd)
    assert [call[0] for call in order.mock_calls] == [
        "capture",
        "rebuild",
        "verify",
        "restore",
    ]
    capture.assert_called_once_with(executor, cmd.runtime_lane)
    assert restore.call_args.args[:2] == (executor, point)
    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None and job.status == "failed"
    assert job.errors == [
        f"{failure}; auto-rollback: restored=True verified={verified} to {'a' * 12}"
    ]


@pytest.mark.parametrize(
    ("lane", "scope"),
    [
        (EnumRuntimeLane.DEV, Scope.RUNTIME),
        (EnumRuntimeLane.PROD, Scope.RUNTIME),
        (EnumRuntimeLane.DEV, Scope.CORE),
    ],
)
def test_agent_passing_verify_never_restores_and_capture_is_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: EnumRuntimeLane, scope: Scope
) -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=scope,
        runtime_lane=lane,
        image_digest=_IMAGE_ID if lane == EnumRuntimeLane.PROD else None,
    )
    agent, executor = _agent(tmp_path, monkeypatch, cmd)
    capture = Mock(return_value=_point())
    restore = Mock()
    monkeypatch.setattr(agent_mod, "capture_rollback_point", capture)
    monkeypatch.setattr(agent_mod, "restore_rollback_point", restore)
    agent._run_deploy(cmd)
    restore.assert_not_called()
    if lane == EnumRuntimeLane.DEV and scope == Scope.RUNTIME:
        capture.assert_called_once_with(executor, lane)
    else:
        capture.assert_not_called()
    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None and job.status == "success"


def test_agent_failed_verify_without_point_keeps_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )
    agent, executor = _agent(tmp_path, monkeypatch, cmd)
    monkeypatch.setattr(agent_mod, "capture_rollback_point", Mock(return_value=None))
    restore = Mock()
    monkeypatch.setattr(agent_mod, "restore_rollback_point", restore)
    failure = VerificationFailedError([_check("runtime-effects", healthy=False)])
    executor.verify.side_effect = failure
    agent._run_deploy(cmd)
    restore.assert_not_called()
    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None and job.status == "failed"
    assert job.errors == [str(failure)]


def test_rollback_error_preserves_verification_failure_and_outcome() -> None:
    failure = VerificationFailedError([_check("runtime", healthy=False)])
    outcome = executor_mod.ModelRollbackOutcome(
        restored=False, verified=False, detail="tag failed"
    )
    error = executor_mod.DeployRolledBackError(failure, outcome, _point())
    assert isinstance(error, VerificationFailedError)
    assert error.failures == failure.failures
    assert error.outcome == outcome
    assert str(error).startswith(str(failure) + "; auto-rollback:")


def test_compose_restore_recovery_also_never_builds_or_pulls(
    monkeypatch: pytest.MonkeyPatch,
    command_runner: Mock,
) -> None:
    executor = DeployExecutor()
    monkeypatch.setattr(executor, "_ensure_runtime_migrations_ready", Mock())
    monkeypatch.setattr(executor, "_record_container_residue", Mock())
    monkeypatch.setattr(
        executor_mod,
        "verify_containers_up",
        Mock(side_effect=[(False, ["omninode-runtime"]), (True, [])]),
    )
    recovery = Mock(
        return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")
    )
    monkeypatch.setattr(executor_mod.subprocess, "run", recovery)
    executor._compose_up(
        Phase.RUNTIME, Scope.RUNTIME, ["omninode-runtime"], Mock(), build=False
    )
    argv = recovery.call_args.args[0]
    assert "--no-build" in argv
    assert argv[argv.index("--pull") + 1] == "never"
