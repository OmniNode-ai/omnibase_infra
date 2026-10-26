# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression: 70315f50 must not wait out 3a55c329's exhausted boot budget."""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import consumer as consumer_mod
from deploy_agent import executor as executor_mod
from deploy_agent import lock as lock_mod
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    DeployInProgressError,
    EnumRuntimeLane,
    ModelRebuildRequested,
    ModelRollbackDeclaration,
    Phase,
    PhaseStatus,
)
from deploy_agent.executor import DeployExecutor, VerificationFailedError
from deploy_agent.job_state import JobStore
from kafka import TopicPartition

from tests.unit.test_auto_rollback import _agent, _check, _point
from tests.unit.test_verify_force_recreate_omn18640 import (
    _completed,
    _Lane,
    _recreated_services,
)

pytestmark = pytest.mark.unit
TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
GOOD_TAG = "dev-20261025120130-26d49cec3536"
EXHAUSTED = "[entrypoint] RETRIES EXHAUSTED: 5 consecutive failing boots"


def _cmd(**overrides: object) -> ModelRebuildRequested:
    return ModelRebuildRequested.model_validate(
        {
            "correlation_id": str(uuid4()),
            "requested_by": "test/omn21279",
            "scope": "runtime",
            "runtime_lane": "dev",
            "git_ref": "a" * 40,
            **overrides,
        }
    )


def _rollback(**overrides: object) -> ModelRebuildRequested:
    return _cmd(
        **{
            "git_ref": GOOD_TAG,
            "rollback": ModelRollbackDeclaration(actor="operator", reason="bad boot"),
            **overrides,
        }
    )


def _message(cmd: ModelRebuildRequested, offset: int, partition: int = 0):
    return SimpleNamespace(
        topic=TOPIC,
        partition=partition,
        offset=offset,
        value={**cmd.model_dump(mode="json"), "_signature": "valid"},
    )


def _consumer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, messages: list):
    monkeypatch.setattr(
        consumer_mod,
        "verify_command",
        lambda payload: payload.get("_signature") == "valid",
    )
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = JobStore(tmp_path / "jobs")
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    consumer.consumer.poll.return_value = {TopicPartition(TOPIC, 0): messages}
    return consumer


def _commits(consumer: DeployConsumer) -> list[int]:
    return [
        meta.offset
        for call in consumer.consumer.commit.call_args_list
        for meta in call.args[0].values()
    ]


def test_tag_rollback_runs_first_and_earlier_forward_is_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forward, rollback, later = _cmd(), _rollback(), _cmd()
    messages = [_message(forward, 10), _message(rollback, 11), _message(later, 12)]
    consumer = _consumer(tmp_path, monkeypatch, messages)
    accepted, reason = consumer.poll_and_accept()
    assert reason is None and accepted == rollback
    assert consumer.job_store.load(forward.correlation_id) is None
    assert consumer.job_store.load(later.correlation_id) is None
    assert _commits(consumer) == [10], "never acknowledge an unprocessed forward deploy"
    consumer.consumer.seek.assert_called_with(TopicPartition(TOPIC, 0), 10)

    # A process restart re-reads the committed frontier. The durable rollback
    # record prevents priority from choosing it twice; the forward still runs.
    consumer.job_store.complete(rollback.correlation_id, status="success")
    consumer.job_store.mark_published(rollback.correlation_id)
    consumer = _consumer(tmp_path, monkeypatch, messages)
    accepted, reason = consumer.poll_and_accept()
    assert reason is None and accepted == forward
    assert _commits(consumer) == [11]
    consumer.job_store.complete(forward.correlation_id, status="success")
    consumer.job_store.mark_published(forward.correlation_id)
    consumer.consumer.poll.return_value = {TopicPartition(TOPIC, 0): messages[1:]}
    accepted, reason = consumer.poll_and_accept()
    assert accepted is None and reason == "duplicate"
    assert _commits(consumer) == [11, 12]
    consumer.consumer.poll.return_value = {TopicPartition(TOPIC, 0): messages[2:]}
    accepted, reason = consumer.poll_and_accept()
    assert reason is None and accepted == later
    assert _commits(consumer) == [11, 12, 13]


@pytest.mark.parametrize(
    "case", ["unsigned", "undeclared", "branch", "sha", "other-lane", "other-instance"]
)
def test_priority_requires_signed_declared_tag_on_the_same_lane_and_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    forward = _cmd()
    overrides = {
        "undeclared": {"rollback": None},
        "branch": {"git_ref": "origin/dev"},
        "sha": {"git_ref": "b" * 40},
        "other-lane": {"runtime_lane": EnumRuntimeLane.STABILITY_TEST},
    }.get(case, {})
    candidate = _message(_rollback(**overrides), 11)
    if case == "unsigned":
        candidate.value["_signature"] = "forged"
    consumer = _consumer(tmp_path, monkeypatch, [_message(forward, 10), candidate])
    if case == "other-instance":
        consumer.router = Mock()
        consumer.router.is_mine.side_effect = lambda cmd: (
            cmd.correlation_id == forward.correlation_id,
            None,
        )
    accepted, reason = consumer.poll_and_accept()
    assert reason is None and accepted == forward
    assert _commits(consumer) == [11]


def test_priority_is_fifo_among_rollbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _rollback(), _rollback()
    consumer = _consumer(
        tmp_path,
        monkeypatch,
        [_message(_cmd(), 10), _message(first, 11), _message(second, 12)],
    )
    accepted, reason = consumer.poll_and_accept()
    assert reason is None and accepted == first


def test_priority_self_update_rewinds_to_the_earliest_unprocessed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumer = _consumer(
        tmp_path, monkeypatch, [_message(_cmd(), 10), _message(_rollback(), 11)]
    )

    class Reexec(BaseException):
        pass

    def update(rewind):
        rewind()
        raise Reexec

    consumer.self_update_hook = update
    with pytest.raises(Reexec):
        consumer.poll_and_accept()
    assert _commits(consumer) == [10]
    consumer.consumer.seek.assert_called_with(TopicPartition(TOPIC, 0), 10)


def test_priority_keeps_single_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumer = _consumer(
        tmp_path, monkeypatch, [_message(_cmd(), 10), _message(_rollback(), 11)]
    )
    active = _cmd()
    consumer.job_store.accept(active.correlation_id, active.model_dump(mode="json"))
    accepted, reason = consumer.poll_and_accept()
    assert accepted is None and reason == "busy"
    assert _commits(consumer) == [10]


@pytest.mark.parametrize("when", ["initial", "start-wait", "recreate-wait"])
def test_exhausted_boot_fails_without_spending_the_remaining_budget(
    monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    executor = DeployExecutor()
    lane = _Lane(effects_health=["timeout"])
    log_reads = 0
    sleeps: list[float] = []

    def run(argv: list[str], timeout: int, **kwargs):
        nonlocal log_reads
        if argv[:2] == ["docker", "logs"]:
            log_reads += 1
            return _completed(
                argv,
                stderr=EXHAUSTED
                if log_reads >= (1 if when == "initial" else 2)
                else "",
            )
        if (
            when == "start-wait"
            and argv[0] == "bash"
            and argv[5] == "runtime_health_state"
        ):
            return _completed(argv, stdout="starting")
        return lane(argv, timeout, **kwargs)

    monkeypatch.setattr(executor_mod, "_run", run)
    monkeypatch.setattr(executor_mod, "_verify_recreate_sleep", sleeps.append)
    phases = Mock()
    with pytest.raises(VerificationFailedError, match="boot retries exhausted"):
        executor.verify(phases)
    assert len(sleeps) <= 1, "a parked boot must not consume a 2000-second budget"
    assert log_reads <= 2
    assert (Phase.VERIFICATION, PhaseStatus.SUCCESS) not in [
        c.args for c in phases.call_args_list
    ]
    assert any(
        "boot retries exhausted" in check.detail for check in executor.health_checks
    )
    assert {c.service for c in executor.health_checks} >= {
        "omninode-runtime",
        "runtime-effects",
    }, "early failure must retain the up-front reading of every runtime target"
    if when != "recreate-wait":
        assert _recreated_services(lane.cmds) == []


@pytest.mark.parametrize(
    "logs", ["[entrypoint] BOOT FAILED 1/5: transient", "unreadable"]
)
def test_transient_or_unreadable_boot_log_keeps_the_health_checks(
    monkeypatch: pytest.MonkeyPatch, logs: str
) -> None:
    executor = DeployExecutor()
    lane = _Lane(effects_health=["timeout", "healthy"])

    def run(argv: list[str], timeout: int, **kwargs):
        if argv[:2] == ["docker", "logs"]:
            return _completed(
                argv, stderr=logs, returncode=1 if logs == "unreadable" else 0
            )
        return lane(argv, timeout, **kwargs)

    monkeypatch.setattr(executor_mod, "_run", run)
    monkeypatch.setattr(executor_mod, "_verify_recreate_sleep", lambda seconds: None)
    checks = executor.verify(Mock())
    assert all(c.status == "pass" for c in checks if c.service != "docker")
    assert _recreated_services(lane.cmds) == ["runtime-effects"]


@pytest.mark.parametrize("failure_kind", ["verification", "timeout", "exhausted"])
def test_failure_restores_own_point_before_unlock_and_next_job_can_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str
) -> None:
    # Dev recovers through its boot guard (OMN-21279 #5354); the other
    # non-prod lanes restore their own retained rollback point.
    lane_value = (
        EnumRuntimeLane.DEV
        if failure_kind == "exhausted"
        else EnumRuntimeLane.STABILITY_TEST
    )
    cmd = _cmd(runtime_lane=lane_value)
    monkeypatch.setattr(lock_mod, "_LOCK_PATH", tmp_path / "single-flight.lock")
    agent, executor = _agent(tmp_path, monkeypatch, cmd)
    point = _point(cmd.runtime_lane)
    monkeypatch.setattr(agent_mod, "capture_rollback_point", Mock(return_value=point))
    held = False

    @contextmanager
    def lock(*args, **kwargs):
        nonlocal held
        assert not held
        held = True
        try:
            yield
        finally:
            held = False

    def assert_lane_still_locked() -> None:
        assert held, "restore must finish under the lane lock"
        with pytest.raises(DeployInProgressError):
            with lock_mod.single_flight_lock():
                pass

    def restore(executor_arg, point_arg, callback):
        assert_lane_still_locked()
        assert point_arg is point
        return executor_mod.ModelRollbackOutcome(
            restored=True, verified=True, detail="restored"
        )

    @contextmanager
    def protect(job, commit, mark_failed):
        try:
            yield
        except VerificationFailedError as exc:
            assert_lane_still_locked()
            assert "boot retries exhausted" in str(exc)
            mark_failed(f"commit={commit}: {exc}")
            raise RuntimeError(f"{exc}; forward redeploy verified=True") from exc

    guard = Mock()
    guard.protect.side_effect = protect
    monkeypatch.setattr(agent_mod, "DevBootGuard", Mock(return_value=guard))
    monkeypatch.setattr(agent_mod, "lane_lock", lock)
    monkeypatch.setattr(agent_mod, "restore_rollback_point", restore)
    monkeypatch.setattr(agent, "_self_update_post_terminal", lambda: None)
    executor.verify.side_effect = (
        VerificationFailedError([_check("runtime", healthy=False)])
        if failure_kind == "verification"
        else subprocess.TimeoutExpired("curl", 10)
    )
    if failure_kind == "exhausted":
        real_executor = DeployExecutor()
        lane = _Lane(effects_health=["timeout"])

        def run(argv: list[str], timeout: int, **kwargs):
            if argv[:2] == ["docker", "logs"]:
                return _completed(argv, stderr=EXHAUSTED)
            return lane(argv, timeout, **kwargs)

        monkeypatch.setattr(executor_mod, "_run", run)
        monkeypatch.setattr(
            executor_mod,
            "_verify_recreate_sleep",
            Mock(side_effect=AssertionError("parked boot must not sleep")),
        )
        executor.verify.side_effect = real_executor.verify
    agent._execute_command(cmd)
    assert not held
    record = agent.job_store.load(cmd.correlation_id)
    assert record is not None and record.status == "failed"
    assert (
        "forward redeploy verified=True"
        if failure_kind == "exhausted"
        else "auto-rollback: restored=True verified=True"
    ) in " ".join(record.errors)
    executor.verify.side_effect = None
    next_cmd = _cmd(runtime_lane=lane_value)
    agent.job_store.accept(next_cmd.correlation_id, next_cmd.model_dump(mode="json"))
    agent._execute_command(next_cmd)
    assert not held
    assert agent.job_store.load(next_cmd.correlation_id).status == "success"
