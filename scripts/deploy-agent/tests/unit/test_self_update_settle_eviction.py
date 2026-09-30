# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A pre-accept self-update that waits out a long settle must not lose its command (OMN-20133).

Live evidence this file encodes, job ``7b970ab9-cdd6-4048-bd03-aab2e380d661``
for omnibase_infra ``d8c786d04`` on the .201 dev lane, 2026-09-30T03:17:59Z to
03:24:43Z (journal of ``deploy-agent-dev``):

1. The command cleared every acceptance check and reached the pre-accept
   self-update boundary. The clone was behind, so ``self_update`` decided to
   re-exec and called ``on_before_reexec``.
2. That callback first waited for the lab-overlay settle in flight (OMN-19501),
   for 6m43s. Nothing called ``poll()`` in that time, which is longer than
   kafka-python's default ``max_poll_interval_ms`` of 300 s, so the client
   left group ``onex-deploy-agent`` at 03:22:59Z.
3. The callback then rewound the committed offset. The commit raised
   ``CommitFailedError`` because the member was no longer in the group, so no
   re-exec happened.
4. The consumer's error rail logged "proceeding on the current image",
   ``job_store.accept`` persisted the job, and the accept commit raised the same
   ``CommitFailedError`` uncaught. The process exited 1.
5. The replacement process recovered the accepted job as crashed (failed, every
   phase skipped) and refused the redelivered command as a duplicate. The dev
   lane never deployed that sha. The same sequence hit ``5b0051ae`` at
   2026-09-29T22:53Z and ``d969b2d9`` at 2026-09-30T05:07Z.

Two properties close it:

* the rewind is committed while the member is still live, BEFORE the settle
  wait, so whatever the wait does to the membership the replacement process
  re-reads the command and runs it on the new code (the retry after the update);
* a commit that fails after the job has been accepted is logged at ERROR and
  the job runs, rather than crashing the process and orphaning the job (the
  completion without the update).
"""

from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent.agent import DeployAgent
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import EnumRuntimeLane, EnumSelfUpdateBoundary
from deploy_agent.job_state import JobStore
from kafka import TopicPartition
from kafka.errors import CommitFailedError

TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
OFFSET = 53432


class _GroupMember:
    """A kafka client double that models group membership.

    kafka-python refuses an offset commit from a member the coordinator has
    dropped. That refusal is the only broker behaviour these tests depend on,
    so it is the only one modelled.
    """

    def __init__(self) -> None:
        self.in_group = True
        self.committed: dict[TopicPartition, int] = {}
        self.seeks: list[int] = []

    def evict(self) -> None:
        """What a poll gap longer than ``max_poll_interval_ms`` does."""
        self.in_group = False

    def commit(self, offsets: dict[TopicPartition, Any]) -> None:
        if not self.in_group:
            raise CommitFailedError(
                "Offset commit cannot be completed since the consumer is not "
                "part of an active group for auto partition assignment"
            )
        for topic_partition, meta in offsets.items():
            self.committed[topic_partition] = meta.offset

    def seek(self, topic_partition: TopicPartition, offset: int) -> None:
        self.seeks.append(offset)


def _message(correlation_id: str) -> SimpleNamespace:
    payload = {
        "correlation_id": correlation_id,
        "git_ref": "d8c786d04697a03b62692c66e768ef8dcc019c2f",
        "requested_by": "gha/omnibase_infra/pr-4319",
        "scope": "full",
        "runtime_lane": "dev",
        "services": [],
        "_signature": "a" * 64,
    }
    return SimpleNamespace(value=payload, topic=TOPIC, partition=0, offset=OFFSET)


def _consumer(job_store: JobStore, member: _GroupMember, hook: Any) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = member
    consumer.job_store = job_store
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = hook
    return consumer


class _ReexecNow(BaseException):
    """What ``os.execv`` does to the calling image: it never returns."""


class _BehindExecutor:
    """``self_update`` for a clone that is behind: hand off, then re-exec."""

    def __init__(self) -> None:
        self.boundaries: list[EnumSelfUpdateBoundary] = []

    def self_update(
        self,
        *,
        boundary: EnumSelfUpdateBoundary,
        skip: bool = False,
        on_before_reexec: Any = None,
    ) -> None:
        self.boundaries.append(boundary)
        if on_before_reexec is not None:
            on_before_reexec()
        raise _ReexecNow


class _AgentWithLongSettle:
    """The slice of ``DeployAgent`` its pre-accept boundary reads.

    ``_await_settle_before_reexec`` stands in for a settle that outlasts the
    poll interval: by the time it returns, the coordinator has dropped the
    member.
    """

    def __init__(self, member: _GroupMember) -> None:
        self.member = member
        self.executor = _BehindExecutor()
        self._skip_self_update = False
        self.events: list[str] = []

    def _await_settle_before_reexec(self) -> None:
        self.events.append("settle-wait")
        self.member.evict()


@pytest.mark.unit
def test_a_settle_wait_that_evicts_the_member_still_reexecs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """AC1: the re-exec happens and the replacement re-reads the command.

    RED against the wait-then-rewind order: the rewind commit raises after the
    eviction, the boundary's error rail swallows it, the job is accepted, and
    the accept commit raises ``CommitFailedError`` out of ``_process_message``.
    """
    job_store = JobStore(state_dir=tmp_path / "jobs")
    member = _GroupMember()
    agent = _AgentWithLongSettle(member)
    consumer = _consumer(
        job_store,
        member,
        lambda rewind: DeployAgent._self_update_pre_accept(agent, rewind),  # type: ignore[arg-type]
    )
    correlation_id = str(uuid.uuid4())

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        pytest.raises(_ReexecNow),
    ):
        consumer._process_message(_message(correlation_id))

    assert agent.executor.boundaries == [EnumSelfUpdateBoundary.PRE_ACCEPT]
    assert "settle-wait" in agent.events
    assert member.committed == {TopicPartition(TOPIC, 0): OFFSET}, (
        "the committed offset must point AT the command, so the replacement "
        "process re-reads it and runs it on the new code"
    )
    assert job_store.load(uuid.UUID(correlation_id)) is None, (
        "nothing may be accepted before a re-exec: an accepted job left behind "
        "is recovered as crashed and its redelivery is refused as a duplicate"
    )


@pytest.mark.unit
def test_the_rewind_is_committed_before_the_settle_wait(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The order itself: rewind while the member is live, then wait."""
    member = _GroupMember()
    agent = _AgentWithLongSettle(member)
    order: list[str] = []

    def _rewind() -> None:
        order.append("rewind" if member.in_group else "rewind-after-eviction")

    def _wait() -> None:
        order.append("settle-wait")
        member.evict()

    agent._await_settle_before_reexec = _wait  # type: ignore[method-assign]
    with pytest.raises(_ReexecNow):
        DeployAgent._self_update_pre_accept(agent, _rewind)  # type: ignore[arg-type]

    assert order == ["rewind", "settle-wait"]


@pytest.mark.unit
def test_accept_commit_failure_runs_the_job_and_says_so(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    """AC2: a commit refused after the accept does not crash the process.

    The job is on disk as accepted. Raising here is what turned it into
    "Recovered 1 crashed job(s)" with every phase skipped. Returning the command
    runs it; its redelivery after the rejoin is then refused as the duplicate
    it genuinely is.
    """
    job_store = JobStore(state_dir=tmp_path / "jobs")
    member = _GroupMember()
    member.evict()
    consumer = _consumer(job_store, member, lambda rewind: None)
    correlation_id = str(uuid.uuid4())

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        caplog.at_level(logging.ERROR, logger="deploy_agent.consumer"),
    ):
        cmd, reason = consumer._process_message(_message(correlation_id))

    assert reason is None
    assert cmd is not None
    assert str(cmd.correlation_id) == correlation_id
    record = job_store.load(uuid.UUID(correlation_id))
    assert record is not None
    assert record.status == "accepted"
    assert any(
        correlation_id in r.getMessage()
        and "friction_type=accept_commit_failed" in r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR
    ), "the failed commit must be reported at ERROR, naming the command"


@pytest.mark.unit
def test_rewind_refused_before_the_wait_still_runs_the_command(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    """A member already dropped when the boundary starts cannot rewind.

    The boundary's own error rail logs that and proceeds on the current image;
    the accept-commit property then has to carry the job to completion.
    """
    job_store = JobStore(state_dir=tmp_path / "jobs")
    member = _GroupMember()
    member.evict()
    agent = _AgentWithLongSettle(member)
    consumer = _consumer(
        job_store,
        member,
        lambda rewind: DeployAgent._self_update_pre_accept(agent, rewind),  # type: ignore[arg-type]
    )
    correlation_id = str(uuid.uuid4())

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        caplog.at_level(logging.ERROR, logger="deploy_agent.consumer"),
    ):
        cmd, reason = consumer._process_message(_message(correlation_id))

    assert reason is None
    assert cmd is not None
    assert job_store.load(uuid.UUID(correlation_id)) is not None
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("friction_type=self_update_boundary_failed" in m for m in messages)
    assert any("friction_type=accept_commit_failed" in m for m in messages)
