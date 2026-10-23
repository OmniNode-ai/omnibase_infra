# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A refused offset commit is a group rejoin, never a crash (OMN-17427).

THE MEASURED CRASH
------------------

On 2026-10-22T00:13:06Z the dev deploy agent finished a 12-minute job and
exited 1 on the first record it polled afterwards: a command routed to another
instance. The agent does not poll while a job runs, so a job longer than
kafka-python's ``max_poll_interval_ms`` (300 s) costs the group membership, and
the routing skip's ``_commit_through`` raised
``CommitFailedError: [Error 25] UnknownMemberIdError``. OMN-20133 caught that
refusal on the accept path only; every reject, skip, dedup, quarantine and
rewind commit still let it escape.

kafka-python 2.3.2 has already reset the generation and requested a rejoin
when it raises ``CommitFailedError``. The next ``poll()`` rejoins, the fetch
position is re-read from the committed offset, and the uncommitted record is
redelivered -- so the refusal is logged and the poll loop carries on.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested
from kafka import TopicPartition
from kafka.errors import CommitFailedError, CorruptRecordError

TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
TP = TopicPartition(TOPIC, 0)
OFFSET = 7100
CORRELATION = "94a57f48-0000-4000-8000-000000000001"


class _DroppedMember:
    """A kafka client double whose next ``refusals`` commits are refused.

    A refusal is what kafka-python raises for a member the coordinator has
    dropped. ``poll`` hands back the same record every time, which is what the
    rejoin does with a record whose commit never landed.
    """

    def __init__(self, record: Any = None, *, refusals: int = 1) -> None:
        self.record = record
        self.refusals = refusals
        self.committed: dict[TopicPartition, int] = {}
        self.seeks: list[tuple[TopicPartition, int]] = []
        self.positions: dict[TopicPartition, int] = {TP: OFFSET}
        self.highwaters: dict[TopicPartition, int] = {TP: OFFSET + 1}

    def poll(self, timeout_ms: int = 0) -> dict[TopicPartition, list[Any]]:
        return {TP: [self.record]} if self.record is not None else {}

    def commit(self, offsets: dict[TopicPartition, Any]) -> None:
        if self.refusals > 0:
            self.refusals -= 1
            raise CommitFailedError("UnknownMemberIdError")
        for topic_partition, meta in offsets.items():
            self.committed[topic_partition] = meta.offset

    def seek(self, topic_partition: TopicPartition, offset: int) -> None:
        self.seeks.append((topic_partition, offset))

    def assignment(self) -> set[TopicPartition]:
        return {TP}

    def position(self, topic_partition: TopicPartition) -> int:
        return self.positions[topic_partition]

    def highwater(self, topic_partition: TopicPartition) -> int:
        return self.highwaters[topic_partition]


class _RoutedElsewhere:
    """A router that gives every command to another instance."""

    instance = SimpleNamespace(name="dev-a")

    def is_mine(self, cmd: ModelRebuildRequested) -> tuple[bool, Any]:
        return False, SimpleNamespace(instance="dev-b", basis="routing table")


def _record(runtime_lane: str = "dev") -> SimpleNamespace:
    payload = {
        "correlation_id": CORRELATION,
        "git_ref": "39e52ccb2a6f0d7c1e7b9d1d3c0f4a5b6c7d8e9f",
        "requested_by": "gha/omnibase_infra/pr-5200",
        "scope": "full",
        "runtime_lane": runtime_lane,
        "services": [],
        "_signature": "a" * 64,
    }
    return SimpleNamespace(
        value=payload, topic=TOPIC, partition=0, offset=OFFSET, key=None
    )


class _JobStore:
    def __init__(self, *, duplicate: bool = False) -> None:
        self.duplicate = duplicate
        self.accepted: list[Any] = []

    def has_active_job(self) -> bool:
        return False

    def is_duplicate(self, correlation_id: Any) -> bool:
        return self.duplicate

    def accept(self, **kwargs: Any) -> None:
        self.accepted.append(kwargs["correlation_id"])


def _consumer(member: _DroppedMember, **attrs: Any) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = member
    consumer.job_store = attrs.pop("job_store", _JobStore())
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    for name, value in attrs.items():
        setattr(consumer, name, value)
    return consumer


def _refusal_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "commit_refused_rejoin" in r.getMessage()
    ]


@pytest.mark.unit
def test_routed_elsewhere_refused_commit_returns_nothing_and_rejoins(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The 00:13:06Z path: a skip whose commit is refused does not raise."""
    member = _DroppedMember(_record())
    consumer = _consumer(member, router=_RoutedElsewhere())

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        caplog.at_level(logging.WARNING, logger="deploy_agent.consumer"),
    ):
        assert consumer.poll_and_accept() == (None, None)

    assert member.committed == {}
    assert member.seeks == [], "nothing was committed, so nothing is sought past"
    warnings = _refusal_warnings(caplog)
    assert len(warnings) == 1
    assert f"{TOPIC}@0:{OFFSET}" in warnings[0]
    assert "UnknownMemberIdError" in warnings[0]
    assert "redelivered" in warnings[0]


@pytest.mark.unit
def test_lane_not_allowed_refused_commit_still_rejects(
    caplog: pytest.LogCaptureFixture,
) -> None:
    member = _DroppedMember(_record(runtime_lane=EnumRuntimeLane.STABILITY_TEST.value))
    consumer = _consumer(member)

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        caplog.at_level(logging.WARNING, logger="deploy_agent.consumer"),
    ):
        assert consumer.poll_and_accept() == (None, "lane_not_allowed")

    assert member.committed == {}
    assert len(_refusal_warnings(caplog)) == 1


@pytest.mark.unit
def test_duplicate_refused_commit_still_rejects(
    caplog: pytest.LogCaptureFixture,
) -> None:
    member = _DroppedMember(_record())
    consumer = _consumer(member, job_store=_JobStore(duplicate=True))

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        caplog.at_level(logging.WARNING, logger="deploy_agent.consumer"),
    ):
        assert consumer.poll_and_accept() == (None, "duplicate")

    assert len(_refusal_warnings(caplog)) == 1


@pytest.mark.unit
def test_accept_refused_commit_returns_the_command_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """OMN-20133's property survives the move into ``_commit_through``."""
    member = _DroppedMember(_record())
    job_store = _JobStore()
    consumer = _consumer(member, job_store=job_store)

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        caplog.at_level(logging.WARNING, logger="deploy_agent.consumer"),
    ):
        cmd, reason = consumer.poll_and_accept()

    assert reason is None
    assert cmd is not None
    assert str(cmd.correlation_id) == CORRELATION
    assert len(job_store.accepted) == 1
    assert any(
        CORRELATION in r.getMessage()
        and "friction_type=accept_commit_failed" in r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR
    )


@pytest.mark.unit
def test_quarantine_record_refused_commit_does_not_raise(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A record the contract refuses is quarantined; its refused commit is logged."""
    record = _record()
    record.value["scope"] = "not-a-scope"
    member = _DroppedMember(record)
    consumer = _consumer(member, quarantine_dir=tmp_path / "quarantine")

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        patch.object(DeployConsumer, "_publish_dlq"),
        caplog.at_level(logging.WARNING, logger="deploy_agent.consumer"),
    ):
        assert consumer.poll_and_accept() == (None, "invalid_payload")

    assert len(_refusal_warnings(caplog)) == 1


@pytest.mark.unit
def test_undecodable_fetch_refused_commit_does_not_raise(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The fetch-level quarantine commit: refused, logged, and not reported advanced."""
    member = _DroppedMember()
    consumer = _consumer(member, quarantine_dir=tmp_path / "quarantine")

    with (
        patch.object(DeployConsumer, "_publish_dlq"),
        caplog.at_level(logging.WARNING, logger="deploy_agent.consumer"),
    ):
        reason = consumer._quarantine_undecodable_fetch(CorruptRecordError("crc"))

    assert reason == "undecodable_fetch_commit_refused"
    assert member.committed == {}
    warnings = _refusal_warnings(caplog)
    assert len(warnings) == 1
    assert f"{TOPIC}@0:{OFFSET + 1}" in warnings[0]


@pytest.mark.unit
def test_rewind_refused_commit_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    member = _DroppedMember()
    consumer = _consumer(member)

    with caplog.at_level(logging.WARNING, logger="deploy_agent.consumer"):
        consumer._rewind_committed_offset_to(_record())

    assert member.committed == {}
    warnings = _refusal_warnings(caplog)
    assert len(warnings) == 1
    assert "last committed offset" in warnings[0]
    assert not any(
        "Rewound committed offset" in r.getMessage() for r in caplog.records
    ), "a refused rewind must not be reported as a rewind"


@pytest.mark.unit
def test_the_next_poll_after_a_refusal_handles_the_redelivery_and_commits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """After the rejoin the same record comes back, is skipped again, and commits."""
    member = _DroppedMember(_record())
    consumer = _consumer(member, router=_RoutedElsewhere())

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        caplog.at_level(logging.WARNING, logger="deploy_agent.consumer"),
    ):
        assert consumer.poll_and_accept() == (None, None)
        assert member.committed == {}
        assert consumer.poll_and_accept() == (None, None)

    assert member.committed == {TP: OFFSET + 1}
    assert member.seeks == [(TP, OFFSET + 1)]
    assert len(_refusal_warnings(caplog)) == 1
