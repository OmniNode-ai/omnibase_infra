# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A newer dev build request cancels every older queued request (OMN-26637).

THE RULE (operator, 2026-11-24 about 12:20Z): when a new build comes out, all
previous queued builds are canceled. Until now the dev lane folded only a
queue whose refs were each contained in a later one. Squash merges publish
their commands out of order, so the chain broke and the agent rebuilt each stale
ref in turn at about 28 minutes a job, once walking dev six merges backwards.

Two properties, one per test class:

* ``supersede_dev_queue``: on a lane the policy names, accepting a request
  supersedes every older queued request, whatever git says about their refs,
  and each carries one record that names the accepting request;
* ``dev_accepts_only_merged_refs``: on that lane a request whose ref is not on
  the tracking branch is refused with a typed reason, never built.

The lanes are injected (``supersede_lanes``, ``branch_lanes``), so no test
here and no module under test spells a lane into the rule.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from deploy_agent.coalesce import (
    EnumCoalesceRefusal,
    ModelQueuedCommand,
    ModelSupersession,
    plan_coalesce,
)
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    EnumRejectionReason,
    EnumRuntimeLane,
    ModelRebuildRequested,
    ModelRejectionNotice,
    ModelRollbackDeclaration,
)
from deploy_agent.job_state import JobStore
from kafka import TopicPartition

pytestmark = pytest.mark.unit

TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
TRACKING = "origin/dev"
DEV = frozenset({EnumRuntimeLane.DEV})
OTHER_LANE = EnumRuntimeLane.STABILITY_TEST

#: Four refs no pair of which is ancestor-related, as squash-merged PR refs are.
SHA_A = "a1" * 20
SHA_B = "b2" * 20
SHA_C = "c3" * 20
SHA_D = "d4" * 20


def _cmd(sha: str, **overrides: Any) -> ModelRebuildRequested:
    payload: dict[str, Any] = {
        "correlation_id": uuid4(),
        "requested_by": "gha/omnibase_infra/pr-1",
        "scope": "full",
        "runtime_lane": "dev",
        "build_source": "workspace",
        "services": [],
        "git_ref": sha,
    }
    payload.update(overrides)
    return ModelRebuildRequested.model_validate(payload)


def _queued(cmd: ModelRebuildRequested, offset: int) -> ModelQueuedCommand:
    return ModelQueuedCommand(command=cmd, partition=0, offset=offset)


def _nothing_contains_anything(earlier: str, later: str) -> bool | None:
    """Every pair is proven NOT ancestor-related."""
    return False


def _nothing_provable(earlier: str, later: str) -> bool | None:
    """Every ancestry question is unanswerable, as with a clone that lacks the shas."""
    return None


class TestSupersedeDevQueue:
    def test_supersede_dev_queue_folds_every_older_request_whatever_the_ancestry(
        self,
    ) -> None:
        a, b, c, d = _cmd(SHA_A), _cmd(SHA_B), _cmd(SHA_C), _cmd(SHA_D)
        plan = plan_coalesce(
            [_queued(a, 10), _queued(b, 11), _queued(c, 12), _queued(d, 13)],
            contains=_nothing_contains_anything,
            supersede_lanes=DEV,
        )
        assert plan.runner.command.correlation_id == d.correlation_id
        assert [s.superseded.command.correlation_id for s in plan.superseded] == [
            a.correlation_id,
            b.correlation_id,
            c.correlation_id,
        ]
        assert {s.superseded_by_sha for s in plan.superseded} == {SHA_D}
        assert {s.superseded_by_correlation_id for s in plan.superseded} == {
            d.correlation_id
        }
        assert plan.stop_reason is None

    def test_supersede_dev_queue_asks_git_nothing(self) -> None:
        asked: list[tuple[str, str]] = []

        def _recording(earlier: str, later: str) -> bool | None:
            asked.append((earlier, later))
            return False

        plan_coalesce(
            [_queued(_cmd(SHA_A), 10), _queued(_cmd(SHA_B), 11)],
            contains=_recording,
            supersede_lanes=DEV,
        )
        assert asked == []

    def test_supersede_dev_queue_folds_when_ancestry_is_unprovable(self) -> None:
        plan = plan_coalesce(
            [_queued(_cmd(SHA_A), 10), _queued(_cmd(SHA_B), 11)],
            contains=_nothing_provable,
            supersede_lanes=DEV,
        )
        assert plan.superseded_count == 1

    def test_supersede_dev_queue_is_off_for_a_lane_the_policy_does_not_name(
        self,
    ) -> None:
        """The control: the same queue, no lane named, ends where it always did."""
        plan = plan_coalesce(
            [_queued(_cmd(SHA_A), 10), _queued(_cmd(SHA_B), 11)],
            contains=_nothing_contains_anything,
            supersede_lanes=frozenset(),
        )
        assert plan.superseded == ()
        assert plan.stop_reason is EnumCoalesceRefusal.NOT_A_DESCENDANT

    def test_supersede_dev_queue_never_folds_another_lane(self) -> None:
        plan = plan_coalesce(
            [
                _queued(_cmd(SHA_A), 10),
                _queued(_cmd(SHA_B, runtime_lane=OTHER_LANE.value), 11),
            ],
            contains=_nothing_contains_anything,
            supersede_lanes=DEV,
        )
        assert plan.superseded == ()
        assert plan.stop_reason is EnumCoalesceRefusal.DIFFERENT_LANE

    def test_supersede_dev_queue_keeps_a_declared_rollback(self) -> None:
        rollback = ModelRollbackDeclaration(actor="operator", reason="restore tag")
        plan = plan_coalesce(
            [_queued(_cmd(SHA_A), 10), _queued(_cmd(SHA_B, rollback=rollback), 11)],
            contains=_nothing_contains_anything,
            supersede_lanes=DEV,
        )
        assert plan.superseded == ()
        assert plan.stop_reason is EnumCoalesceRefusal.ROLLBACK_DECLARED

    def test_supersede_dev_queue_ends_at_a_redelivered_copy(self) -> None:
        a, b = _cmd(SHA_A), _cmd(SHA_B)
        plan = plan_coalesce(
            [_queued(a, 10), _queued(b, 11), _queued(a, 12)],
            contains=_nothing_contains_anything,
            supersede_lanes=DEV,
        )
        assert plan.runner.command.correlation_id == b.correlation_id
        assert plan.stop_reason is EnumCoalesceRefusal.DUPLICATE_COMMAND

    def test_supersede_dev_queue_record_names_the_accepting_request(
        self, tmp_path: Any
    ) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        runner = uuid4()
        job = store.record_superseded(
            correlation_id=uuid4(),
            command={"scope": "full"},
            superseded_by_sha=SHA_D,
            superseded_by_correlation_id=runner,
        )
        assert len(job.errors) == 1
        assert str(runner) in job.errors[0]
        assert SHA_D in job.errors[0]
        assert "contains this command's ref" not in job.errors[0]

    def test_supersede_dev_queue_end_to_end_leaves_none_queued(
        self, tmp_path: Any
    ) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        announced: list[ModelSupersession] = []
        a, b, c = _cmd(SHA_A), _cmd(SHA_B), _cmd(SHA_C)
        consumer = _consumer(
            store,
            ancestry_resolver=_nothing_contains_anything,
            branch_lanes=DEV,
            on_superseded=announced.append,
        )
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [
                _message(a, 290),
                _message(b, 291),
                _message(c, 292),
            ]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, reason = consumer.poll_and_accept()

        assert reason is None
        assert accepted is not None
        assert accepted.correlation_id == c.correlation_id
        for cmd in (a, b):
            record = store.load(cmd.correlation_id)
            assert record is not None
            assert record.status == "superseded"
            assert record.superseded_by_sha == SHA_C
            assert record.superseded_by_correlation_id == c.correlation_id
            assert len(record.errors) == 1
            assert str(c.correlation_id) in record.errors[0]
        assert [s.superseded.command.correlation_id for s in announced] == [
            a.correlation_id,
            b.correlation_id,
        ]


class TestDevAcceptsOnlyMergedRefs:
    def test_dev_accepts_only_merged_refs_refuses_a_pr_head(
        self, tmp_path: Any
    ) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        consumer = _merged_only_consumer(store, contains=lambda ref, tip: False)
        pr_head = _cmd(SHA_A, requested_by="operator-manual")
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [_message(pr_head, 5)]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, reason = consumer.poll_and_accept()

        assert accepted is None
        assert reason == EnumRejectionReason.NOT_A_MERGED_REF.value
        assert [n.reason for n in vars(consumer)["notices"]] == [
            EnumRejectionReason.NOT_A_MERGED_REF
        ]
        assert store.load(pr_head.correlation_id) is None, "nothing is built"

    def test_dev_accepts_only_merged_refs_asks_the_tracking_branch(
        self, tmp_path: Any
    ) -> None:
        asked: list[tuple[str, str]] = []

        def _contains(ref: str, tip: str) -> bool | None:
            asked.append((ref, tip))
            return True

        store = JobStore(state_dir=tmp_path / "jobs")
        consumer = _merged_only_consumer(store, contains=_contains)
        merged = _cmd(SHA_A)
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [_message(merged, 5)]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, reason = consumer.poll_and_accept()

        assert reason is None
        assert accepted is not None
        assert accepted.correlation_id == merged.correlation_id
        assert (SHA_A, TRACKING) in asked

    def test_dev_accepts_only_merged_refs_refuses_a_declared_open_pr(
        self, tmp_path: Any
    ) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        consumer = _merged_only_consumer(store, contains=lambda ref, tip: False)
        declared = _cmd(
            SHA_A,
            requested_by="operator-manual",
            rollback=ModelRollbackDeclaration(actor="slot", reason="build PR"),
            open_pr=f"omnibase_infra#5564@{SHA_A}",
        )
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [_message(declared, 5)]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, reason = consumer.poll_and_accept()

        assert accepted is None
        assert reason == EnumRejectionReason.NOT_A_MERGED_REF.value

    def test_dev_accepts_only_merged_refs_builds_when_the_clone_cannot_say(
        self, tmp_path: Any
    ) -> None:
        """Missing facts build, as the lineage fence's sixth rule has it."""
        store = JobStore(state_dir=tmp_path / "jobs")
        consumer = _merged_only_consumer(store, contains=lambda ref, tip: None)
        cmd = _cmd(SHA_A)
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [_message(cmd, 5)]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, reason = consumer.poll_and_accept()

        assert reason is None
        assert accepted is not None

    def test_dev_accepts_only_merged_refs_leaves_an_unnamed_lane_alone(
        self, tmp_path: Any
    ) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        consumer = _merged_only_consumer(store, contains=lambda ref, tip: False)
        consumer.branch_lanes = frozenset()
        cmd = _cmd(SHA_A)
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [_message(cmd, 5)]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, reason = consumer.poll_and_accept()

        assert reason is None
        assert accepted is not None


# ── the consumer path ────────────────────────────────────────────────────────
def _message(cmd: ModelRebuildRequested, offset: int) -> SimpleNamespace:
    payload = cmd.model_dump(mode="json")
    payload["_signature"] = "a" * 64
    return SimpleNamespace(value=payload, topic=TOPIC, partition=0, offset=offset)


def _consumer(store: JobStore, **attrs: Any) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = store
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    for key, value in attrs.items():
        setattr(consumer, key, value)
    return consumer


def _merged_only_consumer(store: JobStore, *, contains: Any) -> DeployConsumer:
    consumer = _consumer(
        store,
        ancestry_resolver=contains,
        branch_lanes=DEV,
        tracking_ref=TRACKING,
    )
    notices: list[ModelRejectionNotice] = []
    vars(consumer)["notices"] = notices
    consumer.on_rejected = notices.append
    return consumer
