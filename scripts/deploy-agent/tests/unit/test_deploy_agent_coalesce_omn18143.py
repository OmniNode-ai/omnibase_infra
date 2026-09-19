# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The agent runs the newest queued rebuild and records the ones it replaced (OMN-18143).

THE MEASURED COST
-----------------
At 2026-09-19T02:58Z the .201 dev agent was staging job ``52e430b2`` against
``f8475f9b``, sixteen commits behind ``origin/dev``, with the job before it on
an older sha still, at a rolling mean service time of 1799.8 s per job. Forty
runtime-affecting merges in one night means forty full rebuilds in commit
order, each recreating the runtime family for a sha a later commit already
contains, while the actual dev head waits at the back.

WHAT IS PINNED HERE
-------------------
Three properties, and the second and third are what make the first safe:

* the NEWEST foldable command in a fetched batch runs, once;
* every command it replaces gets a durable terminal record and a terminal
  event that says ``superseded`` -- never a failure and never silence;
* every condition under which folding would LOSE work refuses to fold, and
  each refusal falls back to the pre-change behaviour of running every
  command in order.

The refusal table is the substance. A coalescer that folds the wrong pair does
not slow a lane down, it drops a deploy and reports success.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest
from deploy_agent.coalesce import (
    EnumCoalesceRefusal,
    GitAncestryResolver,
    ModelQueuedCommand,
    ModelSupersession,
    plan_coalesce,
)
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    EnumRejectionReason,
    EnumRuntimeLane,
    ModelRebuildRejected,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.job_state import JobState, JobStore
from deploy_agent.queue_depth import ModelControlTopicLag, compute_queue_snapshot
from kafka import TopicPartition

pytestmark = pytest.mark.unit

TOPIC = "onex.cmd.deploy.rebuild-requested.v1"

#: Three shas on one line of history. ``SHA_A`` is the oldest.
SHA_A = "f8475f9b" + "0" * 32
SHA_B = "52e430b2" + "1" * 32
SHA_C = "a5d95fa2" + "2" * 32
#: A sha on no shared line of history with the three above.
SHA_OFF_LINE = "deadbeef" + "3" * 32


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


def _always_contained(earlier: str, later: str) -> bool | None:
    """Every pair is an ancestor pair. Isolates the static half of the rules."""
    return True


class TestTheRuleTable:
    def test_the_newest_of_a_chain_runs_and_the_rest_are_superseded(self) -> None:
        a, b, c = _cmd(SHA_A), _cmd(SHA_B), _cmd(SHA_C)
        plan = plan_coalesce(
            [_queued(a, 10), _queued(b, 11), _queued(c, 12)],
            contains=_always_contained,
        )
        assert plan.runner.command.correlation_id == c.correlation_id
        assert plan.superseded_count == 2
        assert [s.superseded.command.correlation_id for s in plan.superseded] == [
            a.correlation_id,
            b.correlation_id,
        ]
        assert {s.superseded_by_sha for s in plan.superseded} == {SHA_C}
        assert plan.stop_reason is None
        assert plan.examined == 2

    def test_one_command_alone_supersedes_nothing(self) -> None:
        a = _cmd(SHA_A)
        plan = plan_coalesce([_queued(a, 10)], contains=_always_contained)
        assert plan.runner.command.correlation_id == a.correlation_id
        assert plan.superseded == ()
        assert plan.examined == 0

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            (
                {"runtime_lane": "stability-test"},
                EnumCoalesceRefusal.DIFFERENT_LANE,
            ),
            ({"scope": "core"}, EnumCoalesceRefusal.NOT_FULL_SCOPE),
            (
                {"services": ["omninode-runtime"]},
                EnumCoalesceRefusal.SERVICES_DIFFER,
            ),
            ({"build_source": "release"}, EnumCoalesceRefusal.BUILD_SOURCE_DIFFERS),
            (
                {"image_digest": "sha256:" + "c" * 64},
                EnumCoalesceRefusal.PINNED_IMAGE,
            ),
            ({"git_ref": "origin/dev"}, EnumCoalesceRefusal.REF_NOT_A_SHA),
        ],
    )
    def test_each_difference_ends_the_group_with_its_own_reason(
        self, overrides: dict[str, Any], expected: EnumCoalesceRefusal
    ) -> None:
        """A reason per refusal, because they send a reader to different places."""
        head = _cmd(SHA_A)
        candidate = _cmd(SHA_B, **overrides)
        plan = plan_coalesce(
            [_queued(head, 10), _queued(candidate, 11)], contains=_always_contained
        )
        assert plan.superseded == ()
        assert plan.runner.command.correlation_id == head.correlation_id
        assert plan.stop_reason is expected

    def test_a_prod_command_can_never_be_folded(self) -> None:
        """Not by a lane check, but because prod must pin a digest.

        ``ModelRebuildRequested`` refuses a prod request with no
        ``image_digest``, and a digest is a ``PINNED_IMAGE`` refusal, so the
        prod case is closed by two independent rules rather than by one lane
        comparison somebody could later widen.
        """
        head = _cmd(SHA_A, runtime_lane="prod", image_digest="sha256:" + "d" * 64)
        candidate = _cmd(SHA_B, runtime_lane="prod", image_digest="sha256:" + "e" * 64)
        plan = plan_coalesce(
            [_queued(head, 10), _queued(candidate, 11)], contains=_always_contained
        )
        assert plan.superseded == ()
        assert plan.stop_reason is EnumCoalesceRefusal.PINNED_IMAGE

        with pytest.raises(ValueError, match="requires image_digest"):
            _cmd(SHA_B, runtime_lane="prod")

    def test_an_unproven_ancestry_does_not_fold(self) -> None:
        """An unanswerable clone falls back to running both, in order."""
        plan = plan_coalesce(
            [_queued(_cmd(SHA_A), 10), _queued(_cmd(SHA_B), 11)],
            contains=lambda earlier, later: None,
        )
        assert plan.superseded == ()
        assert plan.stop_reason is EnumCoalesceRefusal.ANCESTRY_UNPROVEN

    def test_a_sha_that_does_not_contain_the_earlier_one_does_not_fold(self) -> None:
        """The whole safety argument: folding here would DROP the earlier change."""
        plan = plan_coalesce(
            [_queued(_cmd(SHA_A), 10), _queued(_cmd(SHA_OFF_LINE), 11)],
            contains=lambda earlier, later: False,
        )
        assert plan.superseded == ()
        assert plan.stop_reason is EnumCoalesceRefusal.NOT_A_DESCENDANT

    def test_the_group_is_a_prefix_and_never_reorders_around_a_refusal(self) -> None:
        """A foldable command BEHIND a refused one stays queued.

        Folding it would run a later command ahead of an earlier one the scan
        declined to fold, which is a reordering of the control topic and not a
        coalesce.
        """
        a, b = _cmd(SHA_A), _cmd(SHA_B)
        other_lane = _cmd(SHA_B, runtime_lane="stability-test")
        d = _cmd(SHA_C)
        plan = plan_coalesce(
            [_queued(a, 10), _queued(b, 11), _queued(other_lane, 12), _queued(d, 13)],
            contains=_always_contained,
        )
        assert plan.runner.command.correlation_id == b.correlation_id
        assert [s.superseded.command.correlation_id for s in plan.superseded] == [
            a.correlation_id
        ]
        assert plan.stop_reason is EnumCoalesceRefusal.DIFFERENT_LANE
        assert plan.examined == 2, (
            "the scan stops at the refusal; it must not report having looked "
            "at a command it never reached"
        )

    def test_ancestry_is_asked_of_consecutive_members_not_of_the_head(self) -> None:
        """Containment is transitive, so a chain costs one comparison each."""
        asked: list[tuple[str, str]] = []

        def _record(earlier: str, later: str) -> bool:
            asked.append((earlier, later))
            return True

        plan = plan_coalesce(
            [
                _queued(_cmd(SHA_A), 10),
                _queued(_cmd(SHA_B), 11),
                _queued(_cmd(SHA_C), 12),
            ],
            contains=_record,
        )
        assert plan.superseded_count == 2
        assert asked == [(SHA_A, SHA_B), (SHA_B, SHA_C)]

    def test_the_journal_line_names_the_position_the_count_and_the_reason(
        self,
    ) -> None:
        plan = plan_coalesce(
            [
                _queued(_cmd(SHA_A), 290),
                _queued(_cmd(SHA_B), 291),
                _queued(_cmd(SHA_C, scope="core"), 292),
            ],
            contains=_always_contained,
        )
        line = plan.journal_line()
        assert "0:291" in line
        assert "superseding 1 command(s)" in line
        assert EnumCoalesceRefusal.NOT_FULL_SCOPE.value in line

    def test_an_empty_queue_is_a_programming_error_not_an_empty_plan(self) -> None:
        with pytest.raises(ValueError, match="at least the command"):
            plan_coalesce([], contains=_always_contained)


class TestSupersessionModel:
    def test_a_supersession_names_an_exact_sha(self) -> None:
        with pytest.raises(ValueError, match="40-character lowercase"):
            ModelSupersession(
                superseded=_queued(_cmd(SHA_A), 10),
                superseded_by_sha="52e430b2",
                superseded_by_correlation_id=uuid4(),
            )

    def test_a_command_cannot_supersede_itself(self) -> None:
        cmd = _cmd(SHA_A)
        with pytest.raises(ValueError, match="cannot supersede itself"):
            ModelSupersession(
                superseded=_queued(cmd, 10),
                superseded_by_sha=SHA_B,
                superseded_by_correlation_id=cmd.correlation_id,
            )


class TestTheTerminalEvent:
    def test_a_superseded_rejection_carries_both_fields(self) -> None:
        cid, runner = uuid4(), uuid4()
        event = ModelRebuildRejected(
            correlation_id=cid,
            reason=EnumRejectionReason.SUPERSEDED,
            scope=Scope.FULL,
            superseded_by_sha=SHA_C,
            superseded_by_correlation_id=runner,
        )
        assert event.to_wire() == {
            "correlation_id": str(cid),
            "reason": "superseded",
            "scope": "full",
            "superseded_by_sha": SHA_C,
            "superseded_by_correlation_id": str(runner),
        }

    def test_a_superseded_rejection_that_names_nothing_is_refused(self) -> None:
        """AC6: a supersession that cannot say what replaced it is a drop."""
        with pytest.raises(ValueError, match="disagrees with the supersession"):
            ModelRebuildRejected(
                correlation_id=uuid4(),
                reason=EnumRejectionReason.SUPERSEDED,
                scope=Scope.FULL,
            )

    def test_a_timeout_shaped_rejection_may_not_claim_a_replacement(self) -> None:
        """The other half of AC6's "distinguishably from a timeout"."""
        with pytest.raises(ValueError, match="disagrees with the supersession"):
            ModelRebuildRejected(
                correlation_id=uuid4(),
                reason=EnumRejectionReason.BUSY,
                scope=Scope.FULL,
                superseded_by_sha=SHA_C,
                superseded_by_correlation_id=uuid4(),
            )

    def test_an_ordinary_rejection_is_byte_identical_to_the_shape_it_replaces(
        self,
    ) -> None:
        """A consumer predating this model must parse a `busy` event unchanged."""
        cid = uuid4()
        event = ModelRebuildRejected(
            correlation_id=cid, reason=EnumRejectionReason.BUSY, scope=Scope.CORE
        )
        assert event.to_wire() == {
            "correlation_id": str(cid),
            "reason": "busy",
            "scope": "core",
        }


class TestTheSupersededJobRecord:
    def test_the_record_is_born_terminal_and_names_its_replacement(
        self, tmp_path: Any
    ) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        cid, runner = uuid4(), uuid4()
        job = store.record_superseded(
            correlation_id=cid,
            command={"scope": "full"},
            superseded_by_sha=SHA_C,
            superseded_by_correlation_id=runner,
        )
        assert job.status == "superseded"
        assert job.completed_at is not None
        assert job.superseded_by_sha == SHA_C
        assert job.superseded_by_correlation_id == runner
        assert job.result_publish_pending is True, (
            "the terminal event AC6 asks for is owed to the bus, and the "
            "retry loop is what makes it survive a broker being away"
        )
        # Read back off disk: the record has to survive the process that wrote
        # it, because the retry loop may run in a replacement image.
        assert store.load(cid) == job

    def test_a_superseded_record_does_not_block_the_next_command(
        self, tmp_path: Any
    ) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        store.record_superseded(
            correlation_id=uuid4(),
            command={"scope": "full"},
            superseded_by_sha=SHA_C,
            superseded_by_correlation_id=uuid4(),
        )
        assert store.has_active_job() is False
        assert store.load_active() is None

    def test_a_superseded_record_that_names_nothing_cannot_be_constructed(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="must name what replaced it"):
            JobState(correlation_id=uuid4(), command={}, status="superseded")

    def test_a_running_record_may_not_claim_a_replacement(self) -> None:
        with pytest.raises(ValueError, match="only a superseded record may"):
            JobState(
                correlation_id=uuid4(),
                command={},
                status="in_progress",
                superseded_by_sha=SHA_C,
                superseded_by_correlation_id=uuid4(),
            )

    def test_the_running_record_counts_what_it_superseded(self, tmp_path: Any) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        replaced = [uuid4(), uuid4()]
        job = store.accept(
            uuid4(), {"scope": "full"}, superseded_correlation_ids=replaced
        )
        assert job.superseded_count == 2
        assert job.superseded_correlation_ids == replaced

    def test_a_count_that_does_not_match_its_ids_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            JobState(
                correlation_id=uuid4(),
                command={},
                superseded_count=3,
                superseded_correlation_ids=[uuid4()],
            )


class TestTheServiceTimeSample:
    def test_a_superseded_job_does_not_shrink_the_mean_service_time(
        self, tmp_path: Any
    ) -> None:
        """The bound OMN-18144 derives from this mean must not collapse.

        A superseded record is written and completed in the same millisecond.
        Averaging thirty-nine of them against one real 30-minute rebuild would
        report a mean of roughly 46 seconds, and the derived wait bound --
        commands ahead multiplied by mean service time -- would then be short
        by more than an order of magnitude exactly when the queue is deepest.
        """
        store = JobStore(state_dir=tmp_path / "jobs")
        real = JobState(
            correlation_id=uuid4(),
            command={},
            status="success",
            accepted_at=_ts(0),
            completed_at=_ts(1800),
        )
        _persist(store, real)
        for _ in range(39):
            _persist(
                store,
                JobState(
                    correlation_id=uuid4(),
                    command={},
                    status="superseded",
                    accepted_at=_ts(0),
                    completed_at=_ts(0.05),
                    superseded_by_sha=SHA_C,
                    superseded_by_correlation_id=uuid4(),
                ),
            )
        snapshot = compute_queue_snapshot(
            store, ModelControlTopicLag(value=0, basis="committed")
        )
        assert snapshot.mean_service_time_seconds == pytest.approx(1800.0)
        assert snapshot.service_sample_size == 1
        assert snapshot.commands_ahead == 0, (
            "a superseded record is terminal; the agent owes it no work"
        )


def _ts(offset_seconds: float) -> Any:
    from datetime import UTC, datetime, timedelta

    return datetime(2026, 9, 19, 2, 58, tzinfo=UTC) + timedelta(seconds=offset_seconds)


def _persist(store: JobStore, job: JobState) -> None:
    store.state_dir.mkdir(parents=True, exist_ok=True)
    (store.state_dir / f"{job.correlation_id}.json").write_text(job.model_dump_json())


class TestTheAncestryResolver:
    def _resolver(self, results: dict[tuple[str, ...], int], *, now: Any = None) -> Any:
        calls: list[list[str]] = []

        def _run(argv: list[str], timeout: int) -> Any:
            calls.append(argv)
            for prefix, code in results.items():
                if tuple(argv[3:4]) == prefix:
                    return SimpleNamespace(returncode=code, stderr="", stdout="")
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        return GitAncestryResolver(
            "/clone", run=_run, clock=now or (lambda: 0.0)
        ), calls

    def test_exit_zero_is_contained_and_exit_one_is_not(self) -> None:
        resolver, _ = self._resolver({("merge-base",): 0})
        assert resolver(SHA_A, SHA_B) is True
        resolver, _ = self._resolver({("merge-base",): 1})
        assert resolver(SHA_A, SHA_B) is False

    def test_any_other_exit_status_is_unproven_not_a_refusal(self) -> None:
        """git documents exactly two statuses; a third is a failure to ANSWER."""
        resolver, _ = self._resolver({("merge-base",): 128})
        assert resolver(SHA_A, SHA_B) is None

    def test_a_missing_object_is_fetched_once_and_then_given_up_on(self) -> None:
        resolver, calls = self._resolver({("cat-file",): 1})
        assert resolver(SHA_A, SHA_B) is None
        assert resolver(SHA_B, SHA_C) is None
        fetches = [argv for argv in calls if "fetch" in argv]
        assert len(fetches) == 1, (
            "a batch of twelve commands against a clone that cannot see them "
            "must not become twelve network round trips"
        )

    def test_the_fetch_cooldown_expires_so_coalescing_does_not_die_with_it(
        self,
    ) -> None:
        """A once-ever flag would silently stop coalescing for the process life.

        The resolver outlives every scan, so the first scan meeting a sha the
        clone lacks would otherwise be the only one ever allowed to fetch.
        """
        now = [0.0]
        resolver, calls = self._resolver({("cat-file",): 1}, now=lambda: now[0])

        assert resolver(SHA_A, SHA_B) is None
        assert len([a for a in calls if "fetch" in a]) == 1

        now[0] = 30.0
        assert resolver(SHA_B, SHA_C) is None
        assert len([a for a in calls if "fetch" in a]) == 1, (
            "inside the cooldown a deep batch must not become one round trip "
            "per command"
        )

        now[0] = 1000.0
        assert resolver(SHA_A, SHA_C) is None
        assert len([a for a in calls if "fetch" in a]) == 2, (
            "past the cooldown the clone may be brought current again"
        )

    def test_a_raising_git_is_unproven(self) -> None:
        def _boom(argv: list[str], timeout: int) -> Any:
            raise TimeoutError("git took too long")

        assert GitAncestryResolver("/clone", run=_boom)(SHA_A, SHA_B) is None


# ── the consumer path ────────────────────────────────────────────────────────
def _message(
    cmd: ModelRebuildRequested, offset: int, partition: int = 0
) -> SimpleNamespace:
    payload = cmd.model_dump(mode="json")
    payload["_signature"] = "a" * 64
    return SimpleNamespace(
        value=payload, topic=TOPIC, partition=partition, offset=offset
    )


def _consumer(store: JobStore, **attrs: Any) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = store
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    for key, value in attrs.items():
        setattr(consumer, key, value)
    return consumer


def _committed(consumer: DeployConsumer) -> list[int]:
    offsets: list[int] = []
    for call in consumer.consumer.commit.call_args_list:
        offsets.extend(meta.offset for meta in call.args[0].values())
    return offsets


class TestTheConsumerFoldsTheBatch:
    def test_the_newest_runs_and_the_older_two_are_recorded_and_announced(
        self, tmp_path: Any
    ) -> None:
        """The end-to-end shape, on the batch the poll already fetched."""
        store = JobStore(state_dir=tmp_path / "jobs")
        announced: list[ModelSupersession] = []
        a, b, c = _cmd(SHA_A), _cmd(SHA_B), _cmd(SHA_C)
        consumer = _consumer(
            store,
            ancestry_resolver=_always_contained,
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
        assert accepted.correlation_id == c.correlation_id, (
            "the agent must run the newest queued sha, not the oldest"
        )

        for cmd in (a, b):
            record = store.load(cmd.correlation_id)
            assert record is not None
            assert record.status == "superseded"
            assert record.superseded_by_sha == SHA_C
            assert record.superseded_by_correlation_id == c.correlation_id

        runner_record = store.load(c.correlation_id)
        assert runner_record is not None
        assert runner_record.status == "accepted"
        assert runner_record.superseded_count == 2
        assert set(runner_record.superseded_correlation_ids) == {
            a.correlation_id,
            b.correlation_id,
        }

        assert [s.superseded.command.correlation_id for s in announced] == [
            a.correlation_id,
            b.correlation_id,
        ]
        assert _committed(consumer) == [293], (
            "one commit, through the runner, which is at or past every "
            "superseded record's offset"
        )

    def test_the_self_update_boundary_rewinds_to_the_runner_not_the_head(
        self, tmp_path: Any
    ) -> None:
        """A re-exec must re-read the command that is going to run."""
        store = JobStore(state_dir=tmp_path / "jobs")
        rewound: list[int] = []

        def _hook(rewind: Any) -> None:
            rewind()

        consumer = _consumer(
            store, ancestry_resolver=_always_contained, self_update_hook=_hook
        )
        consumer._rewind_committed_offset_to = lambda msg: rewound.append(msg.offset)  # type: ignore[method-assign]
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [
                _message(_cmd(SHA_A), 290),
                _message(_cmd(SHA_C), 291),
            ]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            consumer.poll_and_accept()

        assert rewound == [291]

    def test_with_no_resolver_nothing_is_folded(self, tmp_path: Any) -> None:
        """The pre-change behaviour, reached by the same code path."""
        store = JobStore(state_dir=tmp_path / "jobs")
        a, c = _cmd(SHA_A), _cmd(SHA_C)
        consumer = _consumer(store)
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [
                _message(a, 290),
                _message(c, 291),
            ]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, _ = consumer.poll_and_accept()

        assert accepted is not None
        assert accepted.correlation_id == a.correlation_id
        assert store.load(c.correlation_id) is None
        assert _committed(consumer) == [291]

    def test_a_batch_spanning_partitions_is_not_folded(self, tmp_path: Any) -> None:
        """Kafka orders within a partition and nowhere else."""
        store = JobStore(state_dir=tmp_path / "jobs")
        a, c = _cmd(SHA_A), _cmd(SHA_C)
        consumer = _consumer(store, ancestry_resolver=_always_contained)
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [_message(a, 290)],
            TopicPartition(TOPIC, 1): [_message(c, 5, partition=1)],
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, _ = consumer.poll_and_accept()

        assert accepted is not None
        assert accepted.correlation_id == a.correlation_id
        assert store.load(c.correlation_id) is None

    def test_an_unverifiable_look_ahead_record_is_left_completely_alone(
        self, tmp_path: Any
    ) -> None:
        """The scan looks; it never refuses a command on the head path's behalf.

        A record whose signature does not verify must be quarantined, rejected
        and committed past by the head path when the agent REACHES it -- not
        by a scan that was only supposed to decide whether to fold it.
        """
        store = JobStore(state_dir=tmp_path / "jobs")
        a, bad = _cmd(SHA_A), _cmd(SHA_C)
        consumer = _consumer(store, ancestry_resolver=_always_contained)
        head_message, bad_message = _message(a, 290), _message(bad, 291)
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [head_message, bad_message]
        }

        def _verify(payload: dict[str, Any]) -> bool:
            return payload["correlation_id"] != str(bad.correlation_id)

        with patch("deploy_agent.consumer.verify_command", side_effect=_verify):
            accepted, _ = consumer.poll_and_accept()

        assert accepted is not None
        assert accepted.correlation_id == a.correlation_id
        assert store.load(bad.correlation_id) is None
        assert _committed(consumer) == [291], (
            "the commit is bounded by the head, so the unverifiable record is "
            "still there to be refused properly on the next poll"
        )
        assert (
            not list((store.state_dir.parent / "quarantine").glob("*"))
            if (store.state_dir.parent / "quarantine").exists()
            else True
        )

    def test_an_out_of_lane_look_ahead_record_ends_the_group(
        self, tmp_path: Any
    ) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        a = _cmd(SHA_A)
        other = _cmd(SHA_C, runtime_lane="stability-test")
        consumer = _consumer(store, ancestry_resolver=_always_contained)
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [
                _message(a, 290),
                _message(other, 291),
            ]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, _ = consumer.poll_and_accept()

        assert accepted is not None
        assert accepted.correlation_id == a.correlation_id
        assert store.load(other.correlation_id) is None

    def test_a_failing_announcement_does_not_lose_the_durable_record(
        self, tmp_path: Any
    ) -> None:
        """The record is written first, and it carries the publish debt."""
        store = JobStore(state_dir=tmp_path / "jobs")
        a, c = _cmd(SHA_A), _cmd(SHA_C)

        def _explode(supersession: ModelSupersession) -> None:
            raise RuntimeError("broker unreachable")

        consumer = _consumer(
            store, ancestry_resolver=_always_contained, on_superseded=_explode
        )
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [
                _message(a, 290),
                _message(c, 291),
            ]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, _ = consumer.poll_and_accept()

        assert accepted is not None
        assert accepted.correlation_id == c.correlation_id
        record = store.load(a.correlation_id)
        assert record is not None
        assert record.status == "superseded"
        assert record.result_publish_pending is True

    def test_a_duplicate_head_still_refuses_before_any_folding_happens(
        self, tmp_path: Any
    ) -> None:
        """A redelivered superseded head is refused, not folded a second time.

        This is the crash-recovery shape: records were written and the process
        died before the runner's offset was committed, so the whole batch is
        redelivered.
        """
        store = JobStore(state_dir=tmp_path / "jobs")
        a, c = _cmd(SHA_A), _cmd(SHA_C)
        store.record_superseded(
            correlation_id=a.correlation_id,
            command={"scope": "full"},
            superseded_by_sha=SHA_C,
            superseded_by_correlation_id=c.correlation_id,
        )
        announced: list[ModelSupersession] = []
        consumer = _consumer(
            store,
            ancestry_resolver=_always_contained,
            on_superseded=announced.append,
        )
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [
                _message(a, 290),
                _message(c, 291),
            ]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, reason = consumer.poll_and_accept()

        assert accepted is None
        assert reason == "duplicate"
        assert announced == []


class TestTheJobEndpointServesTheSupersession:
    def test_the_payload_carries_both_halves(self, tmp_path: Any) -> None:
        """The CI guard reads this route; a field it cannot see does not exist."""
        from deploy_agent.health import _job_handler

        store = JobStore(state_dir=tmp_path / "jobs")
        cid, runner = uuid4(), uuid4()
        store.record_superseded(
            correlation_id=cid,
            command={"scope": "full"},
            superseded_by_sha=SHA_C,
            superseded_by_correlation_id=runner,
        )
        request = SimpleNamespace(
            app={"job_store": store}, match_info={"correlation_id": str(cid)}
        )
        response = _run_async(_job_handler(request))
        import json

        body = json.loads(response.text)
        assert body["status"] == "superseded"
        assert body["superseded_by_sha"] == SHA_C
        assert body["superseded_by_correlation_id"] == str(runner)


def _run_async(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


def _uuid(value: str) -> UUID:
    return UUID(value)
