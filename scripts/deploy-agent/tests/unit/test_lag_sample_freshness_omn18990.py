# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lag sample is only a count while it is current (OMN-18990).

THE SHAPE THESE PIN
-------------------
``LagSampler`` is a cache, written only from inside ``poll_and_accept``, and
the agent's run loop does not poll while a rebuild executes. So for the whole
20-40 minutes of a command the ``/queue`` route served the lag observed BEFORE
that rebuild started, as a confident integer, with nothing on the wire saying
how old it was. That window is exactly the window in which later merges queue.

Measured 2026-09-21, receipt artifact ``10632895303``
(``lab-pass-receipt-compose-dev-430ff3434cc3...``): ``commands_ahead=0`` at
09:18:50Z while two commands sat unconsumed, a 1560s wait bound derived from
that zero, and the probe gave up at 0h26m. The agent accepted the receipt's own
command at 10:01:09Z -- 42m21s after the probe began, 5m30s after the receipt
was written and its FAIL made permanent.

The property throughout: a sample whose age cannot be shown to be inside the
bound is UNREAD, and an unread queue is not an empty one. The positive control
is in ``TestAGenuinelyEmptyQueueStillReadsZero``, without which every case here
could be satisfied by reporting unknown always.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent.events import TOPIC_REBUILD_REQUESTED
from deploy_agent.job_state import JobState, JobStore
from deploy_agent.lag_refresher import BASIS, LagRefresher
from deploy_agent.queue_depth import (
    MAX_LAG_SAMPLE_AGE_SECONDS,
    LagSampler,
    ModelControlTopicLag,
    compute_queue_snapshot,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 9, 21, 9, 18, 50, tzinfo=UTC)


def _store(tmp_path) -> JobStore:  # type: ignore[no-untyped-def]
    return JobStore(state_dir=tmp_path / "jobs")


def _write(store: JobStore, *, status: str, completed_at: datetime | None) -> None:
    job = JobState(
        correlation_id=uuid4(),
        command={},
        status=status,  # type: ignore[arg-type]
        accepted_at=_NOW - timedelta(minutes=40),
        completed_at=completed_at,
    )
    store.state_dir.mkdir(parents=True, exist_ok=True)
    (store.state_dir / f"{job.correlation_id}.json").write_text(job.model_dump_json())


def _lag(value: int, *, age_seconds: float) -> ModelControlTopicLag:
    return ModelControlTopicLag(
        value=value,
        basis="committed",
        observed_at=_NOW - timedelta(seconds=age_seconds),
    )


class TestAValuedSampleMustSayWhenItWasTaken:
    """The structural half. A count with no time cannot be shown to be current."""

    def test_a_value_without_an_observation_time_is_refused(self) -> None:
        with pytest.raises(ValueError, match="observed"):
            ModelControlTopicLag(value=0, basis="committed")

    def test_an_unknown_needs_no_observation_time(self) -> None:
        """An absence is already an absence; dating it is not what makes it honest."""
        lag = ModelControlTopicLag.unknown("no highwater")
        assert lag.value is None
        assert lag.observed_at is None


class TestAStaleSampleIsUnreadNotEmpty:
    def test_a_sample_older_than_the_bound_reports_no_count(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The incident, reduced: a pre-rebuild zero served mid-rebuild.

        Fails on the pre-change tree at ``commands_ahead is None``, where the
        same inputs return 0 -- the value that sized a 1560s bound against a
        42-minute wait.
        """
        store = _store(tmp_path)
        snapshot = compute_queue_snapshot(store, _lag(0, age_seconds=2280.0), now=_NOW)
        assert snapshot.commands_ahead is None
        assert snapshot.lag_age_seconds == pytest.approx(2280.0)
        assert "2280s ago" in snapshot.lag_staleness_reason
        assert (
            "an unread queue, which is not an empty one"
            in snapshot.lag_staleness_reason
        )

    def test_a_stale_nonzero_sample_is_also_refused(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Staleness is about currency, not about the value being convenient."""
        store = _store(tmp_path)
        snapshot = compute_queue_snapshot(store, _lag(3, age_seconds=600.0), now=_NOW)
        assert snapshot.commands_ahead is None

    def test_the_bound_is_inclusive_and_one_second_past_it_is_not(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        store = _store(tmp_path)
        at_bound = compute_queue_snapshot(
            store, _lag(1, age_seconds=MAX_LAG_SAMPLE_AGE_SECONDS), now=_NOW
        )
        past_bound = compute_queue_snapshot(
            store, _lag(1, age_seconds=MAX_LAG_SAMPLE_AGE_SECONDS + 1), now=_NOW
        )
        assert at_bound.commands_ahead == 1
        assert past_bound.commands_ahead is None

    def test_exactly_one_reason_is_served_for_one_absence(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A reader must never have to decide which absence it is looking at."""
        store = _store(tmp_path)
        stale = compute_queue_snapshot(
            store, _lag(0, age_seconds=2280.0), now=_NOW
        ).to_payload()
        unreadable = compute_queue_snapshot(
            store, ModelControlTopicLag.unknown("no highwater"), now=_NOW
        ).to_payload()
        assert "2280s ago" in str(stale["control_topic_lag_reason"])
        assert unreadable["control_topic_lag_reason"] == "no highwater"


class TestTheWireCarriesTheSamplesOwnAge:
    def test_the_payload_dates_the_sample_not_the_snapshot(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The snapshot is built per request and is always fresh.

        A reader that had only the snapshot's ``observed_at`` saw a current
        timestamp beside a half-hour-old count, which is how the stale zero
        read as a live one.
        """
        store = _store(tmp_path)
        payload = compute_queue_snapshot(
            store, _lag(2, age_seconds=30.0), now=_NOW
        ).to_payload()
        assert payload["observed_at"] == _NOW.isoformat()
        assert payload["control_topic_lag_observed_at"] == (
            (_NOW - timedelta(seconds=30)).isoformat()
        )
        assert payload["control_topic_lag_age_seconds"] == pytest.approx(30.0)
        assert payload["commands_ahead"] == 2

    def test_an_unreadable_sample_carries_a_null_age(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        store = _store(tmp_path)
        payload = compute_queue_snapshot(
            store, ModelControlTopicLag.unknown("no highwater"), now=_NOW
        ).to_payload()
        assert payload["control_topic_lag_observed_at"] is None
        assert payload["control_topic_lag_age_seconds"] is None
        assert payload["commands_ahead"] is None


class TestAGenuinelyEmptyQueueStillReadsZero:
    """The positive control. Without it, "report unknown always" passes the rest."""

    def test_a_fresh_zero_on_an_idle_agent_is_still_zero(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        store = _store(tmp_path)
        _write(store, status="success", completed_at=_NOW - timedelta(minutes=10))
        snapshot = compute_queue_snapshot(store, _lag(0, age_seconds=5.0), now=_NOW)
        assert snapshot.commands_ahead == 0
        assert snapshot.lag_staleness_reason == ""

    def test_a_fresh_sample_still_sums_with_the_store_half(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        store = _store(tmp_path)
        _write(store, status="in_progress", completed_at=None)
        snapshot = compute_queue_snapshot(store, _lag(2, age_seconds=5.0), now=_NOW)
        assert snapshot.commands_ahead == 3


class _FakeConsumer:
    """A control-topic client that records every method reached on it."""

    def __init__(
        self,
        *,
        partitions: set[int] | None = None,
        end: dict[int, int] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.partitions = {0} if partitions is None else partitions
        self.end = end or {0: 10}
        self.raises = raises
        self.calls: list[str] = []
        self.closed = False

    def partitions_for_topic(self, topic: str) -> set[int]:
        self.calls.append("partitions_for_topic")
        if self.raises is not None:
            raise self.raises
        return self.partitions

    def end_offsets(self, topic_partitions: list[Any]) -> dict[Any, int]:
        self.calls.append("end_offsets")
        return {tp: self.end[tp.partition] for tp in topic_partitions}

    def close(self) -> None:
        self.closed = True


def _refresher(sampler: LagSampler, consumer: _FakeConsumer) -> LagRefresher:
    return LagRefresher(
        kafka_config=None,  # type: ignore[arg-type]
        sampler=sampler,
        topic=TOPIC_REBUILD_REQUESTED,
        consumer_factory=lambda: consumer,
    )


class TestTheRefresherKeepsTheSampleCurrentDuringAJob:
    def test_it_records_a_fresh_count_from_the_brokers_end_offset(self) -> None:
        """AC1, reduced to its arithmetic.

        On the pre-change tree there is no refresher at all, so the sampler
        still holds whatever the last poll left and this assertion reads the
        pre-job value.
        """
        from kafka import TopicPartition

        sampler = LagSampler()
        sampler.note_commit(TopicPartition(TOPIC_REBUILD_REQUESTED, 0), 8)
        consumer = _FakeConsumer(end={0: 10})
        before = datetime.now(UTC)

        _refresher(sampler, consumer).refresh()

        lag = sampler.latest()
        assert lag.value == 2, "two records past this agent's committed offset"
        assert lag.basis == BASIS
        assert lag.observed_at is not None and lag.observed_at >= before

    def test_a_sample_taken_mid_job_supersedes_the_pre_job_one(self) -> None:
        """The incident's own sequence: poll, then the topic grows, then a read."""
        from kafka import TopicPartition

        topic_partition = TopicPartition(TOPIC_REBUILD_REQUESTED, 0)
        sampler = LagSampler()
        sampler.record(_lag(0, age_seconds=0.0))
        sampler.note_commit(topic_partition, 8)
        consumer = _FakeConsumer(end={0: 10})

        _refresher(sampler, consumer).refresh()

        assert sampler.latest().value == 2, (
            "the two commands published while the rebuild ran must be visible "
            "before the rebuild ends, not after it"
        )

    def test_it_never_reaches_the_polling_consumers_verbs(self) -> None:
        """AC2. It observes; it cannot advance, assign or commit anything."""
        from kafka import TopicPartition

        sampler = LagSampler()
        sampler.note_commit(TopicPartition(TOPIC_REBUILD_REQUESTED, 0), 8)
        consumer = _FakeConsumer()

        _refresher(sampler, consumer).refresh()

        assert consumer.calls == ["partitions_for_topic", "end_offsets"], (
            "a refresher that could poll, seek or commit would be able to lose "
            "a command in order to make its own number smaller"
        )

    def test_it_takes_no_reference_to_the_agents_own_consumer(self) -> None:
        """AC2, structurally: there is no parameter through which to pass one."""
        import inspect

        parameters = set(inspect.signature(LagRefresher.__init__).parameters)
        assert "consumer" not in parameters
        assert "deploy_consumer" not in parameters


class TestTheRefresherReportsAnUnreadableSampleAsUnread:
    def test_a_broken_client_becomes_a_reason_not_a_zero(self) -> None:
        sampler = LagSampler()
        consumer = _FakeConsumer(raises=OSError("broker unreachable"))

        refresher = _refresher(sampler, consumer)
        refresher.refresh()

        lag = sampler.latest()
        assert lag.value is None
        assert "broker unreachable" in lag.reason
        assert consumer.closed, (
            "a client that failed is dropped so the next refresh rebuilds it, "
            "rather than failing forever against a dead connection"
        )

    def test_a_topic_with_no_partitions_becomes_a_reason(self) -> None:
        sampler = LagSampler()
        consumer = _FakeConsumer(partitions=set())

        _refresher(sampler, consumer).refresh()

        lag = sampler.latest()
        assert lag.value is None
        assert "no partitions" in lag.reason

    def test_no_committed_offset_becomes_a_reason_not_a_guess(self) -> None:
        """This process has nothing of its own to measure against yet."""
        sampler = LagSampler()
        consumer = _FakeConsumer(end={0: 10})

        _refresher(sampler, consumer).refresh()

        lag = sampler.latest()
        assert lag.value is None
        assert "committed no offset" in lag.reason

    def test_a_refresh_failure_does_not_raise_into_the_agent(self) -> None:
        """An observation is never worth the job it is observing."""
        sampler = LagSampler()
        consumer = _FakeConsumer(raises=RuntimeError("boom"))
        _refresher(sampler, consumer).refresh()  # must not raise
        assert sampler.latest().value is None
