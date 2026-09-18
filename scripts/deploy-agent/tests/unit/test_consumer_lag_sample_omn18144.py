# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The consumer samples its own control-topic lag (OMN-18144).

The one test that matters here is the committed-versus-position one. Measuring
lag from the fetch position reports ZERO while commands wait, because
``poll_and_accept`` processes the first record of a batch and leaves the rest
buffered -- OMN-18613's off-by-a-batch, seen from the reader's side. A guard
told "nothing is queued" by a position-based lag would make exactly the
decision this ticket exists to stop it making.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from deploy_agent.consumer import DeployConsumer
from deploy_agent.job_state import JobStore
from deploy_agent.queue_depth import LagSampler
from kafka import KafkaConsumer, TopicPartition

pytestmark = pytest.mark.unit

_TP = TopicPartition("onex.cmd.deploy.rebuild-requested.v1", 0)


def _consumer(tmp_path, kafka: Any, sampler: LagSampler) -> DeployConsumer:  # type: ignore[no-untyped-def]
    """A DeployConsumer whose kafka client is a double, built without __init__.

    The real ``__init__`` constructs a ``KafkaConsumer`` against a live broker.
    What is under test is the sampling arithmetic, so the object is built
    field-by-field rather than by patching a broker into existence.
    """
    consumer = object.__new__(DeployConsumer)
    consumer.consumer = kafka
    consumer.job_store = JobStore(state_dir=tmp_path / "jobs")
    consumer.lag_sampler = sampler
    return consumer


def _kafka(*, position: int, highwater: int) -> MagicMock:
    # spec'd against the real client: a lag sampler that called a method
    # KafkaConsumer does not have would pass against a bare mock and raise on
    # the lab host, which is the one place it must not.
    kafka = MagicMock(spec=KafkaConsumer)
    kafka.assignment.return_value = {_TP}
    kafka.position.return_value = position
    kafka.highwater.return_value = highwater
    return kafka


class TestLagBasis:
    def test_lag_is_measured_from_the_committed_offset_not_the_fetch_position(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        """The 2026-09-18 shape: two buffered records the agent has not looked at.

        The consumer fetched through offset 295 and has committed through 293.
        Two commands are genuinely queued. A position-based read reports zero.
        """
        sampler = LagSampler()
        kafka = _kafka(position=295, highwater=295)
        consumer = _consumer(tmp_path, kafka, sampler)
        sampler.note_commit(_TP, 293)

        consumer._sample_lag()

        lag = sampler.latest()
        assert lag.value == 2, (
            "measured from the fetch position this reads 0 while two commands "
            "wait -- the guard would then start a wait it cannot finish"
        )
        assert lag.basis == "committed"

    def test_before_any_commit_the_position_is_used_and_named(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        sampler = LagSampler()
        consumer = _consumer(tmp_path, _kafka(position=290, highwater=293), sampler)

        consumer._sample_lag()

        lag = sampler.latest()
        assert lag.value == 3
        assert lag.basis == "position", (
            "a first-poll fallback may under-report by a batch, so the reader "
            "is told which basis produced the number"
        )

    def test_committing_through_a_record_moves_the_basis_forward(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        sampler = LagSampler()
        kafka = _kafka(position=295, highwater=295)
        consumer = _consumer(tmp_path, kafka, sampler)
        msg = MagicMock(topic=_TP.topic, partition=_TP.partition, offset=293)

        consumer._commit_through(msg)
        consumer._sample_lag()

        assert sampler.latest().value == 1
        assert sampler.committed(_TP) == 294

    def test_a_self_update_rewind_moves_the_basis_back(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A rewound record is queued again, and the lag must say so."""
        sampler = LagSampler()
        kafka = _kafka(position=295, highwater=295)
        consumer = _consumer(tmp_path, kafka, sampler)
        msg = MagicMock(topic=_TP.topic, partition=_TP.partition, offset=293)

        consumer._commit_through(msg)
        consumer._rewind_committed_offset_to(msg)
        consumer._sample_lag()

        assert sampler.latest().value == 2


class TestUnreadableLag:
    def test_no_assignment_is_unknown_not_zero(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        sampler = LagSampler()
        kafka = _kafka(position=0, highwater=0)
        kafka.assignment.return_value = set()
        _consumer(tmp_path, kafka, sampler)._sample_lag()
        assert sampler.latest().value is None
        assert "no partition assignment" in sampler.latest().reason

    def test_an_unknown_highwater_is_unknown_not_zero(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        sampler = LagSampler()
        kafka = _kafka(position=10, highwater=10)
        kafka.highwater.return_value = None
        _consumer(tmp_path, kafka, sampler)._sample_lag()
        assert sampler.latest().value is None

    def test_a_raising_client_is_unknown_and_does_not_propagate(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A sampler that could fail a poll would trade the agent's job for an
        observation of it."""
        sampler = LagSampler()
        kafka = _kafka(position=10, highwater=12)
        kafka.highwater.side_effect = RuntimeError("client closed")
        _consumer(tmp_path, kafka, sampler)._sample_lag()
        assert sampler.latest().value is None
        assert "client closed" in sampler.latest().reason

    def test_an_absent_sampler_is_a_no_op(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        consumer = _consumer(tmp_path, _kafka(position=1, highwater=2), LagSampler())
        consumer.lag_sampler = None
        consumer._sample_lag()  # must not raise


class TestSamplingHappensOnEveryPoll:
    def test_an_empty_poll_still_samples(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """ "Nothing arrived" is exactly when a reader needs to know whether the
        queue behind it is empty or three deep."""
        sampler = LagSampler()
        kafka = _kafka(position=295, highwater=298)
        kafka.poll.return_value = {}
        consumer = _consumer(tmp_path, kafka, sampler)
        sampler.note_commit(_TP, 295)

        cmd, reason = consumer.poll_and_accept()

        assert (cmd, reason) == (None, None)
        assert sampler.latest().value == 3
