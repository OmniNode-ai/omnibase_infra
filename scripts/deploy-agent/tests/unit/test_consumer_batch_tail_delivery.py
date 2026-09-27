# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A command fetched behind the head of a poll batch is still delivered (OMN-19259).

THE LOSS
--------

``poll_and_accept`` processes the first record of a poll batch and returns.
kafka-python has by then moved the fetch position past EVERY record the poll
returned, and it keeps no copy of the ones nobody processed: the next ``poll()``
starts at the position. OMN-18613 bounded the commit to the processed record,
so a group rejoin or a re-exec would read the tail again -- but the live process
never did, and the next record it processed committed past the tail for good.

Replayed against a real single-node broker with the pinned kafka-python 2.3.2:
a stability-test command refused ``lane_not_allowed`` and a dev command in one
producer batch left the dev command undelivered, with no job record, no
refusal and no quarantine record.

THE DOUBLE
----------

``KafkaPythonFetchDouble`` models exactly the property the loss depends on:
``poll()`` returns up to ``max_poll_records`` from the position and advances the
position past all of them, ``seek()`` moves it, ``commit()`` records offsets and
does not move it. ``test_the_double_drops_the_tail_without_a_seek`` is its
positive control: without a seek, the double loses the tail just as the broker
replay did, so a green test below is the fix working and not the double being
lenient.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import EnumRuntimeLane
from kafka import TopicPartition

TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
TP0 = TopicPartition(TOPIC, 0)
TP1 = TopicPartition(TOPIC, 1)

SHA_A = "a24338d2e126a85c345a510c2ce916d6b2517f6d"
SHA_B = "0edf5c9145876dbe22f6caf7a06b507c5d0fc7d3"


class KafkaPythonFetchDouble:
    """The fetch-position behaviour of ``kafka.KafkaConsumer`` that the loss needs."""

    def __init__(
        self, log: dict[TopicPartition, list[Any]], *, max_poll_records: int = 500
    ) -> None:
        self.log = log
        self.max_poll_records = max_poll_records
        self.position = {
            tp: (records[0].offset if records else 0) for tp, records in log.items()
        }
        self.committed: dict[TopicPartition, int] = {}

    def poll(self, timeout_ms: int = 0) -> dict[TopicPartition, list[Any]]:
        returned: dict[TopicPartition, list[Any]] = {}
        for tp, records in self.log.items():
            batch = [r for r in records if r.offset >= self.position[tp]]
            batch = batch[: self.max_poll_records]
            if batch:
                returned[tp] = batch
                # kafka-python advances past every record it hands back.
                self.position[tp] = batch[-1].offset + 1
        return returned

    def seek(self, tp: TopicPartition, offset: int) -> None:
        self.position[tp] = offset

    def commit(self, offsets: dict[TopicPartition, Any]) -> None:
        for tp, meta in offsets.items():
            self.committed[tp] = meta.offset

    def assignment(self) -> set[TopicPartition]:
        return set(self.log)


def _record(
    correlation_id: str,
    offset: int,
    *,
    partition: int = 0,
    lane: str = "dev",
    scope: str = "full",
    git_ref: str = SHA_A,
) -> SimpleNamespace:
    payload = {
        "correlation_id": correlation_id,
        "git_ref": git_ref,
        "requested_by": "gha/omnibase_infra/pr-1",
        "scope": scope,
        "runtime_lane": lane,
        "build_source": "workspace",
        "services": [],
        "_signature": "a" * 64,
    }
    return SimpleNamespace(
        value=payload, topic=TOPIC, partition=partition, offset=offset, key=None
    )


def _consumer(fetch: KafkaPythonFetchDouble) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = fetch
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = False
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    return consumer


def _drain(consumer: DeployConsumer, polls: int) -> list[tuple[str | None, str | None]]:
    """Run ``polls`` iterations of the agent loop; (accepted id, refusal) each."""
    outcomes: list[tuple[str | None, str | None]] = []
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        for _ in range(polls):
            cmd, reason = consumer.poll_and_accept()
            outcomes.append(
                (str(cmd.correlation_id) if cmd is not None else None, reason)
            )
    return outcomes


def _accepted_ids(consumer: DeployConsumer) -> list[str]:
    return [
        str(call.kwargs["correlation_id"])
        for call in consumer.job_store.accept.call_args_list
    ]


REFUSED = "11111111-1111-4111-8111-111111111111"
FIRST = "22222222-2222-4222-8222-222222222222"
SECOND = "33333333-3333-4333-8333-333333333333"
THIRD = "44444444-4444-4444-8444-444444444444"


@pytest.mark.unit
def test_the_double_drops_the_tail_without_a_seek() -> None:
    """Positive control: the double reproduces the loss when nothing seeks back.

    Poll once, never seek, and the second record is not returned by any later
    poll. That is the kafka-python property the tests below depend on.
    """
    fetch = KafkaPythonFetchDouble(
        {TP0: [_record(REFUSED, 10, lane="stability-test"), _record(FIRST, 11)]}
    )
    first = fetch.poll()
    assert [r.offset for r in first[TP0]] == [10, 11]
    assert fetch.poll() == {}


@pytest.mark.unit
def test_command_behind_a_lane_not_allowed_head_becomes_a_job() -> None:
    """AC1: the measured S1 shape. The dev command behind a refused head runs."""
    fetch = KafkaPythonFetchDouble(
        {TP0: [_record(REFUSED, 10, lane="stability-test"), _record(FIRST, 11)]}
    )
    consumer = _consumer(fetch)

    outcomes = _drain(consumer, polls=3)

    assert outcomes[0] == (None, "lane_not_allowed")
    assert outcomes[1] == (FIRST, None)
    assert outcomes[2] == (None, None)
    assert _accepted_ids(consumer) == [FIRST]
    assert fetch.committed[TP0] == 12


@pytest.mark.unit
def test_command_behind_an_accepted_head_is_delivered_on_the_next_poll() -> None:
    """AC2 (accepted_head): two dev commands in one batch both become jobs.

    No rejoin and no re-exec happen here; the only thing between the two polls
    is the agent loop calling ``poll_and_accept`` again.
    """
    fetch = KafkaPythonFetchDouble({TP0: [_record(FIRST, 20), _record(SECOND, 21)]})
    consumer = _consumer(fetch)

    outcomes = _drain(consumer, polls=3)

    assert outcomes == [(FIRST, None), (SECOND, None), (None, None)]
    assert _accepted_ids(consumer) == [FIRST, SECOND]
    assert fetch.committed[TP0] == 22


@pytest.mark.unit
def test_refused_head_commits_its_own_record_and_no_further() -> None:
    """AC3: a refusal commits past itself only, so it never stalls the tail."""
    fetch = KafkaPythonFetchDouble(
        {TP0: [_record(REFUSED, 30, lane="stability-test"), _record(FIRST, 31)]}
    )
    consumer = _consumer(fetch)

    outcomes = _drain(consumer, polls=1)

    assert outcomes == [(None, "lane_not_allowed")]
    assert fetch.committed[TP0] == 31
    assert fetch.position[TP0] == 31


@pytest.mark.unit
def test_lookahead_break_record_is_handled_by_the_head_path() -> None:
    """AC4 (lookahead_break): the record the coalescing scan stops on is not lost.

    With a resolver injected the accepted head scans the batch behind it. A
    stability-test record ends the group (the scan never refuses anything
    itself), so it must reach the head path on the next poll and be refused
    there, and the dev command behind it must still become a job.
    """
    fetch = KafkaPythonFetchDouble(
        {
            TP0: [
                _record(FIRST, 40),
                _record(REFUSED, 41, lane="stability-test"),
                _record(SECOND, 42, git_ref=SHA_B),
            ]
        }
    )
    consumer = _consumer(fetch)
    consumer.ancestry_resolver = lambda earlier, later: True

    outcomes = _drain(consumer, polls=4)

    assert outcomes == [
        (FIRST, None),
        (None, "lane_not_allowed"),
        (SECOND, None),
        (None, None),
    ]
    assert _accepted_ids(consumer) == [FIRST, SECOND]


@pytest.mark.unit
def test_coalesced_group_resumes_after_the_runner() -> None:
    """A folded group commits and resumes past its runner, not past the batch."""
    fetch = KafkaPythonFetchDouble(
        {
            TP0: [
                _record(FIRST, 50),
                _record(SECOND, 51, git_ref=SHA_B),
                _record(THIRD, 52, scope="runtime", git_ref=SHA_B),
            ]
        }
    )
    consumer = _consumer(fetch)
    consumer.ancestry_resolver = lambda earlier, later: True

    outcomes = _drain(consumer, polls=3)

    # FIRST folds into SECOND; THIRD is a different scope, ends the group, and
    # must still be delivered.
    assert outcomes == [(SECOND, None), (THIRD, None), (None, None)]
    superseded = consumer.job_store.record_superseded.call_args_list
    assert [call.kwargs["correlation_id"] for call in superseded] == [UUID(FIRST)]


@pytest.mark.unit
def test_records_on_another_partition_are_fetched_again() -> None:
    """A multi-partition poll processes one record; the other partition is kept."""
    fetch = KafkaPythonFetchDouble(
        {
            TP0: [_record(FIRST, 60)],
            TP1: [_record(SECOND, 5, partition=1)],
        }
    )
    consumer = _consumer(fetch)

    outcomes = _drain(consumer, polls=3)

    assert sorted(cid for cid, _ in outcomes if cid is not None) == [FIRST, SECOND]
    assert fetch.committed == {TP0: 61, TP1: 6}
