# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A record buffered behind an accepted one must not be committed past (OMN-18613).

THE MEASURED LOSS
-----------------

On 2026-09-17 one ``poll()`` returned offsets 262 and 263 together, because both
were already on ``onex.cmd.deploy.rebuild-requested.v1``. ``poll_and_accept``
processes the FIRST record only and returns; the remaining records stay in the
client's buffer. The agent accepted 262 (``b046ce97``, ``gha/omnimarket/pr-2620``)
and committed with a bare ``self.consumer.commit()``.

In kafka-python a bare ``commit()`` commits the consumer's POSITION for every
assigned partition -- which is past every record already FETCHED, not past the
one record processed. So accepting 262 committed 264, past the still-buffered
263 (``8ba320c4``, ``gha/omnimarket/pr-2622``). Job 262 then failed and
``self_update[boundary=post_terminal]`` re-exec'd the process image, discarding
the buffer. The replacement resumed from 264 and 263 was never delivered again:
no job record, no acceptance line, no rejection line, no quarantine record.

The file already knows the distinction. ``_rewind_committed_offset_to`` uses an
explicit per-partition ``commit({tp: OffsetAndMetadata(msg.offset, ...)})``
precisely because relying on a record "simply being uncommitted is not enough".
That reasoning was applied on the rewind path and not on the accept path.

WHAT THESE TESTS PIN
--------------------

Every commit ``_process_message`` makes is bounded by the record it is about --
``msg.offset + 1``, explicitly, per partition. That is what leaves a buffered
record uncommitted, which is what makes the ``post_terminal`` re-exec harmless:
the replacement image re-reads from the committed offset and finds it there.

The REFUSAL paths keep advancing past their own record, and that is deliberate:
a refused command that re-delivers forever stalls every command behind it, which
is the stall the bare commit was originally chosen to avoid. Bounding the commit
to the refused record's own offset keeps that property and drops the overreach.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import EnumRuntimeLane
from kafka import TopicPartition

TOPIC = "onex.cmd.deploy.rebuild-requested.v1"

#: The two commands of the measured 2026-09-17 loss, at their real offsets.
ACCEPTED_CORRELATION = "b046ce97-f2f8-43b0-9302-c4579a4b06de"
ACCEPTED_OFFSET = 262
LOST_CORRELATION = "8ba320c4-fdd6-4970-82aa-f0b3c252fe0c"
LOST_OFFSET = 263


def _consumer() -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = False
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    # The pre-accept self-update boundary is an explicit no-op: an absent
    # attribute would be swallowed by the boundary's own error rail and read
    # as a pass.
    consumer.self_update_hook = lambda rewind: None
    return consumer


def _message(correlation_id: str, offset: int, **overrides: Any) -> SimpleNamespace:
    payload = {
        "correlation_id": correlation_id,
        "git_ref": "a24338d2e126a85c345a510c2ce916d6b2517f6d",
        "requested_by": "gha/omnimarket/pr-2620",
        "scope": "full",
        "runtime_lane": "dev",
        "build_source": "workspace",
        "services": [],
        "_signature": "a" * 64,
    }
    payload.update(overrides)
    return SimpleNamespace(value=payload, topic=TOPIC, partition=0, offset=offset)


def _committed_offsets(consumer: DeployConsumer) -> list[int]:
    """Every offset any commit call asked the broker to record.

    A bare ``commit()`` passes no argument at all, so it contributes nothing
    here -- which is exactly the shape these tests refuse.
    """
    offsets: list[int] = []
    for call in consumer.consumer.commit.call_args_list:
        assert call.args, (
            "commit() was called with no argument. A bare commit commits the "
            "consumer POSITION for every assigned partition, which is past "
            "every record already fetched -- including records buffered behind "
            "this one that have not been processed."
        )
        mapping = call.args[0]
        offsets.extend(meta.offset for meta in mapping.values())
    return offsets


@pytest.mark.unit
def test_accept_commits_through_the_accepted_record_only() -> None:
    """AC1: accepting commits ``msg.offset + 1``, never the fetch position."""
    consumer = _consumer()

    with patch("deploy_agent.consumer.verify_command", return_value=True):
        cmd, reason = consumer._process_message(
            _message(ACCEPTED_CORRELATION, ACCEPTED_OFFSET)
        )

    assert reason is None
    assert cmd is not None
    assert _committed_offsets(consumer) == [ACCEPTED_OFFSET + 1]


@pytest.mark.unit
def test_accept_commits_only_the_accepted_partition() -> None:
    """The commit names the record's own partition explicitly."""
    consumer = _consumer()

    with patch("deploy_agent.consumer.verify_command", return_value=True):
        consumer._process_message(_message(ACCEPTED_CORRELATION, ACCEPTED_OFFSET))

    mapping = consumer.consumer.commit.call_args.args[0]
    assert list(mapping) == [TopicPartition(TOPIC, 0)]


@pytest.mark.unit
def test_the_measured_loss_leaves_the_buffered_record_uncommitted() -> None:
    """AC4: the 2026-09-17 pair, at their real offsets.

    Accepting 262 must not commit past 263. This is the whole defect: the
    committed offset reached 264 while 263 sat unprocessed in the client
    buffer, and the ``post_terminal`` re-exec then threw the buffer away.
    """
    consumer = _consumer()

    with patch("deploy_agent.consumer.verify_command", return_value=True):
        consumer._process_message(_message(ACCEPTED_CORRELATION, ACCEPTED_OFFSET))

    committed = _committed_offsets(consumer)
    assert LOST_OFFSET + 1 not in committed, (
        f"committed {committed}: offset {LOST_OFFSET} carries correlation "
        f"{LOST_CORRELATION} and was never processed. Committing past it is "
        "how that command was lost with no trace."
    )
    assert committed == [ACCEPTED_OFFSET + 1]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("setup", "expected_reason"),
    [
        (
            lambda c: setattr(c.job_store, "has_active_job", Mock(return_value=True)),
            "busy",
        ),
        (
            lambda c: setattr(c.job_store, "is_duplicate", Mock(return_value=True)),
            "duplicate",
        ),
        (
            lambda c: setattr(
                c, "allowed_lanes", frozenset({EnumRuntimeLane.STABILITY_TEST})
            ),
            "lane_not_allowed",
        ),
    ],
)
def test_refusal_paths_still_advance_past_their_own_record(
    setup: Any, expected_reason: str
) -> None:
    """AC5, fail-closed direction: a refused command must NOT re-deliver forever.

    Advancing past a refused record is deliberate -- the consumer's own step 4
    comment says re-reading it forever "would stall every command behind it".
    What changes is the BOUND: the commit covers this record and no further.
    """
    consumer = _consumer()
    setup(consumer)

    with patch("deploy_agent.consumer.verify_command", return_value=True):
        cmd, reason = consumer._process_message(
            _message(ACCEPTED_CORRELATION, ACCEPTED_OFFSET)
        )

    assert cmd is None
    assert reason == expected_reason
    assert _committed_offsets(consumer) == [ACCEPTED_OFFSET + 1]


@pytest.mark.unit
def test_invalid_signature_advances_past_its_own_record_only() -> None:
    consumer = _consumer()

    with patch("deploy_agent.consumer.verify_command", return_value=False):
        cmd, reason = consumer._process_message(
            _message(ACCEPTED_CORRELATION, ACCEPTED_OFFSET)
        )

    assert cmd is None
    assert reason == "invalid_signature"
    assert _committed_offsets(consumer) == [ACCEPTED_OFFSET + 1]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("setup", "expected_reason"),
    [
        (
            lambda c: setattr(c.job_store, "has_active_job", Mock(return_value=True)),
            "busy",
        ),
        (
            lambda c: setattr(c.job_store, "is_duplicate", Mock(return_value=True)),
            "duplicate",
        ),
        (
            lambda c: setattr(
                c, "allowed_lanes", frozenset({EnumRuntimeLane.STABILITY_TEST})
            ),
            "lane_not_allowed",
        ),
    ],
)
def test_every_refusal_names_the_correlation_id(
    setup: Any, expected_reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    """AC3: a command that advances an offset without running says which one.

    The path this ticket found was invisible precisely because it logged
    nothing. Every path that CAN advance an offset must name the correlation.
    """
    consumer = _consumer()
    setup(consumer)

    with caplog.at_level(logging.INFO, logger="deploy_agent.consumer"):
        with patch("deploy_agent.consumer.verify_command", return_value=True):
            consumer._process_message(_message(ACCEPTED_CORRELATION, ACCEPTED_OFFSET))

    assert any(
        ACCEPTED_CORRELATION in record.getMessage() for record in caplog.records
    ), f"{expected_reason} advanced an offset without naming the correlation id"
