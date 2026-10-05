# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Cover queue-depth guards, observed backlog boundaries and probe failures."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from omnibase_infra.cli import delegate_queue_depth

_TOPIC = "delegation-command"
_GROUP = "delegate-consumer"
_BROKER = "broker.example:9092"


def test_missing_command_topic_never_probes_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = AsyncMock(spec=delegate_queue_depth.consumer_group_topic_backlog)
    monkeypatch.setattr(delegate_queue_depth, "consumer_group_topic_backlog", probe)

    report = delegate_queue_depth.observe_delegate_queue_depth(
        bus="kafka", broker=_BROKER, command_topic="", consumer_groups=(_GROUP,)
    )

    assert report.records_ahead is None
    assert report.backlog_records is None
    assert report.consumer_group == ""
    assert report.unresolved_reason == "the run resolved no command topic to probe"
    probe.assert_not_called()


@pytest.mark.parametrize(("backlog", "records_ahead"), [(0, 0), (1, 0), (8, 7)])
def test_observed_backlog_excludes_caller_and_uses_first_consumer(
    monkeypatch: pytest.MonkeyPatch, backlog: int, records_ahead: int
) -> None:
    probe = AsyncMock(
        spec=delegate_queue_depth.consumer_group_topic_backlog, return_value=backlog
    )
    monkeypatch.setattr(delegate_queue_depth, "consumer_group_topic_backlog", probe)

    report = delegate_queue_depth.observe_delegate_queue_depth(
        bus="kafka",
        broker=_BROKER,
        command_topic=_TOPIC,
        consumer_groups=(_GROUP, "another-consumer"),
    )

    probe.assert_awaited_once_with(
        topic=_TOPIC, consumer_group=_GROUP, bootstrap_servers=_BROKER
    )
    assert report.consumer_group == _GROUP
    assert report.backlog_records == backlog
    assert report.records_ahead == records_ahead
    assert report.unresolved_reason == ""


@pytest.mark.parametrize("message", ["broker refused", "x" * 400])
def test_failed_lag_query_names_exception_and_bounds_reason(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    probe = AsyncMock(
        spec=delegate_queue_depth.consumer_group_topic_backlog,
        side_effect=RuntimeError(message),
    )
    monkeypatch.setattr(delegate_queue_depth, "consumer_group_topic_backlog", probe)

    report = delegate_queue_depth.observe_delegate_queue_depth(
        bus="kafka", broker=_BROKER, command_topic=_TOPIC, consumer_groups=(_GROUP,)
    )

    probe.assert_awaited_once()
    assert report.records_ahead is None
    assert report.backlog_records is None
    assert report.consumer_group == _GROUP
    assert (
        report.unresolved_reason
        == f"the lag query failed: RuntimeError: {message}"[:300]
    )


def test_timed_out_probe_is_cancelled_and_returns_unresolved_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled: list[bool] = []

    async def blocked_probe(**_: str) -> int:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
        return 0

    monkeypatch.setattr(
        delegate_queue_depth, "consumer_group_topic_backlog", blocked_probe
    )
    report = delegate_queue_depth.observe_delegate_queue_depth(
        bus="kafka",
        broker=_BROKER,
        command_topic=_TOPIC,
        consumer_groups=(_GROUP,),
        probe_seconds=0.01,
    )

    assert cancelled == [True]
    assert report.records_ahead is None
    assert report.backlog_records is None
    assert report.consumer_group == _GROUP
    assert report.unresolved_reason == (
        "the broker did not answer a lag query within 0.01s"
    )
