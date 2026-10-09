# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: a lag that grew is a stall only when the consumer committed nothing.

The compose-dev receipt for c593ef2f2 (run 37899849702, 2026-10-09) failed
``consumer_group_lag`` on ``live-events-writer`` alone: TOTAL-LAG 0 -> 1 against a
bound of 10000, the two reads 5.5 s apart. The group was healthy. It consumes
``onex.evt.platform.node-heartbeat.v1``, read live on the same lane at 09:36Z
advancing 636220 -> 636223 -> 636224 with LAG 0 at every read, so the second read
had caught one heartbeat between its produce and its commit.

Over a window that short, a rising total cannot tell one message in flight from
a stopped consumer. The committed offsets can: a stopped consumer had work
waiting at the first read and committed none of it by the second. These tests
pin both directions with the real ``rpk group describe`` layout.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from scripts.ci.lab_pass_receipt import (
    ModelBrokerAccess,
    check_consumer_group_lag,
    dump_lag_sample,
    load_lag_sample,
    read_group_lag,
    sample_group_lag,
)

pytestmark = pytest.mark.unit

ACCESS = ModelBrokerAccess(container="omnibase-infra-redpanda", brokers="redpanda:9092")
GROUP = "local.omnimarket-projections.live-events-writer.consume.v1"
HEARTBEAT = "onex.evt.platform.node-heartbeat.v1"
FIXTURE = Path(__file__).parent / "fixtures" / "omn17427_rpk_group_describe.json"


class _Runner:
    """Replies with one ``rpk group describe`` stdout per call, in order."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), 0, self.replies.pop(0), "")


def _describe(
    *, heartbeat_committed: int, heartbeat_end: int, routing: tuple[int, int]
) -> str:
    """The live-events-writer reply as the .201 broker printed it, trimmed to 3 rows."""
    hb_lag = heartbeat_end - heartbeat_committed
    rt_committed, rt_end = routing
    rt_lag = rt_end - rt_committed
    member = "omnimarket-projection-f4fb1dfb  omnimarket-projection  172.19.0.19"
    return (
        f"GROUP        {GROUP}\n"
        "COORDINATOR  0\n"
        "STATE        Stable\n"
        "BALANCER     roundrobin\n"
        "MEMBERS      1\n"
        f"TOTAL-LAG    {hb_lag + rt_lag}\n"
        "\n"
        "TOPIC                                        PARTITION  CURRENT-OFFSET  "
        "LOG-START-OFFSET  LOG-END-OFFSET  LAG   MEMBER-ID  CLIENT-ID  HOST\n"
        f"onex.evt.omnibase-infra.routing-decision.v1  0          {rt_committed}  "
        f"8826              {rt_end}  {rt_lag}  {member}\n"
        f"{HEARTBEAT}          0          {heartbeat_committed}  617629  "
        f"{heartbeat_end}  {hb_lag}  {member}\n"
        "onex.evt.platform.log-entry.v1               0          -               "
        f"0                 0               -     {member}\n"
    )


def _probe(first: str, second: str, *, max_lag: int = 10_000):
    sample = sample_group_lag(ACCESS, [GROUP], runner=_Runner(first))
    return check_consumer_group_lag(
        ACCESS, [GROUP], max_lag=max_lag, first_sample=sample, runner=_Runner(second)
    )


def test_one_heartbeat_in_flight_on_a_caught_up_group_is_not_a_stall() -> None:
    """RED on the 2026-10-09 receipt: 0 -> 1, nothing was waiting at the first read."""
    check = _probe(
        _describe(
            heartbeat_committed=636220, heartbeat_end=636220, routing=(29593, 29593)
        ),
        _describe(
            heartbeat_committed=636220, heartbeat_end=636221, routing=(29593, 29593)
        ),
    )
    assert check.ok is True, check.evidence
    assert "none growing" in check.evidence
    assert "max TOTAL-LAG 1 against bound 10000" in check.evidence


def test_a_consumer_that_committed_none_of_its_waiting_work_still_fails() -> None:
    """NEGATIVE CONTROL: the OMN-18851 shape, work waiting and the offset frozen."""
    check = _probe(
        _describe(
            heartbeat_committed=636000, heartbeat_end=636498, routing=(29593, 29593)
        ),
        _describe(
            heartbeat_committed=636000, heartbeat_end=636601, routing=(29593, 29593)
        ),
    )
    assert check.ok is False
    assert check.indeterminate is False
    assert "GROWING" in check.evidence
    assert "498->601" in check.evidence
    assert f"{HEARTBEAT}/0 committed 636000 with 498 waiting" in check.evidence


def test_one_frozen_partition_fails_while_another_advances() -> None:
    """NEGATIVE CONTROL: progress on one partition does not excuse another."""
    check = _probe(
        _describe(
            heartbeat_committed=636220, heartbeat_end=636225, routing=(29500, 29593)
        ),
        _describe(
            heartbeat_committed=636225, heartbeat_end=636231, routing=(29500, 29600)
        ),
    )
    assert check.ok is False
    assert "onex.evt.omnibase-infra.routing-decision.v1/0 committed 29500" in (
        check.evidence
    )


def test_a_backlog_being_committed_is_not_a_stall() -> None:
    """POSITIVE CONTROL: the total grew, and every waiting partition committed."""
    check = _probe(
        _describe(
            heartbeat_committed=636000, heartbeat_end=636100, routing=(29593, 29593)
        ),
        _describe(
            heartbeat_committed=636050, heartbeat_end=636180, routing=(29593, 29593)
        ),
    )
    assert check.ok is True, check.evidence


def test_the_bound_still_fails_a_caught_up_group_over_it() -> None:
    check = _probe(
        _describe(
            heartbeat_committed=636220, heartbeat_end=636220, routing=(29593, 29593)
        ),
        _describe(
            heartbeat_committed=636220, heartbeat_end=636221, routing=(29593, 29593)
        ),
        max_lag=0,
    )
    assert check.ok is False
    assert "over bound" in check.evidence


def test_growth_with_no_partition_table_is_judged_on_the_total_alone() -> None:
    """With no offsets to read, growth stays a failure and the evidence says why."""
    first = f"GROUP {GROUP}\nSTATE Stable\nTOTAL-LAG 0\n"
    second = f"GROUP {GROUP}\nSTATE Stable\nTOTAL-LAG 1\n"
    check = _probe(first, second)
    assert check.ok is False
    assert "0->1" in check.evidence
    assert "no partition offsets" in check.evidence


def test_the_sample_file_round_trips_the_partition_offsets(tmp_path: Path) -> None:
    reply = _describe(
        heartbeat_committed=636220, heartbeat_end=636221, routing=(29593, 29593)
    )
    sample = sample_group_lag(ACCESS, [GROUP], runner=_Runner(reply))
    path = tmp_path / "lag-sample.json"
    path.write_text(dump_lag_sample(sample), encoding="utf-8")
    assert load_lag_sample(path) == sample
    assert json.loads(path.read_text(encoding="utf-8"))[GROUP]["total_lag"] == 1


def test_the_recorded_broker_reply_parses_into_partition_offsets() -> None:
    """The live capture: a committed row, and rows with no commit printed as '-'."""
    reply = json.loads(FIXTURE.read_text(encoding="utf-8"))["response"]["stdout"]
    reading = read_group_lag(
        ACCESS, "agent-observability-postgres", runner=_Runner(reply)
    )
    assert reading.total_lag == 0
    assert reading.partitions is not None
    assert reading.partitions["onex.evt.omniclaude.agent-actions.v1/0"] == (19861, 0)
    assert reading.partitions["onex.evt.omniclaude.agent-status.v1/0"] == (None, 0)
    assert len(reading.partitions) == 7
