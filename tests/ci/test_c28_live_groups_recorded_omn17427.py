# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay C28's live-groups read as dev-201's Postgres answered it.

C28 run 38968109234 graded INDETERMINATE when ``live_groups()`` ran
``select distinct consumer_group ... where window_start > ...`` over
``omninode_internal.consumer_flow_windows`` (48M rows, 32 GB) and the parallel
seq scan outran the 120 s ``docker exec`` timeout. The recording holds both
queries' stdout, plan and wall time from the same host minutes apart, captured
with the command ``live_groups()`` builds.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_constants import (
    LIVE_WINDOW_SQL,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_lane import (
    ConsumerFlowLane,
)
from scripts.ci import c28_consumer_flow_probe as script

pytestmark = pytest.mark.unit

_PSQL_TIMEOUT_MS = 120_000


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/c28_live_groups_dev201_omn17427.json"
)
def test_recorded_index_backed_read_is_the_sql_both_collectors_run(
    recorded_response,
) -> None:
    recorded = recorded_response["response"]["index_backed"]["sql"]
    assert recorded == LIVE_WINDOW_SQL
    assert recorded == script.LIVE_WINDOW_SQL


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/c28_live_groups_dev201_omn17427.json"
)
def test_recorded_index_backed_read_returns_the_same_groups_as_the_scan(
    recorded_response,
) -> None:
    response = recorded_response["response"]
    stdout = "\n".join(response["index_backed"]["stdout_lines"]) + "\n"
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    lane = ConsumerFlowLane(
        docker="docker", base_url="http://projection.test", runner=runner
    )

    live = lane.live_groups()
    assert len(calls) == 1
    assert f'-c "{LIVE_WINDOW_SQL}"' in calls[0][5]
    assert live == sorted(set(response["whole_table"]["stdout_lines"]))
    assert len(live) == 533


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/c28_live_groups_dev201_omn17427.json"
)
def test_recorded_plans_show_the_index_walk_replacing_the_seq_scan(
    recorded_response,
) -> None:
    response = recorded_response["response"]
    new_plan = "\n".join(response["index_backed"]["explain"])
    old_plan = "\n".join(response["whole_table"]["explain"])
    assert "Parallel Seq Scan on consumer_flow_windows" in old_plan
    assert "Seq Scan" not in new_plan
    assert "idx_consumer_flow_windows_group_time" in new_plan
    new_ms = response["index_backed"]["elapsed_ms"]
    old_ms = response["whole_table"]["elapsed_ms"]
    assert new_ms * 10 < old_ms
    assert new_ms * 10 < _PSQL_TIMEOUT_MS
