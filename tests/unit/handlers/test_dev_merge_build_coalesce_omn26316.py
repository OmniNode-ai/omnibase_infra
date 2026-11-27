# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev merge build requests are coalesced inside a declared window (OMN-26316).

THE RECORD
----------
On 2026-11-25 dev was rebuilt at 07:09Z, 07:23Z and 07:36Z. The hourly idle-restore job
``943bd20f`` (``origin/dev``, accepted 07:00:50Z, lineage unread) carried nothing, so the
overdue check asked for builds for merges ``3a527226`` (job ``d844bdf1``, accepted 07:15:07Z),
``c0622a1d`` (``7d9991bf``, 07:29:00Z) and ``5be4cfa4`` (``7db4beaf``, 07:42:13Z). Each of those
jobs pinned the dev head ``f553b1d6`` (a descendant of all three merges), a ref that is neither
``origin/dev`` nor a merge sha, so a build asked for one merge never carried the next.

The window is the caller's, read from the node contract's ``dev_merge_build_coalesce_seconds``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_dev_merge_build_overdue import (
    HandlerDevMergeBuildOverdue,
)
from omnibase_infra.models.model_overdue_dev_merge_build_decision import (
    ModelOverdueDevMergeBuildDecision,
)
from omnibase_infra.models.model_overdue_dev_merge_build_request import (
    ModelOverdueDevMergeBuildRequest,
)

pytestmark = pytest.mark.unit

_NODE = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
)
_COALESCE = yaml.safe_load(_NODE.read_text())["settings"][
    "dev_merge_build_coalesce_seconds"
]
_HEAD = "f" * 40
_MERGES = {"a": "a" * 40, "b": "b" * 40, "c": "c" * 40}


def _at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(UTC)


def _job(
    job_id: str,
    accepted_at: str,
    *,
    git_ref: str,
    requested_by: str,
    built_sha: str = "",
    status: str = "success",
) -> dict[str, str]:
    return {
        "job_id": job_id,
        "runtime_lane": "dev",
        "accepted_at": accepted_at,
        "status": status,
        "git_ref": git_ref,
        "requested_by": requested_by,
        "built_sha": built_sha,
    }


def _decide(
    key: str, merged_at: str, now: str, jobs: list[dict[str, str]]
) -> ModelOverdueDevMergeBuildDecision:
    return HandlerDevMergeBuildOverdue().handle(
        ModelOverdueDevMergeBuildRequest.model_validate(
            {
                "repository": "omnibase_infra",
                "merge_sha": _MERGES[key],
                "merged_at": merged_at,
                "runtime_class": "RUNTIME",
                "now": now,
                "window_seconds": 3600,
                "coalesce_seconds": _COALESCE,
                "jobs": jobs,
                "descendant_shas": [_HEAD],
            }
        )
    )


def test_three_merges_in_thirty_minutes_get_one_build() -> None:
    merged = {
        "a": "2026-11-25T05:00:00Z",
        "b": "2026-11-25T05:10:00Z",
        "c": "2026-11-25T05:30:00Z",
    }
    asked = _job(
        "00000000-0000-5000-8000-000000000001",
        "2026-11-25T06:00:05Z",
        git_ref=_HEAD,
        requested_by="agent/overdue-dev-merge-build",
        built_sha=_HEAD,
    )
    assert _decide("a", merged["a"], "2026-11-25T06:00:00Z", []).outcome == "REQUEST"
    later = [
        _decide(key, merged[key], "2026-11-25T06:31:00Z", [asked])
        for key in ("a", "b", "c")
    ]
    assert [(d.outcome, d.reason) for d in later] == [("SATISFIED", "built_by_job")] * 3


def test_a_merge_just_after_a_build_waits_for_the_next() -> None:
    hourly = _job(
        "00000000-0000-5000-8000-000000000002",
        "2026-11-25T06:00:50Z",
        git_ref="origin/dev",
        requested_by="agent/idle-restore",
    )
    inside = _decide("a", "2026-11-25T06:05:00Z", "2026-11-25T07:06:00Z", [])
    assert inside.outcome == "REQUEST"  # no job on record: the window alone
    waits = _decide("a", "2026-11-25T06:05:00Z", "2026-11-25T06:40:00Z", [hourly])
    assert (waits.outcome, waits.reason) == ("WAIT", "within_window")
    asked_again = _decide("a", "2026-11-25T05:50:00Z", "2026-11-25T06:51:00Z", [hourly])
    assert (asked_again.outcome, asked_again.reason) == ("WAIT", "coalesced")


def test_a_merge_fifty_minutes_before_the_hour_is_carried_within_the_hour() -> None:
    hourly = _job(
        "00000000-0000-5000-8000-000000000003",
        "2026-11-25T07:00:50Z",
        git_ref="origin/dev",
        requested_by="agent/idle-restore",
    )
    due = _decide("a", "2026-11-25T06:10:00Z", "2026-11-25T07:12:00Z", [hourly])
    assert (due.outcome, due.reason) == ("WAIT", "coalesced")
    assert due.request_id is None


def test_a_stability_follow_job_does_not_coalesce_a_request() -> None:
    follow = _job(
        "00000000-0000-5000-8000-000000000004",
        "2026-11-25T07:00:50Z",
        git_ref="origin/dev",
        requested_by="agent/stability-follow-tag",
    )
    decision = _decide("a", "2026-11-25T06:10:00Z", "2026-11-25T07:12:00Z", [follow])
    assert decision.outcome == "REQUEST"


def test_a_build_older_than_the_window_does_not_coalesce() -> None:
    old = _job(
        "00000000-0000-5000-8000-000000000005",
        "2026-11-25T05:00:00Z",
        git_ref="origin/dev",
        requested_by="agent/idle-restore",
    )
    decision = _decide("a", "2026-11-25T06:10:00Z", "2026-11-25T07:12:00Z", [old])
    assert decision.outcome == "REQUEST"
