# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overdue dev merge build decision, driven by the real record of omnibase_infra#6191 (OMN-26316).

omnibase_infra#6191 merged at 17:46:52Z on 2026-11-23. The agent's next dev job was accepted
at 19:00:23Z, 73 minutes later, and nothing asked for a build in between. The fixture is the
agent's own job records and the watcher's merge time. Each test states the clock it reads at.
"""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import yaml

from omnibase_infra.handlers.handler_dev_merge_build_overdue import (
    HandlerDevMergeBuildOverdue,
    request_id_for,
)
from omnibase_infra.models.model_overdue_dev_merge_build_decision import (
    ModelOverdueDevMergeBuildDecision,
)
from omnibase_infra.models.model_overdue_dev_merge_build_request import (
    ModelOverdueDevMergeBuildRequest,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE = (
    _ROOT
    / "tests/fixtures/overdue_dev_merge_build_omn26316/merge_6191_job_records.json"
)
_NODE = _ROOT / "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
_WINDOW = 3600
_SIBLING_SHA = "f" * 40


def _at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(UTC)


def _record() -> Any:
    return json.loads(_FIXTURE.read_text())


def _request(
    now: str,
    *,
    drop: tuple[str, ...] = (),
    extra: tuple[dict[str, str], ...] = (),
    runtime_class: str | None = None,
    jobs_override: list[dict[str, str]] | None = None,
) -> ModelOverdueDevMergeBuildRequest:
    record = _record()
    merge = record["merge"]
    assert isinstance(merge, dict)
    jobs = jobs_override if jobs_override is not None else record["jobs"]
    assert isinstance(jobs, list)
    kept = [
        {k: v for k, v in job.items() if k != "completed_at"}
        for job in jobs
        if job["job_id"][:8] not in drop
    ] + list(extra)
    return ModelOverdueDevMergeBuildRequest.model_validate(
        {
            "repository": merge["repository"],
            "merge_sha": merge["merge_sha"],
            "merged_at": merge["merged_at"],
            "runtime_class": runtime_class or merge["runtime_class"],
            "now": now,
            "window_seconds": _WINDOW,
            "jobs": kept,
        }
    )


def _decide(
    request: ModelOverdueDevMergeBuildRequest,
) -> ModelOverdueDevMergeBuildDecision:
    return HandlerDevMergeBuildOverdue().handle(request)


def test_the_merge_is_inside_its_window_one_second_before_the_hour_ends() -> None:
    decision = _decide(_request("2026-11-23T18:46:51Z", drop=("2098b18c",)))
    assert (decision.outcome, decision.reason) == ("WAIT", "within_window")
    assert decision.due_at == _at("2026-11-23T18:46:52Z")
    assert decision.request_id is None


def test_the_merge_is_requested_the_second_its_hour_ends_with_no_dev_build() -> None:
    """The real gap: 18:46:52Z to 19:00:23Z, when the agent's next dev job was accepted."""
    decision = _decide(_request("2026-11-23T18:46:52Z", drop=("2098b18c",)))
    assert (decision.outcome, decision.reason) == ("REQUEST", "overdue_no_build")
    assert decision.request_id == request_id_for(
        "omnibase_infra", "a27c0c218555c93421803e9f6dc4dd95c28b3da5"
    )


def test_a_job_of_another_lane_after_the_merge_does_not_satisfy_the_dev_lane() -> None:
    """The agent runs several lanes from one state directory; only a dev job carries a dev merge."""
    other = {
        "job_id": "aaaaaaaa-0000-4000-8000-000000000005",
        "runtime_lane": "other",
        "accepted_at": "2026-11-23T18:00:21Z",
        "status": "success",
        "git_ref": "origin/dev",
    }
    decision = _decide(
        _request("2026-11-23T18:50:00Z", drop=("2098b18c",), extra=(other,))
    )
    assert decision.outcome == "REQUEST"


def test_the_dev_job_accepted_before_the_merge_never_carries_it() -> None:
    """82d5fd01 was accepted at 17:39:09Z, seven minutes before the merge."""
    decision = _decide(_request("2026-11-23T18:50:00Z", drop=("2098b18c",)))
    assert decision.job_id is None
    assert decision.reason == "overdue_no_build"


def test_the_agent_job_accepted_at_19_00_is_in_flight_and_nothing_more_is_requested() -> (
    None
):
    running = [
        {**job, "status": "in_progress"}
        if job["job_id"].startswith("2098b18c")
        else job
        for job in _record()["jobs"]
    ]
    decision = _decide(_request("2026-11-23T19:05:00Z", jobs_override=running))
    assert (decision.outcome, decision.reason) == ("WAIT", "job_in_flight")
    assert decision.job_id == UUID("2098b18c-b02e-4721-97ec-43d744ab1120")


def test_the_completed_agent_job_satisfies_the_merge() -> None:
    decision = _decide(_request("2026-11-23T19:14:00Z"))
    assert (decision.outcome, decision.reason) == ("SATISFIED", "built_by_job")
    assert decision.job_id == UUID("2098b18c-b02e-4721-97ec-43d744ab1120")


def test_a_non_runtime_merge_is_never_owed_a_build() -> None:
    decision = _decide(
        _request(
            "2026-11-24T03:00:00Z", drop=("2098b18c",), runtime_class="NON_RUNTIME"
        )
    )
    assert (decision.outcome, decision.reason) == ("SATISFIED", "non_runtime")


def test_a_failed_job_that_carries_the_merge_is_left_to_the_failed_head_gate() -> None:
    failed = {
        "job_id": "aaaaaaaa-0000-4000-8000-000000000001",
        "runtime_lane": "dev",
        "accepted_at": "2026-11-23T18:20:00Z",
        "status": "failed",
        "git_ref": "origin/dev",
    }
    decision = _decide(
        _request("2026-11-23T19:00:00Z", drop=("2098b18c",), extra=(failed,))
    )
    assert (decision.outcome, decision.reason) == ("WAIT", "failed_head")
    assert decision.job_id == UUID(failed["job_id"])


def test_a_superseded_job_and_a_job_pinned_to_another_sha_carry_nothing() -> None:
    superseded = {
        "job_id": "aaaaaaaa-0000-4000-8000-000000000002",
        "runtime_lane": "dev",
        "accepted_at": "2026-11-23T18:10:00Z",
        "status": "superseded",
        "git_ref": "origin/dev",
    }
    pinned = {
        "job_id": "aaaaaaaa-0000-4000-8000-000000000003",
        "runtime_lane": "dev",
        "accepted_at": "2026-11-23T18:20:00Z",
        "status": "success",
        "git_ref": _SIBLING_SHA,
    }
    decision = _decide(
        _request("2026-11-23T19:00:00Z", drop=("2098b18c",), extra=(superseded, pinned))
    )
    assert decision.outcome == "REQUEST"


def test_a_job_pinned_to_the_merge_sha_itself_satisfies_it() -> None:
    pinned = {
        "job_id": "aaaaaaaa-0000-4000-8000-000000000004",
        "runtime_lane": "dev",
        "accepted_at": "2026-11-23T18:20:00Z",
        "status": "success",
        "git_ref": "a27c0c218555c93421803e9f6dc4dd95c28b3da5",
    }
    decision = _decide(
        _request("2026-11-23T19:00:00Z", drop=("2098b18c",), extra=(pinned,))
    )
    assert decision.outcome == "SATISFIED"


def test_the_request_id_is_one_value_per_sha_whatever_the_clock_says() -> None:
    ids = {
        _decide(_request(stamp, drop=("2098b18c",))).request_id
        for stamp in (
            "2026-11-23T18:46:52Z",
            "2026-11-23T18:55:00Z",
            "2026-11-24T02:00:00Z",
        )
    }
    assert len(ids) == 1
    assert ids != {request_id_for("omnibase_infra", _SIBLING_SHA)}


def test_the_window_is_the_callers_not_a_constant() -> None:
    request = _request("2026-11-23T18:46:52Z", drop=("2098b18c",)).model_copy(
        update={"window_seconds": _WINDOW * 2}
    )
    assert _decide(request).due_at == _at("2026-11-23T17:46:52Z") + timedelta(
        seconds=_WINDOW * 2
    )
    assert _decide(request).outcome == "WAIT"


def test_the_node_contract_binds_the_operation_to_the_handler() -> None:
    contract = yaml.safe_load(_NODE.read_text())
    bound = {
        entry["operation"]: entry["handler"]
        for entry in contract["handler_routing"]["handlers"]
    }
    entry = bound["request_overdue_dev_merge_build"]
    assert entry["name"] == "HandlerDevMergeBuildOverdue"
    assert (
        getattr(importlib.import_module(entry["module"]), entry["name"])
        is HandlerDevMergeBuildOverdue
    )
    assert contract["settings"]["dev_merge_build_window_seconds"] == _WINDOW
