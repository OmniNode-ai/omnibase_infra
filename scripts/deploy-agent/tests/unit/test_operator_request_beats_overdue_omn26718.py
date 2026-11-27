# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An operator-path request is admitted ahead of an automatic one and never dropped (OMN-26718).

THE RECORD
----------
2026-11-26 on h201: an operator-path rebuild request (signed command c94ae2a7, 18:38Z) was
deferred to the 19:00Z recreate window. The window admitted the automatic overdue job and the
dev lane's queue-supersede rule ("newest wins", OMN-26637) replaced the operator's request with
it, so the request someone made on purpose never ran. An operator-path request now outranks an
automatic one for the same window, and the job it replaces names it and says why.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from deploy_agent.coalesce import (
    EnumCoalesceRefusal,
    ModelCoalescePlan,
    ModelQueuedCommand,
    plan_coalesce,
)
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

OPERATOR_SHA = "c94ae2a7" + "1" * 32
HEAD = "f553b1d6" + "8" * 32
OVERDUE = "agent/overdue-dev-merge-build"


def _command(requested_by: str, sha: str, **extra: Any) -> ModelRebuildRequested:
    payload: dict[str, Any] = {
        "correlation_id": uuid4(),
        "requested_by": requested_by,
        "scope": "runtime",
        "runtime_lane": "dev",
        "build_source": "workspace",
        "git_ref": sha,
    }
    payload.update(extra)
    return ModelRebuildRequested.model_validate(payload)


def _operator() -> ModelRebuildRequested:
    return _command("operator-manual", OPERATOR_SHA)


def _overdue() -> ModelRebuildRequested:
    # The overdue build declares a rollback only to pass the lineage fence (OMN-26316).
    return _command(
        OVERDUE, HEAD, rollback={"actor": OVERDUE, "reason": "merge had no dev build"}
    )


def _plan(commands: list[ModelRebuildRequested]) -> ModelCoalescePlan:
    return plan_coalesce(
        [
            ModelQueuedCommand(command=c, partition=0, offset=10 + i)
            for i, c in enumerate(commands)
        ],
        contains=lambda earlier, later: True,
        supersede_lanes=frozenset({EnumRuntimeLane.DEV}),
    )


def test_operator_request_is_the_admitted_job_over_a_later_automatic_one() -> None:
    operator, overdue = _operator(), _overdue()
    plan = _plan([overdue, operator])
    assert plan.runner.command.correlation_id == operator.correlation_id
    assert [s.superseded.command.correlation_id for s in plan.superseded] == [
        overdue.correlation_id
    ]
    assert "operator-path request" in plan.superseded[0].reason


def test_operator_request_at_the_head_still_runs_alone() -> None:
    operator, overdue = _operator(), _overdue()
    plan = _plan([operator, overdue])
    assert plan.runner.command.correlation_id == operator.correlation_id
    assert plan.superseded_count == 0


def test_automatic_request_after_the_operator_one_is_left_for_a_later_poll() -> None:
    first, operator, later = _overdue(), _operator(), _overdue()
    plan = _plan([first, operator, later])
    assert plan.runner.command.correlation_id == operator.correlation_id
    assert plan.superseded_count == 1
    assert plan.stop_reason is EnumCoalesceRefusal.OPERATOR_REQUEST


def test_superseded_record_names_the_operator_job_and_the_reason(tmp_path: Any) -> None:
    operator, overdue = _operator(), _overdue()
    supersession = _plan([overdue, operator]).superseded[0]
    store = JobStore(str(tmp_path))
    job = store.record_superseded(
        overdue.correlation_id,
        overdue.model_dump(mode="json"),
        superseded_by_sha=supersession.superseded_by_sha,
        superseded_by_correlation_id=supersession.superseded_by_correlation_id,
        reason=supersession.reason,
    )
    assert job.superseded_by_correlation_id == operator.correlation_id
    assert "operator-path request" in job.errors[0]
    assert str(operator.correlation_id) in job.errors[0]
