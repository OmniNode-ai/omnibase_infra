# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Queued overdue dev merge builds fold into the newest one (OMN-26316).

THE RECORD
----------
2026-11-25 on h201: jobs d844bdf1 (accepted 07:15:07Z), 7d9991bf (07:29:00Z) and 7db4beaf
(07:42:13Z) were each requested by ``agent/overdue-dev-merge-build`` for the same dev head
``f553b1d6``, scope ``runtime``, each carrying a rollback declaration of that actor. The
coalescer refused a declared rollback and any scope but ``full`` as static refusals, so the
queue the dev lane supersedes outright ran one build after another (about 13 minutes each,
a recreate and a 503 each). The refusals stay for every other command.
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

pytestmark = pytest.mark.unit

HEAD = "f553b1d6" + "8" * 32
NEWER = "a1b2c3d4" + "9" * 32
OVERDUE = "agent/overdue-dev-merge-build"


def _overdue(sha: str = HEAD, **overrides: Any) -> ModelRebuildRequested:
    payload: dict[str, Any] = {
        "correlation_id": uuid4(),
        "requested_by": OVERDUE,
        "scope": "runtime",
        "runtime_lane": "dev",
        "build_source": "workspace",
        "git_ref": sha,
        "rollback": {
            "actor": OVERDUE,
            "reason": "merge 3a527226 had no dev build by 2026-11-25T05:51:09+00:00",
        },
        "open_pr": f"omnibase_infra#6266@{sha}",
    }
    payload.update(overrides)
    return ModelRebuildRequested.model_validate(payload)


def _queued(cmd: ModelRebuildRequested, offset: int) -> ModelQueuedCommand:
    return ModelQueuedCommand(command=cmd, partition=0, offset=offset)


def _contained(earlier: str, later: str) -> bool | None:
    return True


def _plan(commands: list[ModelRebuildRequested]) -> ModelCoalescePlan:
    return plan_coalesce(
        [_queued(c, 10 + i) for i, c in enumerate(commands)],
        contains=_contained,
        supersede_lanes=frozenset({EnumRuntimeLane.DEV}),
    )


def test_three_queued_overdue_requests_run_as_one_build() -> None:
    first, second, third = _overdue(), _overdue(), _overdue()
    plan = _plan([first, second, third])
    assert plan.runner.command.correlation_id == third.correlation_id
    assert plan.superseded_count == 2
    assert plan.stop_reason is None


def test_an_operators_rollback_still_ends_the_group() -> None:
    operator = _overdue(
        requested_by="operator/lane",
        rollback={"actor": "operator/lane", "reason": "back out 6191"},
    )
    plan = _plan([_overdue(), operator])
    assert plan.superseded_count == 0
    assert plan.stop_reason is EnumCoalesceRefusal.ROLLBACK_DECLARED


def test_an_overdue_request_never_folds_into_a_full_scope_build() -> None:
    full = ModelRebuildRequested.model_validate(
        {
            "correlation_id": uuid4(),
            "requested_by": "gha/omnibase_infra/pr-1",
            "scope": "full",
            "runtime_lane": "dev",
            "build_source": "workspace",
            "git_ref": NEWER,
        }
    )
    plan = _plan([_overdue(), full])
    assert plan.superseded_count == 0
