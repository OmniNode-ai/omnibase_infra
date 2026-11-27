# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""node_queued_request_priority_compute: operator-path ahead of automatic (OMN-26718)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.handlers.queued_request_priority.handler_queued_request_priority import (
    HandlerQueuedRequestPriority,
)
from omnibase_infra.models.model_queued_request_entry import ModelQueuedRequestEntry
from omnibase_infra.models.model_queued_request_priority_decision import (
    ModelQueuedRequestPriorityDecision,
)
from omnibase_infra.models.model_queued_request_priority_request import (
    ModelQueuedRequestPriorityRequest,
)

pytestmark = pytest.mark.unit

OVERDUE = "agent/overdue-dev-merge-build"


def _entry(requested_by: str, sha: str, **kw: object) -> ModelQueuedRequestEntry:
    fields: dict[str, object] = {
        "correlation_id": uuid4(),
        "runtime_lane": "dev",
        "requested_by": requested_by,
        "git_ref": sha,
        "scope": "runtime",
        "build_source": "workspace",
    }
    fields.update(kw)
    return ModelQueuedRequestEntry.model_validate(fields)


def _decide(
    *entries: ModelQueuedRequestEntry, supersede: bool = True
) -> ModelQueuedRequestPriorityDecision:
    return HandlerQueuedRequestPriority().handle(
        ModelQueuedRequestPriorityRequest(entries=entries, supersede_queue=supersede)
    )


def test_operator_request_behind_an_automatic_one_runs_and_names_the_reason() -> None:
    auto, operator = _entry(OVERDUE, "a" * 40), _entry("operator-manual", "b" * 40)
    decision = _decide(auto, operator)
    assert decision.operator_path is True
    assert decision.runner_index == 1
    assert str(operator.correlation_id) in decision.reasons[0]
    assert "operator-path request" in decision.reasons[0]


def test_automatic_requests_alone_are_not_prioritised() -> None:
    decision = _decide(_entry(OVERDUE, "a" * 40), _entry(OVERDUE, "b" * 40))
    assert decision.operator_path is False
    assert decision.foldable == 2


def test_a_lane_that_does_not_supersede_its_queue_is_not_reordered() -> None:
    decision = _decide(
        _entry(OVERDUE, "a" * 40), _entry("operator-manual", "b" * 40), supersede=False
    )
    assert decision.operator_path is False


def test_an_operator_request_on_another_lane_is_not_promoted() -> None:
    decision = _decide(
        _entry(OVERDUE, "a" * 40),
        _entry("operator-manual", "b" * 40, runtime_lane="stability"),
    )
    assert decision.operator_path is False
    assert decision.refusal == "different_lane"


@pytest.mark.parametrize(
    ("override", "refusal"),
    [
        ({"scope": "core"}, "not_full_scope"),
        ({"services": ("x",)}, "services_differ"),
        ({"build_source": "image"}, "build_source_differs"),
        ({"pinned_image": True}, "pinned_image"),
        ({"git_ref": "short"}, "ref_not_a_sha"),
        ({"rollback_declared": True}, "rollback_declared"),
    ],
)
def test_static_refusal_ends_the_foldable_prefix(
    override: dict[str, object], refusal: str
) -> None:
    head = _entry("ci/dev", "a" * 40, scope="full")
    other = _entry("ci/dev", "b" * 40, scope="full")
    other = other.model_copy(update=override)
    decision = _decide(head, other, supersede=False)
    assert (decision.foldable, decision.refusal) == (1, refusal)


def test_a_redelivered_copy_ends_the_prefix() -> None:
    head = _entry("ci/dev", "a" * 40, scope="full")
    decision = _decide(head, head, supersede=False)
    assert (decision.foldable, decision.refusal) == (1, "duplicate_command")
