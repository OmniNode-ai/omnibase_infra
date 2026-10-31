# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21536: validation plan building handles one typed request."""

import inspect
from uuid import uuid4

from omnibase_infra.enums import EnumCheckSeverity
from omnibase_infra.nodes.node_validation_orchestrator.handlers.handler_build_plan import (
    HandlerBuildPlan,
)
from omnibase_infra.nodes.node_validation_orchestrator.models import (
    ModelPatternCandidate,
)


def _request() -> ModelPatternCandidate:
    return ModelPatternCandidate(
        candidate_id=uuid4(),
        pattern_id=uuid4(),
        source_path="/src/test_module",
    )


async def test_handle_typed_request_returns_full_plan() -> None:
    request = _request()

    plan = await HandlerBuildPlan().handle(request)

    assert plan.candidate_id == request.candidate_id
    assert len(plan.checks) == 12
    assert plan.score_threshold == 0.8
    assert all(check.enabled for check in plan.checks)
    assert (
        sum(check.severity is EnumCheckSeverity.REQUIRED for check in plan.checks) == 7
    )
    assert (
        sum(check.severity is EnumCheckSeverity.RECOMMENDED for check in plan.checks)
        == 3
    )
    assert (
        sum(check.severity is EnumCheckSeverity.INFORMATIONAL for check in plan.checks)
        == 2
    )


def test_handle_signature() -> None:
    assert tuple(inspect.signature(HandlerBuildPlan().handle).parameters) == (
        "request",
    )
