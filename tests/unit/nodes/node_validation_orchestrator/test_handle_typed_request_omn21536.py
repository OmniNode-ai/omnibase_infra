# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21536: validation plan building handles one typed request."""

import inspect
import logging
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumCheckSeverity
from omnibase_infra.nodes.node_validation_orchestrator.handlers.handler_build_plan import (
    HandlerBuildPlan,
)
from omnibase_infra.nodes.node_validation_orchestrator.models import (
    ModelPatternCandidate,
)
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope

_HANDLER_LOGGER = (
    "omnibase_infra.nodes.node_validation_orchestrator.handlers.handler_build_plan"
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


async def test_plan_logs_dispatch_envelope_correlation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = _request()
    correlation_id = uuid4()
    envelope = ModelEventEnvelope[object](
        payload=request.model_dump(mode="json"), correlation_id=correlation_id
    )

    with caplog.at_level(logging.INFO, logger=_HANDLER_LOGGER):
        with bind_dispatch_envelope(envelope):
            await HandlerBuildPlan().handle(request)

    logged = [r for r in caplog.records if r.name == _HANDLER_LOGGER]
    assert [getattr(r, "correlation_id", None) for r in logged] == [str(correlation_id)]


async def test_standalone_plans_get_distinct_correlation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = HandlerBuildPlan()

    with caplog.at_level(logging.INFO, logger=_HANDLER_LOGGER):
        await handler.handle(_request())
        await handler.handle(_request())

    ids = [
        UUID(r.correlation_id)
        for r in caplog.records
        if r.name == _HANDLER_LOGGER and hasattr(r, "correlation_id")
    ]
    assert len(ids) == 2
    assert ids[0] != ids[1]
