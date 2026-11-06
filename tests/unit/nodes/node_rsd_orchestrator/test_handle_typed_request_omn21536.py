# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The declared RSD command is the initiate handler's typed request."""

import inspect
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_rsd_data_fetch_effect.handlers.handler_rsd_data_fetch import (
    HandlerRsdDataFetch,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models import (
    ModelRsdDataFetchRequest,
)
from omnibase_infra.nodes.node_rsd_orchestrator.handlers.handler_rsd_initiate import (
    HandlerRsdInitiate,
)
from omnibase_infra.nodes.node_rsd_orchestrator.models import ModelRsdCommand
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_factor_weights import (
    ModelRsdFactorWeights,
)


async def test_handle_request_dispatches_fetch() -> None:
    request = ModelRsdCommand(
        correlation_id=uuid4(),
        ticket_ids=("OMN-21536", "OMN-21537"),
        weights=ModelRsdFactorWeights(),
        include_dependencies=False,
        include_agent_requests=False,
        include_plan_overrides=False,
    )

    fetch_request = await HandlerRsdInitiate().handle(request)

    assert fetch_request == ModelRsdDataFetchRequest(
        correlation_id=request.correlation_id,
        ticket_ids=request.ticket_ids,
        include_dependencies=False,
        include_agent_requests=False,
        include_plan_overrides=False,
    )
    result = await HandlerRsdDataFetch().handle(fetch_request)
    assert result.success is True
    assert result.correlation_id == request.correlation_id
    assert result.tickets == ()


async def test_handle_request_preserves_default_inclusion_flags() -> None:
    request = ModelRsdCommand(
        correlation_id=uuid4(), ticket_ids=(), weights=ModelRsdFactorWeights()
    )

    fetch_request = await HandlerRsdInitiate().handle(request)

    assert fetch_request.correlation_id == request.correlation_id
    assert fetch_request.ticket_ids == ()
    assert fetch_request.include_dependencies is True
    assert fetch_request.include_agent_requests is True
    assert fetch_request.include_plan_overrides is True


def test_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelRsdCommand.model_validate(
            {"correlation_id": uuid4(), "ticket_ids": (), "unknown_field": True}
        )


def test_handle_has_one_request_parameter() -> None:
    assert list(inspect.signature(HandlerRsdInitiate().handle).parameters) == [
        "request"
    ]
