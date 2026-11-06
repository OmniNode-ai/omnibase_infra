# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The existing data fetch model is the handler's typed request."""

import inspect
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_rsd_data_fetch_effect.handlers.handler_rsd_data_fetch import (
    HandlerRsdDataFetch,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models import (
    ModelRsdDataFetchRequest,
    ModelRsdDataFetchResult,
)


async def test_handle_data_fetch_request() -> None:
    request = ModelRsdDataFetchRequest(
        correlation_id=uuid4(),
        ticket_ids=("OMN-21536",),
        include_dependencies=False,
        include_agent_requests=False,
        include_plan_overrides=False,
    )

    result = await HandlerRsdDataFetch().handle(request)

    assert result == ModelRsdDataFetchResult(
        correlation_id=request.correlation_id,
        tickets=(),
        dependency_edges=(),
        agent_requests=(),
        plan_overrides=(),
        fetch_errors=(),
        success=True,
    )


def test_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelRsdDataFetchRequest.model_validate(
            {"correlation_id": uuid4(), "ticket_ids": (), "unknown_field": True}
        )


def test_handle_has_one_request_parameter() -> None:
    assert list(inspect.signature(HandlerRsdDataFetch().handle).parameters) == [
        "request"
    ]
