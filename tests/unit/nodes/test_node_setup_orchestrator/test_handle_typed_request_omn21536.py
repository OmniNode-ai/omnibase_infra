# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request contract for setup orchestration (OMN-21536)."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.enums import EnumDeploymentMode
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.models.core.model_deployment_topology_service import (
    ModelDeploymentTopologyService,
)
from omnibase_infra.nodes.node_setup_orchestrator.handlers.handler_setup_orchestrator import (
    HandlerSetupOrchestrator,
)
from omnibase_infra.nodes.node_setup_orchestrator.models import (
    ModelSetupOrchestratorRequest,
)


async def test_handle_typed_request() -> None:
    correlation_id = uuid4()
    request = ModelSetupOrchestratorRequest(
        topology=ModelDeploymentTopology(
            schema_version="1.0",
            services={
                "postgres": ModelDeploymentTopologyService(
                    mode=EnumDeploymentMode.CLOUD, local=None
                )
            },
            presets={},
            active_preset=None,
        ),
        correlation_id=correlation_id,
        compose_file_path="docker-compose.yml",
    )
    preflight = AsyncMock()
    handler = HandlerSetupOrchestrator(
        container=MagicMock(),
        preflight=preflight,
        provision=AsyncMock(),
        infisical=AsyncMock(),
        validate=AsyncMock(),
    )

    result = await handler.handle(request)

    assert result.result is not None
    assert result.result.correlation_id == correlation_id
    assert [event.event_type for event in result.result.events] == [
        "setup.cloud.unavailable"
    ]
    assert result.result.events[0].payload == {"gated_services": ["postgres"]}
    preflight.run_preflight.assert_not_called()
    assert list(inspect.signature(handler.handle).parameters) == ["request"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelSetupOrchestratorRequest.model_validate(
            {**request.model_dump(), "unknown_field": True}
        )
