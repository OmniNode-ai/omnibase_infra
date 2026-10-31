# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Parity tests for the typed-request handle() of node_registry_api_effect.

The nine handlers declared ``handle(request: object, correlation_id: UUID)``;
each now takes one ``ModelRegistryApiRequest`` that carries the correlation id.
These tests drive ``HandlerRegistryApiGetNode`` (the node's representative
handler) through the new call shape. Against the merge-base handler the call
raises ``TypeError``, which each test turns into its own assertion failure.

The scenarios double as the candidate inputs the adequacy receipt under
``scripts/ci/adequacy_receipts/`` was recorded over.
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_node import (
    HandlerRegistryApiGetNode,
)
from omnibase_infra.nodes.node_registry_api_effect.models import (
    ModelRegistryApiRequest,
    ModelRegistryApiResponse,
)

pytestmark = pytest.mark.unit


class ModelGetNodeScenario(BaseModel):
    """One get_node request plus what the discovery service answers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    node_state: Literal["found", "missing", "no_node_id"]

    @property
    def correlation_id(self) -> UUID:
        return UUID(int=self.seed)

    @property
    def node_id(self) -> UUID | None:
        return None if self.node_state == "no_node_id" else UUID(int=self.seed + 1000)


SCENARIOS: tuple[ModelGetNodeScenario, ...] = (
    ModelGetNodeScenario(seed=1, node_state="found"),
    ModelGetNodeScenario(seed=2, node_state="missing"),
    ModelGetNodeScenario(seed=3, node_state="no_node_id"),
)


def build_handler(scenario: ModelGetNodeScenario) -> HandlerRegistryApiGetNode:
    """A handler whose discovery service answers as the scenario says."""
    node = None
    if scenario.node_state == "found":
        node = MagicMock()
        node.model_dump.return_value = {"id": str(scenario.node_id), "name": "n"}
    service = MagicMock()
    service.get_node = AsyncMock(return_value=(node, []))
    return HandlerRegistryApiGetNode(service=service)


def build_request(scenario: ModelGetNodeScenario) -> ModelRegistryApiRequest:
    return ModelRegistryApiRequest(
        operation="get_node",
        correlation_id=scenario.correlation_id,
        node_id=scenario.node_id,
    )


class TestGetNodeHandleTypedRequest:
    @pytest.mark.asyncio
    async def test_found_node_answers_with_the_request_correlation_id(self) -> None:
        scenario = SCENARIOS[0]
        response = await build_handler(scenario).handle(build_request(scenario))
        assert isinstance(response, ModelRegistryApiResponse)
        assert response.success is True
        assert response.correlation_id == scenario.correlation_id
        assert response.data["result"] == {
            "id": str(scenario.node_id),
            "name": "n",
        }

    @pytest.mark.asyncio
    async def test_missing_node_fails_with_the_request_correlation_id(self) -> None:
        scenario = SCENARIOS[1]
        response = await build_handler(scenario).handle(build_request(scenario))
        assert response.success is False
        assert response.error == "Node not found"
        assert response.correlation_id == scenario.correlation_id

    @pytest.mark.asyncio
    async def test_request_without_node_id_is_rejected(self) -> None:
        scenario = SCENARIOS[2]
        with pytest.raises(ValueError, match="node_id is required"):
            await build_handler(scenario).handle(build_request(scenario))
