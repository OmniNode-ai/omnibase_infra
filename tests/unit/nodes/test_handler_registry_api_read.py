# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for registry API read handlers (OMN-4481).

Tests:
    - test_list_nodes_delegates_to_service_with_correct_args
    - test_get_node_delegates_to_service_with_correct_args
    - test_list_contracts_delegates_to_service_with_correct_args
    - test_get_contract_delegates_to_service_with_correct_args
    - test_list_topics_delegates_to_service_with_correct_args
    - test_get_topic_delegates_to_service_with_correct_args

Ticket: OMN-4481
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_contract import (
    HandlerRegistryApiGetContract,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_node import (
    HandlerRegistryApiGetNode,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_topic import (
    HandlerRegistryApiGetTopic,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_list_contracts import (
    HandlerRegistryApiListContracts,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_list_nodes import (
    HandlerRegistryApiListNodes,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_list_topics import (
    HandlerRegistryApiListTopics,
)
from omnibase_infra.nodes.node_registry_api_effect.models import (
    ModelRegistryApiRequest,
    ModelRegistryApiResponse,
)


def _make_pagination() -> MagicMock:
    p = MagicMock()
    p.model_dump.return_value = {
        "total": 0,
        "limit": 100,
        "offset": 0,
        "has_more": False,
    }
    return p


@pytest.mark.unit
class TestHandlerRegistryApiListNodes:
    """Tests for HandlerRegistryApiListNodes."""

    @pytest.mark.asyncio
    async def test_list_nodes_delegates_to_service_with_correct_args(self) -> None:
        """Handler calls service.list_nodes with state=None, node_type=None, limit=10, offset=0."""
        service = MagicMock()
        service.list_nodes = AsyncMock(return_value=([], _make_pagination(), []))
        handler = HandlerRegistryApiListNodes(service=service)

        request = ModelRegistryApiRequest(operation="list_nodes", limit=10, offset=0)
        correlation_id = uuid4()

        response = await handler.handle(request.model_dump(), correlation_id)

        service.list_nodes.assert_called_once_with(
            limit=10,
            offset=0,
            state=None,
            node_type=None,
            correlation_id=correlation_id,
        )
        assert isinstance(response, ModelRegistryApiResponse)
        assert response.success is True
        assert "results" in response.data


@pytest.mark.unit
class TestHandlerRegistryApiGetNode:
    """Tests for HandlerRegistryApiGetNode."""

    @pytest.mark.asyncio
    async def test_get_node_delegates_to_service_with_correct_args(self) -> None:
        """Handler calls service.get_node with the provided node_id."""
        node_id = uuid4()
        mock_node = MagicMock()
        mock_node.model_dump.return_value = {"id": str(node_id), "name": "test_node"}
        service = MagicMock()
        service.get_node = AsyncMock(return_value=(mock_node, []))
        handler = HandlerRegistryApiGetNode(service=service)

        request = ModelRegistryApiRequest(operation="get_node", node_id=node_id)
        correlation_id = uuid4()

        response = await handler.handle(request.model_dump(), correlation_id)

        service.get_node.assert_called_once_with(
            node_id=node_id,
            correlation_id=correlation_id,
        )
        assert isinstance(response, ModelRegistryApiResponse)
        assert response.success is True
        assert response.data["result"] is not None


@pytest.mark.unit
class TestHandlerRegistryApiListContracts:
    """Tests for HandlerRegistryApiListContracts."""

    @pytest.mark.asyncio
    async def test_list_contracts_delegates_to_service_with_correct_args(self) -> None:
        """Handler calls service.list_contracts with limit and offset."""
        service = MagicMock()
        service.list_contracts = AsyncMock(return_value=([], _make_pagination(), []))
        handler = HandlerRegistryApiListContracts(service=service)

        request = ModelRegistryApiRequest(
            operation="list_contracts", limit=50, offset=0
        )
        correlation_id = uuid4()

        response = await handler.handle(request.model_dump(), correlation_id)

        service.list_contracts.assert_called_once_with(
            limit=50,
            offset=0,
            correlation_id=correlation_id,
        )
        assert isinstance(response, ModelRegistryApiResponse)
        assert response.success is True
        assert "results" in response.data


@pytest.mark.unit
class TestHandlerRegistryApiGetContract:
    """Tests for HandlerRegistryApiGetContract."""

    @pytest.mark.asyncio
    async def test_get_contract_delegates_to_service_with_correct_args(self) -> None:
        """Handler calls service.get_contract with str(contract_id)."""
        contract_id = uuid4()
        mock_contract = MagicMock()
        mock_contract.model_dump.return_value = {
            "id": str(contract_id),
            "name": "test_contract",
        }
        service = MagicMock()
        service.get_contract = AsyncMock(return_value=(mock_contract, []))
        handler = HandlerRegistryApiGetContract(service=service)

        request = ModelRegistryApiRequest(
            operation="get_contract", contract_id=contract_id
        )
        correlation_id = uuid4()

        response = await handler.handle(request.model_dump(), correlation_id)

        service.get_contract.assert_called_once_with(
            contract_id=str(contract_id),
            correlation_id=correlation_id,
        )
        assert isinstance(response, ModelRegistryApiResponse)
        assert response.success is True
        assert response.data["result"] is not None


@pytest.mark.unit
class TestHandlerRegistryApiListTopics:
    """Tests for HandlerRegistryApiListTopics."""

    @pytest.mark.asyncio
    async def test_list_topics_delegates_to_service_with_correct_args(self) -> None:
        """Handler calls service.list_topics with limit and offset."""
        service = MagicMock()
        service.list_topics = AsyncMock(return_value=([], _make_pagination(), []))
        handler = HandlerRegistryApiListTopics(service=service)

        request = ModelRegistryApiRequest(operation="list_topics", limit=25, offset=5)
        correlation_id = uuid4()

        response = await handler.handle(request.model_dump(), correlation_id)

        service.list_topics.assert_called_once_with(
            limit=25,
            offset=5,
            correlation_id=correlation_id,
        )
        assert isinstance(response, ModelRegistryApiResponse)
        assert response.success is True
        assert "results" in response.data


@pytest.mark.unit
class TestHandlerRegistryApiGetTopic:
    """Tests for HandlerRegistryApiGetTopic."""

    @pytest.mark.asyncio
    async def test_get_topic_delegates_to_service_with_correct_args(self) -> None:
        """Handler calls service.get_topic_detail with the provided topic_suffix."""
        mock_topic = MagicMock()
        mock_topic.model_dump.return_value = {
            "suffix": "agent.routing.v1",
            "direction": "publish",
        }
        service = MagicMock()
        service.get_topic_detail = AsyncMock(return_value=(mock_topic, []))
        handler = HandlerRegistryApiGetTopic(service=service)

        request = ModelRegistryApiRequest(
            operation="get_topic", topic_suffix="agent.routing.v1"
        )
        correlation_id = uuid4()

        response = await handler.handle(request.model_dump(), correlation_id)

        service.get_topic_detail.assert_called_once_with(
            topic_suffix="agent.routing.v1",
            correlation_id=correlation_id,
        )
        assert isinstance(response, ModelRegistryApiResponse)
        assert response.success is True
        assert response.data["result"] is not None
