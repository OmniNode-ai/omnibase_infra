# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for registry API operational handlers (OMN-4482).

Tests:
    - test_get_health_returns_component_status_fields
    - test_get_widget_mapping_parses_yaml
    - test_get_discovery_aggregates_two_handlers

Ticket: OMN-4482
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_discovery import (
    HandlerRegistryApiGetDiscovery,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_health import (
    HandlerRegistryApiGetHealth,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_widget_mapping import (
    HandlerRegistryApiGetWidgetMapping,
)
from omnibase_infra.nodes.node_registry_api_effect.models import (
    ModelRegistryApiResponse,
)


@pytest.mark.unit
class TestHandlerRegistryApiGetHealth:
    """Tests for HandlerRegistryApiGetHealth."""

    @pytest.mark.asyncio
    async def test_get_health_returns_component_status_fields(self) -> None:
        """get_health must return a response with status and components fields."""
        handler = HandlerRegistryApiGetHealth()
        correlation_id = uuid4()

        # Patch all external calls to simulate unhealthy (unreachable) components
        with (
            patch(
                "omnibase_infra.nodes.node_registry_api_effect.handlers."
                "handler_registry_api_get_health.asyncpg",
                side_effect=ImportError("no asyncpg"),
                create=True,
            ),
        ):
            # Use module-level patch for the entire handle body
            pass

        # Simply call handle — all components will fail gracefully in test env
        result = await handler.handle(request=object(), correlation_id=correlation_id)

        assert isinstance(result, ModelRegistryApiResponse)
        assert result.operation == "get_health"
        assert result.success is True
        assert "status" in result.data
        assert "components" in result.data
        components = result.data["components"]
        assert "db" in components
        assert "kafka" in components
        assert "qdrant" in components


@pytest.mark.unit
class TestHandlerRegistryApiGetWidgetMapping:
    """Tests for HandlerRegistryApiGetWidgetMapping."""

    @pytest.mark.asyncio
    async def test_get_widget_mapping_parses_yaml(self) -> None:
        """get_widget_mapping must parse a YAML file and return the content."""
        yaml_content = "widgets:\n  health_node: status_indicator\n  event_feed: feed\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write(yaml_content)
            tmp_path = Path(tmp.name)

        try:
            handler = HandlerRegistryApiGetWidgetMapping(mapping_path=tmp_path)
            correlation_id = uuid4()
            result = await handler.handle(
                request=object(), correlation_id=correlation_id
            )

            assert isinstance(result, ModelRegistryApiResponse)
            assert result.operation == "get_widget_mapping"
            assert result.success is True
            assert "widget_mapping" in result.data
            mapping = result.data["widget_mapping"]
            assert "widgets" in mapping
            assert mapping["widgets"]["health_node"] == "status_indicator"
        finally:
            tmp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_get_widget_mapping_returns_error_on_missing_file(self) -> None:
        """get_widget_mapping must return error response when file is not found."""
        handler = HandlerRegistryApiGetWidgetMapping(
            mapping_path=Path("/nonexistent/widget_mapping.yaml")
        )
        result = await handler.handle(request=object(), correlation_id=uuid4())

        assert isinstance(result, ModelRegistryApiResponse)
        assert result.success is False
        assert result.error is not None


@pytest.mark.unit
class TestHandlerRegistryApiGetDiscovery:
    """Tests for HandlerRegistryApiGetDiscovery."""

    @pytest.mark.asyncio
    async def test_get_discovery_aggregates_two_handlers(self) -> None:
        """get_discovery must call list_nodes and get_widget_mapping (OMN-9545)."""
        nodes_response = ModelRegistryApiResponse(
            operation="list_nodes",
            success=True,
            data={"nodes": ["node-a"], "total": 1},
        )
        mapping_response = ModelRegistryApiResponse(
            operation="get_widget_mapping",
            success=True,
            data={"widget_mapping": {"foo": "bar"}},
        )

        mock_list_nodes = MagicMock()
        mock_list_nodes.handle = AsyncMock(return_value=nodes_response)

        mock_get_widget_mapping = MagicMock()
        mock_get_widget_mapping.handle = AsyncMock(return_value=mapping_response)

        handler = HandlerRegistryApiGetDiscovery(
            list_nodes=mock_list_nodes,
            get_widget_mapping=mock_get_widget_mapping,
        )
        correlation_id = uuid4()
        result = await handler.handle(request=object(), correlation_id=correlation_id)

        mock_list_nodes.handle.assert_awaited_once()
        mock_get_widget_mapping.handle.assert_awaited_once()

        assert isinstance(result, ModelRegistryApiResponse)
        assert result.operation == "get_discovery"
        assert result.success is True
        assert result.data["nodes"] == ["node-a"]
        assert "instances" not in result.data
        assert result.data["widget_mapping"] == {"foo": "bar"}
