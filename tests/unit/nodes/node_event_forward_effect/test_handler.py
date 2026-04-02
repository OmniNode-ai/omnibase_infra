# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for HandlerEventForward."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.nodes.node_event_forward_effect.handlers.handler_event_forward import (
    HandlerEventForward,
)
from omnibase_infra.nodes.node_event_forward_effect.models.model_event_forward_request import (
    ModelEventForwardRequest,
)


def _make_request(**overrides) -> ModelEventForwardRequest:  # type: ignore[no-untyped-def]
    defaults = {
        "correlation_id": uuid4(),
        "event_type": "service.started",
        "category": "lifecycle",
        "source": "test-service",
    }
    defaults.update(overrides)
    return ModelEventForwardRequest(**defaults)


@pytest.mark.unit
class TestHandlerEventForward:
    @pytest.mark.asyncio
    async def test_successful_forward(self) -> None:
        mock_response = httpx.Response(
            200,
            request=httpx.Request(
                "POST", "http://backend/api/events/service-lifecycle"
            ),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)

        handler = HandlerEventForward(http_client=mock_client)
        with patch.dict("os.environ", {"EVENT_FORWARD_BACKEND_URL": "http://backend"}):
            result = await handler.handle(_make_request())

        assert result.success is True
        assert result.http_status == 200
        assert "service-lifecycle" in result.endpoint

    @pytest.mark.asyncio
    async def test_backend_rejects(self) -> None:
        mock_response = httpx.Response(
            400,
            request=httpx.Request(
                "POST", "http://backend/api/events/service-lifecycle"
            ),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)

        handler = HandlerEventForward(http_client=mock_client)
        with patch.dict("os.environ", {"EVENT_FORWARD_BACKEND_URL": "http://backend"}):
            result = await handler.handle(_make_request())

        assert result.success is False
        assert result.http_status == 400

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        handler = HandlerEventForward(http_client=mock_client)
        with patch.dict("os.environ", {"EVENT_FORWARD_BACKEND_URL": "http://backend"}):
            result = await handler.handle(_make_request())

        assert result.success is False
        assert "refused" in result.error_message

    @pytest.mark.asyncio
    async def test_category_routing(self) -> None:
        mock_response = httpx.Response(
            200,
            request=httpx.Request("POST", "http://backend/api/events/tool-update"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)

        handler = HandlerEventForward(http_client=mock_client)
        request = _make_request(category="tool", event_type="tool.created")
        with patch.dict("os.environ", {"EVENT_FORWARD_BACKEND_URL": "http://backend"}):
            result = await handler.handle(request)

        assert result.success is True
        assert "tool-update" in result.endpoint

    @pytest.mark.asyncio
    async def test_system_category(self) -> None:
        mock_response = httpx.Response(
            202,
            request=httpx.Request("POST", "http://backend/api/events/system-event"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)

        handler = HandlerEventForward(http_client=mock_client)
        request = _make_request(
            category="system", event_type="system.alert", severity="critical"
        )
        with patch.dict("os.environ", {"EVENT_FORWARD_BACKEND_URL": "http://backend"}):
            result = await handler.handle(request)

        assert result.success is True
        assert "system-event" in result.endpoint
