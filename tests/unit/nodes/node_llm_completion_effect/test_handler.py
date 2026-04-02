# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for HandlerLLMCompletion."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.nodes.node_llm_completion_effect.handlers.handler_llm_completion import (
    HandlerLLMCompletion,
)
from omnibase_infra.nodes.node_llm_completion_effect.models.model_llm_completion_message import (
    ModelLLMCompletionMessage,
)
from omnibase_infra.nodes.node_llm_completion_effect.models.model_llm_completion_request import (
    ModelLLMCompletionRequest,
)


def _make_request(**overrides) -> ModelLLMCompletionRequest:  # type: ignore[no-untyped-def]
    defaults = {
        "correlation_id": uuid4(),
        "messages": (ModelLLMCompletionMessage(role="user", content="Hello"),),
        "model": "test-model",
        "max_tokens": 100,
        "temperature": 0.5,
    }
    defaults.update(overrides)
    return ModelLLMCompletionRequest(**defaults)


@pytest.mark.unit
class TestHandlerLLMCompletion:
    @pytest.mark.asyncio
    async def test_successful_completion(self) -> None:
        mock_response = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hi there"}}],
                "model": "test-model",
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
            request=httpx.Request("POST", "http://test/v1/chat/completions"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)

        handler = HandlerLLMCompletion(http_client=mock_client)
        request = _make_request(endpoint_url="http://test")
        result = await handler.handle(request)

        assert result.success is True
        assert result.content == "Hi there"
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 3

    @pytest.mark.asyncio
    async def test_http_error(self) -> None:
        mock_response = httpx.Response(
            500,
            text="Internal Server Error",
            request=httpx.Request("POST", "http://test/v1/chat/completions"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        # Make raise_for_status actually raise
        mock_response.raise_for_status = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
            httpx.HTTPStatusError(
                "error", request=mock_response.request, response=mock_response
            )
        )

        handler = HandlerLLMCompletion(http_client=mock_client)
        request = _make_request(endpoint_url="http://test")
        result = await handler.handle(request)

        assert result.success is False
        assert "500" in result.error_message

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        handler = HandlerLLMCompletion(http_client=mock_client)
        request = _make_request(endpoint_url="http://test")
        result = await handler.handle(request)

        assert result.success is False
        assert "refused" in result.error_message

    @pytest.mark.asyncio
    async def test_endpoint_routing_fast(self) -> None:
        handler = HandlerLLMCompletion()
        # Short message -> fast endpoint
        request = _make_request(endpoint_url="")
        with patch.dict(
            "os.environ",
            {
                "LLM_CODER_FAST_URL": "http://fast:8001",
                "LLM_CODER_URL": "http://full:8000",
            },
        ):
            endpoint = handler._resolve_endpoint(request)
        assert endpoint == "http://fast:8001"

    @pytest.mark.asyncio
    async def test_endpoint_routing_full(self) -> None:
        handler = HandlerLLMCompletion()
        # Long message -> full endpoint
        long_content = "x" * 200_000  # ~50K tokens
        request = _make_request(
            endpoint_url="",
            messages=(ModelLLMCompletionMessage(role="user", content=long_content),),
        )
        with patch.dict(
            "os.environ",
            {
                "LLM_CODER_FAST_URL": "http://fast:8001",
                "LLM_CODER_URL": "http://full:8000",
            },
        ):
            endpoint = handler._resolve_endpoint(request)
        assert endpoint == "http://full:8000"
