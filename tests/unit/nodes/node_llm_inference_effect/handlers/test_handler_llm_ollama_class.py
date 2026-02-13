# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerLlmOllama class methods.

Tests cover:
    - Initialization with default and custom parameters
    - Property accessors (handler_type, handler_category)
    - close() delegation to mixin
    - _FINISH_REASON_MAP class variable identity
    - _map_finish_reason() for all known and unknown reasons
    - _build_chat_payload() for chat completion requests
    - _build_generate_payload() for completion requests

Related:
    - OMN-2108: Phase 8 Ollama inference handler
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumLlmFinishReason,
    EnumLlmOperationType,
)
from omnibase_infra.nodes.effects.models import (
    ModelLlmFunctionDef,
    ModelLlmInferenceRequest,
    ModelLlmMessage,
    ModelLlmToolDefinition,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_ollama import (
    _OLLAMA_FINISH_REASON_MAP,
    HandlerLlmOllama,
)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_chat_request(
    *,
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
    messages: tuple[ModelLlmMessage, ...] | None = None,
    system_prompt: str | None = None,
    tools: tuple[ModelLlmToolDefinition, ...] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> ModelLlmInferenceRequest:
    """Create a valid CHAT_COMPLETION request with sensible defaults."""
    if messages is None:
        messages = (ModelLlmMessage(role="user", content="Hello"),)
    return ModelLlmInferenceRequest(
        base_url=base_url,
        model=model,
        operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        messages=messages,
        system_prompt=system_prompt,
        tools=tools if tools is not None else (),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def _make_completion_request(
    *,
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
    prompt: str = "Once upon a time",
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> ModelLlmInferenceRequest:
    """Create a valid COMPLETION request with sensible defaults."""
    return ModelLlmInferenceRequest(
        base_url=base_url,
        model=model,
        operation_type=EnumLlmOperationType.COMPLETION,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def _make_tool_definition(
    name: str = "search",
    description: str = "Search the web",
) -> ModelLlmToolDefinition:
    """Create a minimal tool definition."""
    return ModelLlmToolDefinition(
        function=ModelLlmFunctionDef(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlerLlmOllamaInit:
    """Tests for HandlerLlmOllama initialization, properties, and close()."""

    def test_default_initialization(self) -> None:
        """Default constructor sets target_name='ollama', max_timeout=120.0, no client."""
        handler = HandlerLlmOllama()

        assert handler._llm_target_name == "ollama"
        assert handler._max_timeout_seconds == 120.0
        assert handler._http_client is None
        assert handler._owns_http_client is True

    def test_custom_initialization(self) -> None:
        """Custom target_name and max_timeout are forwarded to the mixin."""
        handler = HandlerLlmOllama(
            target_name="my-ollama",
            max_timeout_seconds=60.0,
        )

        assert handler._llm_target_name == "my-ollama"
        assert handler._max_timeout_seconds == 60.0

    def test_injected_http_client(self) -> None:
        """When http_client is provided, _owns_http_client is False."""
        mock_client = MagicMock()
        handler = HandlerLlmOllama(http_client=mock_client)

        assert handler._http_client is mock_client
        assert handler._owns_http_client is False

    def test_handler_type_is_infra(self) -> None:
        """handler_type returns INFRA_HANDLER."""
        handler = HandlerLlmOllama()
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category_is_effect(self) -> None:
        """handler_category returns EFFECT."""
        handler = HandlerLlmOllama()
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    @pytest.mark.asyncio
    async def test_close_delegates_to_mixin(self) -> None:
        """close() calls _close_http_client() from the mixin."""
        handler = HandlerLlmOllama()
        with patch.object(
            handler, "_close_http_client", new_callable=AsyncMock
        ) as mock_close:
            await handler.close()
            mock_close.assert_awaited_once()

    def test_finish_reason_map_class_var(self) -> None:
        """_FINISH_REASON_MAP is identical to _OLLAMA_FINISH_REASON_MAP."""
        assert HandlerLlmOllama._FINISH_REASON_MAP is _OLLAMA_FINISH_REASON_MAP


@pytest.mark.unit
class TestMapFinishReason:
    """Tests for HandlerLlmOllama._map_finish_reason()."""

    def setup_method(self) -> None:
        """Create a handler instance for each test."""
        self.handler = HandlerLlmOllama()

    def test_stop_maps_to_STOP(self) -> None:
        """'stop' maps to EnumLlmFinishReason.STOP."""
        assert self.handler._map_finish_reason("stop") == EnumLlmFinishReason.STOP

    def test_length_maps_to_LENGTH(self) -> None:
        """'length' maps to EnumLlmFinishReason.LENGTH."""
        assert self.handler._map_finish_reason("length") == EnumLlmFinishReason.LENGTH

    def test_tool_calls_maps_to_TOOL_CALLS(self) -> None:
        """'tool_calls' maps to EnumLlmFinishReason.TOOL_CALLS."""
        assert (
            self.handler._map_finish_reason("tool_calls")
            == EnumLlmFinishReason.TOOL_CALLS
        )

    def test_content_filter_maps_to_CONTENT_FILTER(self) -> None:
        """'content_filter' maps to EnumLlmFinishReason.CONTENT_FILTER."""
        assert (
            self.handler._map_finish_reason("content_filter")
            == EnumLlmFinishReason.CONTENT_FILTER
        )

    def test_none_maps_to_UNKNOWN(self) -> None:
        """None maps to EnumLlmFinishReason.UNKNOWN."""
        assert self.handler._map_finish_reason(None) == EnumLlmFinishReason.UNKNOWN

    def test_unrecognized_string_maps_to_UNKNOWN(self) -> None:
        """An unrecognized string maps to EnumLlmFinishReason.UNKNOWN."""
        assert (
            self.handler._map_finish_reason("something_else")
            == EnumLlmFinishReason.UNKNOWN
        )


@pytest.mark.unit
class TestBuildChatPayload:
    """Tests for HandlerLlmOllama._build_chat_payload()."""

    def setup_method(self) -> None:
        """Create a handler instance for each test."""
        self.handler = HandlerLlmOllama()

    def test_minimal_chat_payload(self) -> None:
        """Minimal chat payload has model, messages, and stream=False."""
        request = _make_chat_request()
        payload = self.handler._build_chat_payload(request)

        assert payload["model"] == "llama3"
        assert payload["stream"] is False
        messages_val = payload["messages"]
        assert isinstance(messages_val, list)
        assert len(messages_val) == 1
        msg = messages_val[0]
        assert isinstance(msg, dict)
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"
        # No options, system, or tools keys
        assert "options" not in payload
        assert "system" not in payload
        assert "tools" not in payload

    def test_chat_payload_with_options(self) -> None:
        """Generation parameters appear under the 'options' key."""
        request = _make_chat_request(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
        )
        payload = self.handler._build_chat_payload(request)

        assert "options" in payload
        options = payload["options"]
        assert isinstance(options, dict)
        assert options["temperature"] == 0.7
        assert options["top_p"] == 0.9
        assert options["num_predict"] == 256

    def test_chat_payload_with_system_prompt(self) -> None:
        """system_prompt produces 'system' key in payload."""
        request = _make_chat_request(system_prompt="You are a helpful assistant.")
        payload = self.handler._build_chat_payload(request)

        assert payload["system"] == "You are a helpful assistant."

    def test_chat_payload_without_system_prompt(self) -> None:
        """No 'system' key when system_prompt is None."""
        request = _make_chat_request(system_prompt=None)
        payload = self.handler._build_chat_payload(request)

        assert "system" not in payload

    def test_chat_payload_with_tools(self) -> None:
        """Tool definitions produce 'tools' key in payload."""
        tool = _make_tool_definition()
        request = _make_chat_request(tools=(tool,))
        payload = self.handler._build_chat_payload(request)

        assert "tools" in payload
        tools_val = payload["tools"]
        assert isinstance(tools_val, list)
        assert len(tools_val) == 1
        tool_item = tools_val[0]
        assert isinstance(tool_item, dict)
        assert tool_item["type"] == "function"
        func_val = tool_item["function"]
        assert isinstance(func_val, dict)
        assert func_val["name"] == "search"
        assert func_val["description"] == "Search the web"

    def test_chat_payload_without_tools(self) -> None:
        """No 'tools' key when tools tuple is empty."""
        request = _make_chat_request(tools=())
        payload = self.handler._build_chat_payload(request)

        assert "tools" not in payload


@pytest.mark.unit
class TestBuildGeneratePayload:
    """Tests for HandlerLlmOllama._build_generate_payload()."""

    def setup_method(self) -> None:
        """Create a handler instance for each test."""
        self.handler = HandlerLlmOllama()

    def test_minimal_generate_payload(self) -> None:
        """Minimal generate payload has model, prompt, and stream=False."""
        request = _make_completion_request()
        payload = self.handler._build_generate_payload(request)

        assert payload["model"] == "llama3"
        assert payload["prompt"] == "Once upon a time"
        assert payload["stream"] is False
        assert "options" not in payload

    def test_generate_payload_with_options(self) -> None:
        """Generation parameters appear under the 'options' key."""
        request = _make_completion_request(
            temperature=0.5,
            top_p=0.8,
            max_tokens=100,
        )
        payload = self.handler._build_generate_payload(request)

        assert "options" in payload
        options = payload["options"]
        assert isinstance(options, dict)
        assert options["temperature"] == 0.5
        assert options["top_p"] == 0.8
        assert options["num_predict"] == 100

    def test_generate_payload_no_options(self) -> None:
        """No 'options' key when all generation params are None/empty."""
        request = _make_completion_request(
            temperature=None,
            top_p=None,
            max_tokens=None,
        )
        payload = self.handler._build_generate_payload(request)

        assert "options" not in payload
