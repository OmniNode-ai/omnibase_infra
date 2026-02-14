# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerLlmOpenaiCompatible.

Tests cover the OpenAI-compatible inference handler's:
    - URL building (CHAT_COMPLETION vs COMPLETION routing)
    - Payload construction (messages, system_prompt, tools, tool_choice)
    - Tool serialization round-trip
    - Response parsing (text, tool_calls, usage, finish_reason)
    - Error propagation from transport layer
    - Auth header injection and absence
    - Unknown finish_reason fallback to UNKNOWN
    - Empty/malformed response handling

All tests mock ``_execute_llm_http_call`` on the transport to isolate
handler translation logic from HTTP transport behaviour.

Related:
    - HandlerLlmOpenaiCompatible: The handler under test
    - MixinLlmHttpTransport: Transport mocked at its boundary
    - OMN-2109: Phase 9 inference handler tests
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import httpx
import pytest

from omnibase_infra.enums import (
    EnumInfraTransportType,
    EnumLlmFinishReason,
    EnumLlmOperationType,
)
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraRateLimitedError,
    InfraRequestRejectedError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.models.errors.model_timeout_error_context import (
    ModelTimeoutErrorContext,
)
from omnibase_core.types import JsonType
from omnibase_infra.nodes.effects.models import (
    ModelLlmFunctionDef,
    ModelLlmToolChoice,
    ModelLlmToolDefinition,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
    _parse_tool_calls,
    _parse_usage,
    _serialize_tool_choice,
    _serialize_tool_definition,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "http://localhost:8000"
_MODEL = "qwen2.5-coder-14b"
_CORRELATION_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_transport() -> MagicMock:
    """Create a MagicMock transport with _execute_llm_http_call as AsyncMock."""
    transport = MagicMock(spec=MixinLlmHttpTransport)
    transport._execute_llm_http_call = AsyncMock(return_value={})
    transport._http_client = None
    transport._owns_http_client = True
    return transport


def _make_handler(transport: MagicMock | None = None) -> HandlerLlmOpenaiCompatible:
    """Create a handler with a mock transport."""
    if transport is None:
        transport = _make_transport()
    return HandlerLlmOpenaiCompatible(transport)


def _make_chat_request(**overrides: Any) -> ModelLlmInferenceRequest:
    """Build a valid CHAT_COMPLETION request with sensible defaults."""
    defaults: dict[str, Any] = {
        "base_url": _BASE_URL,
        "model": _MODEL,
        "operation_type": EnumLlmOperationType.CHAT_COMPLETION,
        "messages": ({"role": "user", "content": "Hello"},),
    }
    defaults.update(overrides)
    return ModelLlmInferenceRequest(**defaults)


def _make_completion_request(**overrides: Any) -> ModelLlmInferenceRequest:
    """Build a valid COMPLETION request with sensible defaults."""
    defaults: dict[str, Any] = {
        "base_url": _BASE_URL,
        "model": _MODEL,
        "operation_type": EnumLlmOperationType.COMPLETION,
        "prompt": "Once upon a time",
    }
    defaults.update(overrides)
    return ModelLlmInferenceRequest(**defaults)


def _make_tool_definition(
    name: str = "get_weather",
    description: str = "Get current weather for a city.",
    parameters: dict[str, Any] | None = None,
) -> ModelLlmToolDefinition:
    """Create a tool definition for testing."""
    if parameters is None:
        parameters = {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        }
    return ModelLlmToolDefinition(
        function=ModelLlmFunctionDef(
            name=name,
            description=description,
            parameters=parameters,
        ),
    )


def _make_openai_chat_response(
    content: str = "Hello back!",
    finish_reason: str = "stop",
    provider_id: str = "chatcmpl-abc123",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict[str, Any]:
    """Build a standard OpenAI chat completion response."""
    return {
        "id": provider_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            },
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_openai_tool_call_response(
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "tool_calls",
) -> dict[str, Any]:
    """Build an OpenAI chat completion response with tool calls."""
    if tool_calls is None:
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                },
            },
        ]
    return {
        "id": "chatcmpl-toolcall",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            },
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        },
    }


def _make_error_context() -> ModelInfraErrorContext:
    """Create a minimal error context for exception construction."""
    return ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.HTTP,
        operation="test",
    )


# ---------------------------------------------------------------------------
# URL Building Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildUrl:
    """Tests for HandlerLlmOpenaiCompatible._build_url()."""

    def test_chat_completion_url(self) -> None:
        """CHAT_COMPLETION appends /v1/chat/completions."""
        request = _make_chat_request()
        url = HandlerLlmOpenaiCompatible._build_url(request)
        assert url == f"{_BASE_URL}/v1/chat/completions"

    def test_completion_url(self) -> None:
        """COMPLETION appends /v1/completions."""
        request = _make_completion_request()
        url = HandlerLlmOpenaiCompatible._build_url(request)
        assert url == f"{_BASE_URL}/v1/completions"

    def test_trailing_slash_stripped(self) -> None:
        """Trailing slash on base_url does not produce double slash."""
        request = _make_chat_request(base_url="http://host:8000/")
        url = HandlerLlmOpenaiCompatible._build_url(request)
        assert url == "http://host:8000/v1/chat/completions"
        assert "//" not in url.split("://")[1]

    def test_unsupported_operation_type_raises_value_error(self) -> None:
        """Unsupported operation_type raises ValueError."""
        # EMBEDDING is not supported by the OpenAI handler
        # We can't construct a request with EMBEDDING and messages, so test
        # the static method directly with a mock request
        mock_request = MagicMock()
        mock_request.operation_type = EnumLlmOperationType.EMBEDDING
        mock_request.base_url = _BASE_URL

        with pytest.raises(ValueError, match="Unsupported operation type"):
            HandlerLlmOpenaiCompatible._build_url(mock_request)


# ---------------------------------------------------------------------------
# Payload Building Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildPayload:
    """Tests for HandlerLlmOpenaiCompatible._build_payload()."""

    def test_chat_completion_minimal(self) -> None:
        """Minimal chat payload has model and messages."""
        request = _make_chat_request()
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert payload["model"] == _MODEL
        messages = payload["messages"]
        assert isinstance(messages, list)
        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg, dict)
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"

    def test_completion_uses_prompt_field(self) -> None:
        """COMPLETION request uses prompt field instead of messages."""
        request = _make_completion_request(prompt="Complete this")
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert payload["model"] == _MODEL
        assert payload["prompt"] == "Complete this"
        assert "messages" not in payload

    def test_system_prompt_injected_as_first_message(self) -> None:
        """system_prompt is prepended as a system role message."""
        request = _make_chat_request(
            system_prompt="You are a helpful assistant.",
        )
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        messages = payload["messages"]
        assert isinstance(messages, list)
        assert len(messages) == 2
        system_msg = messages[0]
        assert isinstance(system_msg, dict)
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "You are a helpful assistant."
        user_msg = messages[1]
        assert isinstance(user_msg, dict)
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "Hello"

    def test_system_prompt_none_no_system_message(self) -> None:
        """No system message when system_prompt is None."""
        request = _make_chat_request(system_prompt=None)
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        messages = payload["messages"]
        assert isinstance(messages, list)
        assert len(messages) == 1
        first_msg = messages[0]
        assert isinstance(first_msg, dict)
        assert first_msg["role"] == "user"

    def test_optional_params_included_when_set(self) -> None:
        """max_tokens, temperature, top_p, stop included when set."""
        request = _make_chat_request(
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            stop=("END", "STOP"),
        )
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert payload["max_tokens"] == 256
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9
        assert payload["stop"] == ["END", "STOP"]

    def test_optional_params_excluded_when_none(self) -> None:
        """max_tokens, temperature, top_p excluded when None; stop excluded when empty."""
        request = _make_chat_request(
            max_tokens=None,
            temperature=None,
            top_p=None,
        )
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert "max_tokens" not in payload
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "stop" not in payload


# ---------------------------------------------------------------------------
# Tool Serialization Round-Trip Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolSerializationRoundTrip:
    """Tests for tool definition and tool choice serialization."""

    def test_tool_definition_serialization(self) -> None:
        """ModelLlmToolDefinition serializes to correct OpenAI wire format."""
        params = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["metric", "imperial"]},
            },
            "required": ["city"],
        }
        tool = _make_tool_definition(
            name="get_weather",
            description="Get current weather.",
            parameters=params,
        )
        serialized = _serialize_tool_definition(tool)

        assert serialized == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather.",
                "parameters": params,
            },
        }

    def test_tool_definition_no_description(self) -> None:
        """Tool with no description omits it from serialization."""
        tool = ModelLlmToolDefinition(
            function=ModelLlmFunctionDef(name="ping"),
        )
        serialized = _serialize_tool_definition(tool)

        func = serialized["function"]
        assert isinstance(func, dict)
        assert func["name"] == "ping"
        assert "description" not in func

    def test_tool_definition_no_parameters(self) -> None:
        """Tool with no parameters omits parameters from serialization."""
        tool = ModelLlmToolDefinition(
            function=ModelLlmFunctionDef(name="noop", description="Does nothing"),
        )
        serialized = _serialize_tool_definition(tool)

        func = serialized["function"]
        assert isinstance(func, dict)
        assert "parameters" not in func

    def test_tools_in_payload_round_trip(self) -> None:
        """request.tools -> payload JSON matches expected schema exactly."""
        params = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }
        tool = _make_tool_definition(
            name="search",
            description="Search the web",
            parameters=params,
        )
        request = _make_chat_request(tools=(tool,))
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert "tools" in payload
        tools_val = payload["tools"]
        assert isinstance(tools_val, list)
        assert len(tools_val) == 1
        assert tools_val[0] == {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": params,
            },
        }

    def test_multiple_tools_serialized(self) -> None:
        """Multiple tools are serialized in order."""
        tool_a = _make_tool_definition(name="alpha", description="First")
        tool_b = _make_tool_definition(name="beta", description="Second")
        request = _make_chat_request(tools=(tool_a, tool_b))
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        tools_val = payload["tools"]
        assert isinstance(tools_val, list)
        assert len(tools_val) == 2
        tool_0 = tools_val[0]
        tool_1 = tools_val[1]
        assert isinstance(tool_0, dict)
        assert isinstance(tool_1, dict)
        func_a = tool_0["function"]
        func_b = tool_1["function"]
        assert isinstance(func_a, dict)
        assert isinstance(func_b, dict)
        assert func_a["name"] == "alpha"
        assert func_b["name"] == "beta"


# ---------------------------------------------------------------------------
# Tool Choice Wire Format Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolChoiceWireFormat:
    """Tests for all 4 tool_choice modes producing correct wire format."""

    def test_mode_auto(self) -> None:
        """mode='auto' produces string 'auto'."""
        choice = ModelLlmToolChoice(mode="auto")
        result = _serialize_tool_choice(choice)
        assert result == "auto"

    def test_mode_none(self) -> None:
        """mode='none' produces string 'none'."""
        choice = ModelLlmToolChoice(mode="none")
        result = _serialize_tool_choice(choice)
        assert result == "none"

    def test_mode_required(self) -> None:
        """mode='required' produces string 'required'."""
        choice = ModelLlmToolChoice(mode="required")
        result = _serialize_tool_choice(choice)
        assert result == "required"

    def test_mode_function(self) -> None:
        """mode='function' produces structured dict with function name."""
        choice = ModelLlmToolChoice(mode="function", function_name="get_weather")
        result = _serialize_tool_choice(choice)
        assert result == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_tool_choice_in_payload(self) -> None:
        """tool_choice is included in payload when set."""
        tool = _make_tool_definition()
        choice = ModelLlmToolChoice(mode="auto")
        request = _make_chat_request(tools=(tool,), tool_choice=choice)
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert payload["tool_choice"] == "auto"

    def test_tool_choice_function_in_payload(self) -> None:
        """tool_choice=function in payload produces correct structure."""
        tool = _make_tool_definition()
        choice = ModelLlmToolChoice(mode="function", function_name="get_weather")
        request = _make_chat_request(tools=(tool,), tool_choice=choice)
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }


# ---------------------------------------------------------------------------
# Response Parsing Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandleChatCompletion:
    """Tests for handle() with CHAT_COMPLETION operations."""

    @pytest.mark.asyncio
    async def test_text_response_success(self) -> None:
        """Valid text response is parsed correctly."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response(
            content="Hello back!",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
        )

        resp = await handler.handle(
            _make_chat_request(),
            correlation_id=_CORRELATION_ID,
        )

        assert resp.generated_text == "Hello back!"
        assert resp.model_used == _MODEL
        assert resp.finish_reason == EnumLlmFinishReason.STOP
        assert resp.truncated is False
        assert resp.tool_calls == ()
        assert resp.operation_type == EnumLlmOperationType.CHAT_COMPLETION
        assert resp.status == "success"

    @pytest.mark.asyncio
    async def test_tool_call_response_parsed(self) -> None:
        """Tool call response produces parsed tool calls with no text."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = (
            _make_openai_tool_call_response()
        )

        resp = await handler.handle(
            _make_chat_request(),
            correlation_id=_CORRELATION_ID,
        )

        assert resp.generated_text is None
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "call_abc123"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"city": "London"}'
        assert resp.finish_reason == EnumLlmFinishReason.TOOL_CALLS

    @pytest.mark.asyncio
    async def test_tool_calls_discard_content_xor_invariant(self) -> None:
        """When BOTH content and tool_calls are present, content is discarded."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = {
            "id": "chatcmpl-both",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I will call a tool",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "test"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        resp = await handler.handle(
            _make_chat_request(),
            correlation_id=_CORRELATION_ID,
        )

        assert resp.generated_text is None
        assert len(resp.tool_calls) == 1
        assert resp.finish_reason == EnumLlmFinishReason.TOOL_CALLS

    @pytest.mark.asyncio
    async def test_usage_parsing(self) -> None:
        """Usage tokens are parsed from the response."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response(
            prompt_tokens=100,
            completion_tokens=50,
        )

        resp = await handler.handle(
            _make_chat_request(),
            correlation_id=_CORRELATION_ID,
        )

        assert resp.usage.tokens_input == 100
        assert resp.usage.tokens_output == 50
        assert resp.usage.tokens_total == 150

    @pytest.mark.asyncio
    async def test_provider_id_propagated(self) -> None:
        """Provider ID from response is included in the result."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response(
            provider_id="chatcmpl-xyz789",
        )

        resp = await handler.handle(
            _make_chat_request(),
            correlation_id=_CORRELATION_ID,
        )

        assert resp.provider_id == "chatcmpl-xyz789"

    @pytest.mark.asyncio
    async def test_correlation_id_propagated(self) -> None:
        """Correlation ID is preserved in the response."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response()

        resp = await handler.handle(
            _make_chat_request(),
            correlation_id=_CORRELATION_ID,
        )

        assert resp.correlation_id == _CORRELATION_ID

    @pytest.mark.asyncio
    async def test_correlation_id_auto_generated_when_none(self) -> None:
        """Correlation ID is auto-generated when None."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response()

        resp = await handler.handle(
            _make_chat_request(),
            correlation_id=None,
        )

        assert resp.correlation_id is not None

    @pytest.mark.asyncio
    async def test_truncated_true_on_length_finish(self) -> None:
        """finish_reason='length' sets truncated=True."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response(
            finish_reason="length",
        )

        resp = await handler.handle(
            _make_chat_request(),
            correlation_id=_CORRELATION_ID,
        )

        assert resp.finish_reason == EnumLlmFinishReason.LENGTH
        assert resp.truncated is True


@pytest.mark.unit
class TestHandleCompletion:
    """Tests for handle() with COMPLETION operations."""

    @pytest.mark.asyncio
    async def test_completion_text_response(self) -> None:
        """COMPLETION response text is in choice.text."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = {
            "id": "cmpl-abc",
            "choices": [
                {
                    "text": "Once upon a time, there was a dragon.",
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }

        resp = await handler.handle(
            _make_completion_request(),
            correlation_id=_CORRELATION_ID,
        )

        assert resp.generated_text == "Once upon a time, there was a dragon."
        assert resp.operation_type == EnumLlmOperationType.COMPLETION
        assert resp.finish_reason == EnumLlmFinishReason.STOP

    @pytest.mark.asyncio
    async def test_completion_url_routing(self) -> None:
        """COMPLETION uses /v1/completions endpoint."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = {
            "id": "cmpl-123",
            "choices": [{"text": "result", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

        await handler.handle(
            _make_completion_request(),
            correlation_id=_CORRELATION_ID,
        )

        call_args = transport._execute_llm_http_call.call_args
        url = call_args.kwargs.get("url") or call_args.args[0]
        assert url.endswith("/v1/completions")

    @pytest.mark.asyncio
    async def test_chat_completion_url_routing(self) -> None:
        """CHAT_COMPLETION uses /v1/chat/completions endpoint."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response()

        await handler.handle(
            _make_chat_request(),
            correlation_id=_CORRELATION_ID,
        )

        call_args = transport._execute_llm_http_call.call_args
        url = call_args.kwargs.get("url") or call_args.args[0]
        assert url.endswith("/v1/chat/completions")


# ---------------------------------------------------------------------------
# Finish Reason Mapping Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFinishReasonMapping:
    """Tests for finish_reason string -> EnumLlmFinishReason mapping."""

    @pytest.mark.asyncio
    async def test_stop_maps_to_STOP(self) -> None:
        """'stop' -> EnumLlmFinishReason.STOP."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response(
            finish_reason="stop",
        )
        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )
        assert resp.finish_reason == EnumLlmFinishReason.STOP

    @pytest.mark.asyncio
    async def test_length_maps_to_LENGTH(self) -> None:
        """'length' -> EnumLlmFinishReason.LENGTH."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response(
            finish_reason="length",
        )
        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )
        assert resp.finish_reason == EnumLlmFinishReason.LENGTH

    @pytest.mark.asyncio
    async def test_content_filter_maps_to_CONTENT_FILTER(self) -> None:
        """'content_filter' -> EnumLlmFinishReason.CONTENT_FILTER."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response(
            content="",
            finish_reason="content_filter",
        )
        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )
        assert resp.finish_reason == EnumLlmFinishReason.CONTENT_FILTER

    @pytest.mark.asyncio
    async def test_tool_calls_maps_to_TOOL_CALLS(self) -> None:
        """'tool_calls' -> EnumLlmFinishReason.TOOL_CALLS."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_tool_call_response(
            finish_reason="tool_calls",
        )
        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )
        assert resp.finish_reason == EnumLlmFinishReason.TOOL_CALLS

    @pytest.mark.asyncio
    async def test_function_call_maps_to_TOOL_CALLS(self) -> None:
        """'function_call' (legacy) -> EnumLlmFinishReason.TOOL_CALLS."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_tool_call_response(
            finish_reason="function_call",
        )
        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )
        assert resp.finish_reason == EnumLlmFinishReason.TOOL_CALLS

    @pytest.mark.asyncio
    async def test_unknown_finish_reason_maps_to_UNKNOWN(self) -> None:
        """An unrecognized finish_reason string maps to UNKNOWN, not crash."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response(
            content="",
            finish_reason="new_provider_value_xyz",
        )
        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )
        assert resp.finish_reason == EnumLlmFinishReason.UNKNOWN

    @pytest.mark.asyncio
    async def test_empty_finish_reason_maps_to_UNKNOWN(self) -> None:
        """Empty string finish_reason maps to UNKNOWN."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response(
            content="",
            finish_reason="",
        )
        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )
        assert resp.finish_reason == EnumLlmFinishReason.UNKNOWN

    @pytest.mark.asyncio
    async def test_none_finish_reason_maps_to_UNKNOWN(self) -> None:
        """None/missing finish_reason maps to UNKNOWN."""
        transport = _make_transport()
        handler = _make_handler(transport)
        response = _make_openai_chat_response(content="")
        # Remove finish_reason
        response["choices"][0].pop("finish_reason", None)
        transport._execute_llm_http_call.return_value = response
        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )
        assert resp.finish_reason == EnumLlmFinishReason.UNKNOWN


# ---------------------------------------------------------------------------
# Error Propagation Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestErrorPropagation:
    """Tests for error mapping: HTTP status -> exception type.

    The transport layer (_execute_llm_http_call) raises typed exceptions.
    These tests verify the handler propagates them correctly.
    """

    @pytest.mark.asyncio
    async def test_401_raises_auth_error(self) -> None:
        """401 from transport -> InfraAuthenticationError."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.side_effect = InfraAuthenticationError(
            "Auth failed (401)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraAuthenticationError):
            await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

    @pytest.mark.asyncio
    async def test_429_raises_rate_limited_error(self) -> None:
        """429 from transport -> InfraRateLimitedError."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.side_effect = InfraRateLimitedError(
            "Rate limited (429)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraRateLimitedError):
            await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

    @pytest.mark.asyncio
    async def test_400_raises_request_rejected_error(self) -> None:
        """400 from transport -> InfraRequestRejectedError."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.side_effect = InfraRequestRejectedError(
            "Request rejected (400)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraRequestRejectedError):
            await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

    @pytest.mark.asyncio
    async def test_404_raises_config_error(self) -> None:
        """404 from transport -> ProtocolConfigurationError."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.side_effect = ProtocolConfigurationError(
            "Not found (404)",
            context=_make_error_context(),
        )

        with pytest.raises(ProtocolConfigurationError):
            await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

    @pytest.mark.asyncio
    async def test_500_raises_unavailable_error(self) -> None:
        """500 from transport -> InfraUnavailableError."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.side_effect = InfraUnavailableError(
            "Server error (500)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraUnavailableError):
            await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

    @pytest.mark.asyncio
    async def test_connection_error_propagated(self) -> None:
        """Connection failure from transport -> InfraConnectionError."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.side_effect = InfraConnectionError(
            "Connection refused",
            context=_make_error_context(),
        )

        with pytest.raises(InfraConnectionError):
            await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

    @pytest.mark.asyncio
    async def test_timeout_error_propagated(self) -> None:
        """Timeout from transport -> InfraTimeoutError."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.side_effect = InfraTimeoutError(
            "Request timed out",
            context=ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation="test",
            ),
        )

        with pytest.raises(InfraTimeoutError):
            await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)


# ---------------------------------------------------------------------------
# Auth Header Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAuthHeaderHandling:
    """Tests for api_key -> Authorization header injection."""

    @pytest.mark.asyncio
    async def test_no_api_key_uses_default_transport(self) -> None:
        """When api_key is None, transport's default client is used."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response()

        request = _make_chat_request(api_key=None)
        await handler.handle(request, correlation_id=_CORRELATION_ID)

        # The transport's _execute_llm_http_call should be called directly
        transport._execute_llm_http_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_api_key_injects_auth_client(self) -> None:
        """When api_key is provided, a temporary auth client is injected."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_openai_chat_response()

        # Capture the http_client that is set on the transport during the call
        captured_client: httpx.AsyncClient | None = None

        async def _capture_client(**kwargs: Any) -> dict[str, Any]:
            nonlocal captured_client
            captured_client = transport._http_client
            return _make_openai_chat_response()

        transport._execute_llm_http_call.side_effect = _capture_client

        request = _make_chat_request(api_key="sk-test-key-123")
        await handler.handle(request, correlation_id=_CORRELATION_ID)

        # Verify the transport's _execute_llm_http_call was called
        transport._execute_llm_http_call.assert_awaited_once()

        # Verify an httpx.AsyncClient with the Authorization header was injected
        assert captured_client is not None
        assert isinstance(captured_client, httpx.AsyncClient)
        assert captured_client.headers["authorization"] == "Bearer sk-test-key-123"
        assert captured_client.is_closed

    @pytest.mark.asyncio
    async def test_empty_api_key_raises_value_error(self) -> None:
        """Empty string api_key raises ValueError (misconfiguration)."""
        transport = _make_transport()
        handler = _make_handler(transport)

        request = _make_chat_request(api_key="")
        with pytest.raises(ValueError, match="api_key is an empty string"):
            await handler.handle(request, correlation_id=_CORRELATION_ID)

    @pytest.mark.asyncio
    async def test_auth_client_restored_after_error(self) -> None:
        """Transport client references are restored even when call fails."""
        transport = _make_transport()
        original_client = transport._http_client
        original_owns = transport._owns_http_client
        handler = _make_handler(transport)

        transport._execute_llm_http_call.side_effect = InfraConnectionError(
            "Connection refused",
            context=_make_error_context(),
        )

        request = _make_chat_request(api_key="sk-test-key")
        with pytest.raises(InfraConnectionError):
            await handler.handle(request, correlation_id=_CORRELATION_ID)

        # Verify original references are restored
        assert transport._http_client is original_client
        assert transport._owns_http_client is original_owns


# ---------------------------------------------------------------------------
# Empty/Malformed Response Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmptyMalformedResponses:
    """Tests for edge cases in response parsing."""

    @pytest.mark.asyncio
    async def test_empty_choices_produces_empty_response(self) -> None:
        """Empty choices array produces UNKNOWN finish_reason, no text."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = {
            "id": "chatcmpl-empty",
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        }

        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        assert resp.generated_text is None
        assert resp.finish_reason == EnumLlmFinishReason.UNKNOWN
        assert resp.tool_calls == ()

    @pytest.mark.asyncio
    async def test_no_choices_key_produces_empty_response(self) -> None:
        """Missing choices key produces UNKNOWN finish_reason."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = {"id": "chatcmpl-noc"}

        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        assert resp.generated_text is None
        assert resp.finish_reason == EnumLlmFinishReason.UNKNOWN

    @pytest.mark.asyncio
    async def test_malformed_choice_produces_empty_response(self) -> None:
        """Non-dict choice entry is treated as empty."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = {
            "id": "chatcmpl-bad",
            "choices": ["not a dict"],
        }

        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        assert resp.generated_text is None
        assert resp.finish_reason == EnumLlmFinishReason.UNKNOWN

    @pytest.mark.asyncio
    async def test_none_content_in_message(self) -> None:
        """content=None in message produces generated_text=None."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = {
            "id": "chatcmpl-none",
            "choices": [
                {
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        }

        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        assert resp.generated_text is None


# ---------------------------------------------------------------------------
# Module-Level Helper Function Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseToolCalls:
    """Tests for _parse_tool_calls module-level function."""

    def test_valid_tool_call_parsed(self) -> None:
        """Valid tool call dict produces ModelLlmToolCall."""
        raw: list[JsonType] = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "test"}',
                },
            },
        ]
        result = _parse_tool_calls(raw)
        assert len(result) == 1
        assert result[0].id == "call_123"
        assert result[0].function.name == "search"
        assert result[0].function.arguments == '{"query": "test"}'

    def test_missing_id_skipped(self) -> None:
        """Tool call without id is skipped."""
        raw: list[JsonType] = [
            {
                "function": {
                    "name": "search",
                    "arguments": "{}",
                },
            },
        ]
        result = _parse_tool_calls(raw)
        assert result == ()

    def test_missing_function_skipped(self) -> None:
        """Tool call without function key is skipped."""
        raw: list[JsonType] = [
            {
                "id": "call_123",
            },
        ]
        result = _parse_tool_calls(raw)
        assert result == ()

    def test_missing_function_name_skipped(self) -> None:
        """Tool call with empty function name is skipped."""
        raw: list[JsonType] = [
            {
                "id": "call_123",
                "function": {
                    "name": "",
                    "arguments": "{}",
                },
            },
        ]
        result = _parse_tool_calls(raw)
        assert result == ()

    def test_non_dict_entry_skipped(self) -> None:
        """Non-dict entries in the tool calls list are skipped."""
        raw: list[JsonType] = ["not_a_dict", 42]
        result = _parse_tool_calls(raw)
        assert result == ()

    def test_multiple_tool_calls(self) -> None:
        """Multiple valid tool calls are parsed in order."""
        raw: list[JsonType] = [
            {
                "id": "call_1",
                "function": {"name": "alpha", "arguments": '{"a":1}'},
            },
            {
                "id": "call_2",
                "function": {"name": "beta", "arguments": '{"b":2}'},
            },
        ]
        result = _parse_tool_calls(raw)
        assert len(result) == 2
        assert result[0].function.name == "alpha"
        assert result[1].function.name == "beta"

    def test_missing_arguments_defaults_to_empty_string(self) -> None:
        """Missing arguments field defaults to empty string."""
        raw: list[JsonType] = [
            {
                "id": "call_no_args",
                "function": {"name": "noop"},
            },
        ]
        result = _parse_tool_calls(raw)
        assert len(result) == 1
        assert result[0].function.arguments == ""


@pytest.mark.unit
class TestParseUsage:
    """Tests for _parse_usage module-level function."""

    def test_full_usage_parsed(self) -> None:
        """All usage fields are parsed correctly."""
        raw: JsonType = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        usage = _parse_usage(raw)
        assert usage.tokens_input == 100
        assert usage.tokens_output == 50
        assert usage.tokens_total == 150

    def test_none_usage_returns_defaults(self) -> None:
        """None usage returns zero-valued defaults."""
        usage = _parse_usage(None)
        assert usage.tokens_input == 0
        assert usage.tokens_output == 0

    def test_non_dict_usage_returns_defaults(self) -> None:
        """Non-dict usage returns zero-valued defaults."""
        usage = _parse_usage("not a dict")
        assert usage.tokens_input == 0
        assert usage.tokens_output == 0

    def test_missing_fields_default_to_zero(self) -> None:
        """Missing usage fields default to zero."""
        usage = _parse_usage({})
        assert usage.tokens_input == 0
        assert usage.tokens_output == 0

    def test_non_numeric_values_default_to_zero(self) -> None:
        """Non-numeric string values in usage default to zero."""
        usage = _parse_usage(
            {
                "prompt_tokens": "not_a_number",
                "completion_tokens": "also_not",
            }
        )
        assert usage.tokens_input == 0
        assert usage.tokens_output == 0
