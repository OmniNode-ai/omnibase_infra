# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerLlmOpenaiCompatible class methods.

Tests cover:
    - Initialization with transport injection
    - Class-level constants (_FINISH_REASON_MAP, _OPERATION_PATHS)
    - _build_url() for both operation types and unsupported types
    - _build_payload() for CHAT_COMPLETION and COMPLETION requests
    - _build_empty_response() for empty/malformed provider output
    - _parse_response() for various response shapes
    - Module-level helpers: _parse_usage(), _serialize_tool_definition(),
      _serialize_tool_choice(), _parse_tool_calls(), _safe_int(), _safe_int_or_none()

Related:
    - OMN-2107: Phase 7 OpenAI-compatible inference handler
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import (
    EnumLlmFinishReason,
    EnumLlmOperationType,
)
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.nodes.effects.models import (
    ModelLlmFunctionDef,
    ModelLlmToolChoice,
    ModelLlmToolDefinition,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    _FINISH_REASON_MAP,
    _OPERATION_PATHS,
    HandlerLlmOpenaiCompatible,
    _parse_tool_calls,
    _parse_usage,
    _safe_int,
    _safe_int_or_none,
    _serialize_tool_choice,
    _serialize_tool_definition,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_transport() -> MagicMock:
    """Create a mock MixinLlmHttpTransport."""
    return MagicMock(spec=MixinLlmHttpTransport)


def _make_chat_request(
    *,
    model: str = "gpt-4",
    base_url: str = "http://localhost:8000",
    messages: tuple[dict[str, str], ...] | None = None,
    system_prompt: str | None = None,
    tools: tuple[ModelLlmToolDefinition, ...] | None = None,
    tool_choice: ModelLlmToolChoice | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop: tuple[str, ...] | None = None,
) -> ModelLlmInferenceRequest:
    """Create a valid CHAT_COMPLETION request with sensible defaults."""
    if messages is None:
        messages = ({"role": "user", "content": "Hello"},)
    return ModelLlmInferenceRequest(
        base_url=base_url,
        model=model,
        operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        messages=messages,
        system_prompt=system_prompt,
        tools=tools if tools is not None else (),
        tool_choice=tool_choice,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop if stop is not None else (),
    )


def _make_completion_request(
    *,
    model: str = "gpt-4",
    base_url: str = "http://localhost:8000",
    prompt: str = "Once upon a time",
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop: tuple[str, ...] | None = None,
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
        stop=stop if stop is not None else (),
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
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestHandlerLlmOpenaiCompatibleInit:
    """Tests for HandlerLlmOpenaiCompatible initialization."""

    def test_stores_transport(self) -> None:
        """Constructor stores the transport reference."""
        transport = _make_transport()
        handler = HandlerLlmOpenaiCompatible(transport)

        assert handler._transport is transport

    def test_no_handler_type_property(self) -> None:
        """Handler does NOT implement ProtocolHandler/ProtocolMessageHandler.

        Unlike HandlerLlmOllama, this handler operates at the infrastructure
        layer with typed request/response models and does not expose
        handler_type or handler_category properties.
        """
        transport = _make_transport()
        handler = HandlerLlmOpenaiCompatible(transport)

        assert not hasattr(handler, "handler_type")
        assert not hasattr(handler, "handler_category")


# ---------------------------------------------------------------------------
# Tests: Class-level constants
# ---------------------------------------------------------------------------


class TestClassConstants:
    """Tests for module-level constants used by the handler."""

    def test_finish_reason_map_has_stop(self) -> None:
        """'stop' maps to EnumLlmFinishReason.STOP."""
        assert _FINISH_REASON_MAP["stop"] == EnumLlmFinishReason.STOP

    def test_finish_reason_map_has_length(self) -> None:
        """'length' maps to EnumLlmFinishReason.LENGTH."""
        assert _FINISH_REASON_MAP["length"] == EnumLlmFinishReason.LENGTH

    def test_finish_reason_map_has_content_filter(self) -> None:
        """'content_filter' maps to EnumLlmFinishReason.CONTENT_FILTER."""
        assert (
            _FINISH_REASON_MAP["content_filter"] == EnumLlmFinishReason.CONTENT_FILTER
        )

    def test_finish_reason_map_has_tool_calls(self) -> None:
        """'tool_calls' maps to EnumLlmFinishReason.TOOL_CALLS."""
        assert _FINISH_REASON_MAP["tool_calls"] == EnumLlmFinishReason.TOOL_CALLS

    def test_finish_reason_map_has_function_call(self) -> None:
        """'function_call' maps to EnumLlmFinishReason.TOOL_CALLS."""
        assert _FINISH_REASON_MAP["function_call"] == EnumLlmFinishReason.TOOL_CALLS

    def test_finish_reason_map_unknown_key_returns_default(self) -> None:
        """Unknown keys fall through .get() with default UNKNOWN."""
        result = _FINISH_REASON_MAP.get("something_else", EnumLlmFinishReason.UNKNOWN)
        assert result == EnumLlmFinishReason.UNKNOWN

    def test_operation_paths_chat_completion(self) -> None:
        """CHAT_COMPLETION maps to /v1/chat/completions."""
        assert (
            _OPERATION_PATHS[EnumLlmOperationType.CHAT_COMPLETION]
            == "/v1/chat/completions"
        )

    def test_operation_paths_completion(self) -> None:
        """COMPLETION maps to /v1/completions."""
        assert _OPERATION_PATHS[EnumLlmOperationType.COMPLETION] == "/v1/completions"


# ---------------------------------------------------------------------------
# Tests: _build_url()
# ---------------------------------------------------------------------------


class TestBuildUrl:
    """Tests for HandlerLlmOpenaiCompatible._build_url()."""

    def test_chat_completion_url(self) -> None:
        """CHAT_COMPLETION appends /v1/chat/completions."""
        request = _make_chat_request(base_url="http://localhost:8000")
        url = HandlerLlmOpenaiCompatible._build_url(request)

        assert url == "http://localhost:8000/v1/chat/completions"

    def test_completion_url(self) -> None:
        """COMPLETION appends /v1/completions."""
        request = _make_completion_request(base_url="http://localhost:8000")
        url = HandlerLlmOpenaiCompatible._build_url(request)

        assert url == "http://localhost:8000/v1/completions"

    def test_trailing_slash_stripped(self) -> None:
        """Trailing slash on base_url is stripped before appending path."""
        request = _make_chat_request(base_url="http://localhost:8000/")
        url = HandlerLlmOpenaiCompatible._build_url(request)

        assert url == "http://localhost:8000/v1/chat/completions"

    def test_unsupported_operation_type_raises(self) -> None:
        """Unsupported operation type raises ValueError."""
        # Use SimpleNamespace to bypass model validation and provide an
        # operation_type not in _OPERATION_PATHS (EMBEDDING).
        fake_request = SimpleNamespace(
            operation_type=EnumLlmOperationType.EMBEDDING,
            base_url="http://localhost:8000",
        )

        with pytest.raises(ValueError, match="Unsupported operation type"):
            HandlerLlmOpenaiCompatible._build_url(fake_request)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: _build_payload()
# ---------------------------------------------------------------------------


class TestBuildPayload:
    """Tests for HandlerLlmOpenaiCompatible._build_payload()."""

    def test_minimal_chat_payload(self) -> None:
        """Minimal chat payload has model and messages."""
        request = _make_chat_request()
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert payload["model"] == "gpt-4"
        messages_val = payload["messages"]
        assert isinstance(messages_val, list)
        assert len(messages_val) == 1
        msg = messages_val[0]
        assert isinstance(msg, dict)
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"
        # No optional keys
        assert "max_tokens" not in payload
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "stop" not in payload
        assert "tools" not in payload
        assert "tool_choice" not in payload

    def test_chat_payload_with_system_prompt(self) -> None:
        """system_prompt is prepended as a system message."""
        request = _make_chat_request(system_prompt="You are a helpful assistant.")
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        messages_val = payload["messages"]
        assert isinstance(messages_val, list)
        assert len(messages_val) == 2
        assert messages_val[0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        assert messages_val[1]["role"] == "user"

    def test_chat_payload_without_system_prompt(self) -> None:
        """No system message prepended when system_prompt is None."""
        request = _make_chat_request(system_prompt=None)
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        messages_val = payload["messages"]
        assert isinstance(messages_val, list)
        assert len(messages_val) == 1
        assert messages_val[0]["role"] == "user"

    def test_chat_payload_with_generation_params(self) -> None:
        """Generation parameters are top-level keys (OpenAI format)."""
        request = _make_chat_request(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
        )
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9
        assert payload["max_tokens"] == 256

    def test_chat_payload_with_stop_sequences(self) -> None:
        """Stop sequences appear as 'stop' key."""
        request = _make_chat_request(stop=("END", "DONE"))
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert payload["stop"] == ["END", "DONE"]

    def test_chat_payload_without_stop_sequences(self) -> None:
        """No 'stop' key when stop tuple is empty."""
        request = _make_chat_request(stop=())
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert "stop" not in payload

    def test_chat_payload_with_tools(self) -> None:
        """Tool definitions produce 'tools' key."""
        tool = _make_tool_definition()
        request = _make_chat_request(tools=(tool,))
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert "tools" in payload
        tools_val = payload["tools"]
        assert isinstance(tools_val, list)
        assert len(tools_val) == 1
        tool_item = tools_val[0]
        assert isinstance(tool_item, dict)
        assert tool_item["type"] == "function"

    def test_chat_payload_without_tools(self) -> None:
        """No 'tools' key when tools tuple is empty."""
        request = _make_chat_request(tools=())
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert "tools" not in payload

    def test_chat_payload_with_tool_choice_auto(self) -> None:
        """tool_choice='auto' appears in payload."""
        choice = ModelLlmToolChoice(mode="auto")
        request = _make_chat_request(tool_choice=choice)
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert payload["tool_choice"] == "auto"

    def test_chat_payload_without_tool_choice(self) -> None:
        """No 'tool_choice' key when tool_choice is None."""
        request = _make_chat_request(tool_choice=None)
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert "tool_choice" not in payload

    def test_minimal_completion_payload(self) -> None:
        """Minimal completion payload has model and prompt."""
        request = _make_completion_request()
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert payload["model"] == "gpt-4"
        assert payload["prompt"] == "Once upon a time"
        assert "messages" not in payload
        assert "max_tokens" not in payload
        assert "temperature" not in payload

    def test_completion_payload_with_generation_params(self) -> None:
        """Generation parameters are top-level keys for completion."""
        request = _make_completion_request(
            temperature=0.5,
            top_p=0.8,
            max_tokens=100,
        )
        payload = HandlerLlmOpenaiCompatible._build_payload(request)

        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.8
        assert payload["max_tokens"] == 100


# ---------------------------------------------------------------------------
# Tests: _build_empty_response()
# ---------------------------------------------------------------------------


class TestBuildEmptyResponse:
    """Tests for HandlerLlmOpenaiCompatible._build_empty_response()."""

    def test_returns_unknown_finish_reason(self) -> None:
        """Empty response has finish_reason=UNKNOWN."""
        request = _make_chat_request()
        correlation_id = uuid4()
        execution_id = uuid4()

        response = HandlerLlmOpenaiCompatible._build_empty_response(
            request=request,
            correlation_id=correlation_id,
            execution_id=execution_id,
            latency_ms=42.0,
            provider_id_str=None,
        )

        assert response.finish_reason == EnumLlmFinishReason.UNKNOWN

    def test_returns_no_generated_text(self) -> None:
        """Empty response has generated_text=None."""
        request = _make_chat_request()

        response = HandlerLlmOpenaiCompatible._build_empty_response(
            request=request,
            correlation_id=uuid4(),
            execution_id=uuid4(),
            latency_ms=10.0,
            provider_id_str=None,
        )

        assert response.generated_text is None

    def test_returns_empty_usage(self) -> None:
        """Empty response has zero token usage."""
        request = _make_chat_request()

        response = HandlerLlmOpenaiCompatible._build_empty_response(
            request=request,
            correlation_id=uuid4(),
            execution_id=uuid4(),
            latency_ms=10.0,
            provider_id_str=None,
        )

        assert response.usage.tokens_input == 0
        assert response.usage.tokens_output == 0

    def test_preserves_model_used(self) -> None:
        """Empty response uses model from request."""
        request = _make_chat_request(model="my-model")

        response = HandlerLlmOpenaiCompatible._build_empty_response(
            request=request,
            correlation_id=uuid4(),
            execution_id=uuid4(),
            latency_ms=10.0,
            provider_id_str=None,
        )

        assert response.model_used == "my-model"

    def test_preserves_provider_id(self) -> None:
        """Empty response includes provider_id when provided."""
        request = _make_chat_request()

        response = HandlerLlmOpenaiCompatible._build_empty_response(
            request=request,
            correlation_id=uuid4(),
            execution_id=uuid4(),
            latency_ms=10.0,
            provider_id_str="chatcmpl-abc123",
        )

        assert response.provider_id == "chatcmpl-abc123"

    def test_preserves_correlation_and_execution_ids(self) -> None:
        """Empty response preserves correlation_id and execution_id."""
        request = _make_chat_request()
        cid = uuid4()
        eid = uuid4()

        response = HandlerLlmOpenaiCompatible._build_empty_response(
            request=request,
            correlation_id=cid,
            execution_id=eid,
            latency_ms=10.0,
            provider_id_str=None,
        )

        assert response.correlation_id == cid
        assert response.execution_id == eid

    def test_backend_result_is_success(self) -> None:
        """Empty response has backend_result.success=True."""
        request = _make_chat_request()

        response = HandlerLlmOpenaiCompatible._build_empty_response(
            request=request,
            correlation_id=uuid4(),
            execution_id=uuid4(),
            latency_ms=50.0,
            provider_id_str=None,
        )

        assert response.backend_result.success is True


# ---------------------------------------------------------------------------
# Tests: _parse_response()
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for HandlerLlmOpenaiCompatible._parse_response()."""

    def _parse(
        self,
        data: dict,
        *,
        operation_type: EnumLlmOperationType = EnumLlmOperationType.CHAT_COMPLETION,
        model: str = "gpt-4",
    ):
        """Helper to call _parse_response with sensible defaults."""
        if operation_type == EnumLlmOperationType.CHAT_COMPLETION:
            request = _make_chat_request(model=model)
        else:
            request = _make_completion_request(model=model)
        return HandlerLlmOpenaiCompatible._parse_response(
            data=data,
            request=request,
            correlation_id=uuid4(),
            execution_id=uuid4(),
            latency_ms=100.0,
        )

    def test_chat_completion_text_response(self) -> None:
        """Parses a standard chat completion text response."""
        data = {
            "id": "chatcmpl-abc",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        response = self._parse(data)

        assert response.generated_text == "Hello there!"
        assert response.finish_reason == EnumLlmFinishReason.STOP
        assert response.provider_id == "chatcmpl-abc"
        assert response.usage.tokens_input == 10
        assert response.usage.tokens_output == 5

    def test_completion_text_response(self) -> None:
        """Parses a standard completion (text) response."""
        data = {
            "id": "cmpl-xyz",
            "choices": [
                {
                    "text": "in a land far away",
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            },
        }
        response = self._parse(data, operation_type=EnumLlmOperationType.COMPLETION)

        assert response.generated_text == "in a land far away"
        assert response.finish_reason == EnumLlmFinishReason.LENGTH
        assert response.truncated is True

    def test_empty_choices_returns_empty_response(self) -> None:
        """Empty choices array produces empty response."""
        data = {"id": "chatcmpl-abc", "choices": []}
        response = self._parse(data)

        assert response.generated_text is None
        assert response.finish_reason == EnumLlmFinishReason.UNKNOWN

    def test_no_choices_key_returns_empty_response(self) -> None:
        """Missing choices key produces empty response."""
        data = {"id": "chatcmpl-abc"}
        response = self._parse(data)

        assert response.generated_text is None
        assert response.finish_reason == EnumLlmFinishReason.UNKNOWN

    def test_malformed_choice_returns_empty_response(self) -> None:
        """Non-dict choice entry produces empty response."""
        data = {"id": "chatcmpl-abc", "choices": ["not a dict"]}
        response = self._parse(data)

        assert response.generated_text is None
        assert response.finish_reason == EnumLlmFinishReason.UNKNOWN

    def test_unknown_finish_reason_maps_to_unknown(self) -> None:
        """Unknown finish_reason string maps to UNKNOWN."""
        data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "some_new_reason",
                }
            ],
        }
        response = self._parse(data)

        assert response.finish_reason == EnumLlmFinishReason.UNKNOWN

    def test_none_finish_reason_maps_to_unknown(self) -> None:
        """None finish_reason maps to UNKNOWN (empty string fallback)."""
        data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": None,
                }
            ],
        }
        response = self._parse(data)

        assert response.finish_reason == EnumLlmFinishReason.UNKNOWN

    def test_tool_calls_response(self) -> None:
        """Parses a response with tool calls, clearing generated_text."""
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "test"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        response = self._parse(data)

        assert response.generated_text is None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].function.name == "search"
        assert response.finish_reason == EnumLlmFinishReason.TOOL_CALLS

    def test_provider_id_none_when_missing(self) -> None:
        """provider_id is None when 'id' is missing from response."""
        data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
        }
        response = self._parse(data)

        assert response.provider_id is None

    def test_missing_usage_gives_empty_usage(self) -> None:
        """Missing usage block gives zero tokens."""
        data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
        }
        response = self._parse(data)

        assert response.usage.tokens_input == 0
        assert response.usage.tokens_output == 0


# ---------------------------------------------------------------------------
# Tests: Module-level _parse_usage()
# ---------------------------------------------------------------------------


class TestParseUsage:
    """Tests for the module-level _parse_usage() helper."""

    def test_valid_usage(self) -> None:
        """Parses standard usage block."""
        usage = _parse_usage(
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )

        assert usage.tokens_input == 10
        assert usage.tokens_output == 20
        assert usage.tokens_total == 30

    def test_none_returns_empty(self) -> None:
        """None input returns empty usage."""
        usage = _parse_usage(None)

        assert usage.tokens_input == 0
        assert usage.tokens_output == 0

    def test_non_dict_returns_empty(self) -> None:
        """Non-dict input returns empty usage."""
        usage = _parse_usage("not a dict")

        assert usage.tokens_input == 0
        assert usage.tokens_output == 0

    def test_missing_total_tokens(self) -> None:
        """Missing total_tokens is auto-computed by ModelLlmUsage."""
        usage = _parse_usage({"prompt_tokens": 5, "completion_tokens": 3})

        assert usage.tokens_input == 5
        assert usage.tokens_output == 3
        assert usage.tokens_total == 8

    def test_missing_individual_tokens_default_to_zero(self) -> None:
        """Missing prompt_tokens/completion_tokens default to 0."""
        usage = _parse_usage({})

        assert usage.tokens_input == 0
        assert usage.tokens_output == 0


# ---------------------------------------------------------------------------
# Tests: Module-level _safe_int() and _safe_int_or_none()
# ---------------------------------------------------------------------------


class TestSafeInt:
    """Tests for the module-level _safe_int() helper."""

    def test_int_passthrough(self) -> None:
        """Integer values pass through."""
        assert _safe_int(42) == 42

    def test_float_truncated(self) -> None:
        """Float values are truncated to int."""
        assert _safe_int(3.7) == 3

    def test_numeric_string_converted(self) -> None:
        """Numeric string is converted to int."""
        assert _safe_int("100") == 100

    def test_non_numeric_string_returns_default(self) -> None:
        """Non-numeric string returns default."""
        assert _safe_int("abc") == 0

    def test_none_returns_default(self) -> None:
        """None returns default."""
        assert _safe_int(None) == 0

    def test_bool_returns_default(self) -> None:
        """Bool values return default (not treated as int)."""
        assert _safe_int(True) == 0
        assert _safe_int(False) == 0

    def test_custom_default(self) -> None:
        """Custom default is returned for unsupported types."""
        assert _safe_int(None, -1) == -1

    def test_list_returns_default(self) -> None:
        """List returns default."""
        assert _safe_int([1, 2]) == 0


class TestSafeIntOrNone:
    """Tests for the module-level _safe_int_or_none() helper."""

    def test_int_passthrough(self) -> None:
        """Integer values pass through."""
        assert _safe_int_or_none(42) == 42

    def test_float_truncated(self) -> None:
        """Float values are truncated to int."""
        assert _safe_int_or_none(3.7) == 3

    def test_numeric_string_converted(self) -> None:
        """Numeric string is converted to int."""
        assert _safe_int_or_none("100") == 100

    def test_non_numeric_string_returns_none(self) -> None:
        """Non-numeric string returns None."""
        assert _safe_int_or_none("abc") is None

    def test_none_returns_none(self) -> None:
        """None returns None."""
        assert _safe_int_or_none(None) is None

    def test_bool_returns_none(self) -> None:
        """Bool values return None (not treated as int)."""
        assert _safe_int_or_none(True) is None
        assert _safe_int_or_none(False) is None

    def test_list_returns_none(self) -> None:
        """List returns None."""
        assert _safe_int_or_none([1, 2]) is None


# ---------------------------------------------------------------------------
# Tests: Module-level _serialize_tool_definition()
# ---------------------------------------------------------------------------


class TestSerializeToolDefinition:
    """Tests for the module-level _serialize_tool_definition() helper."""

    def test_full_tool_definition(self) -> None:
        """Serializes a tool definition with all fields."""
        tool = _make_tool_definition(name="search", description="Search the web")
        result = _serialize_tool_definition(tool)

        assert result["type"] == "function"
        func = result["function"]
        assert isinstance(func, dict)
        assert func["name"] == "search"
        assert func["description"] == "Search the web"
        assert "parameters" in func

    def test_tool_definition_without_description(self) -> None:
        """Serializes a tool definition without description."""
        tool = ModelLlmToolDefinition(
            function=ModelLlmFunctionDef(
                name="my_tool",
                description="",
                parameters={"type": "object"},
            ),
        )
        result = _serialize_tool_definition(tool)

        assert result["type"] == "function"
        # Empty description is falsy, so it should not appear
        assert "description" not in result["function"]

    def test_tool_definition_without_parameters(self) -> None:
        """Serializes a tool definition without parameters."""
        tool = ModelLlmToolDefinition(
            function=ModelLlmFunctionDef(
                name="my_tool",
                description="A tool",
            ),
        )
        result = _serialize_tool_definition(tool)

        assert result["type"] == "function"
        assert "parameters" not in result["function"]


# ---------------------------------------------------------------------------
# Tests: Module-level _serialize_tool_choice()
# ---------------------------------------------------------------------------


class TestSerializeToolChoice:
    """Tests for the module-level _serialize_tool_choice() helper."""

    def test_auto_mode(self) -> None:
        """mode='auto' serializes to string 'auto'."""
        choice = ModelLlmToolChoice(mode="auto")
        assert _serialize_tool_choice(choice) == "auto"

    def test_none_mode(self) -> None:
        """mode='none' serializes to string 'none'."""
        choice = ModelLlmToolChoice(mode="none")
        assert _serialize_tool_choice(choice) == "none"

    def test_required_mode(self) -> None:
        """mode='required' serializes to string 'required'."""
        choice = ModelLlmToolChoice(mode="required")
        assert _serialize_tool_choice(choice) == "required"

    def test_function_mode(self) -> None:
        """mode='function' serializes to dict with function name."""
        choice = ModelLlmToolChoice(mode="function", function_name="search")
        result = _serialize_tool_choice(choice)

        assert isinstance(result, dict)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"


# ---------------------------------------------------------------------------
# Tests: Module-level _parse_tool_calls()
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    """Tests for the module-level _parse_tool_calls() helper."""

    def test_valid_tool_call(self) -> None:
        """Parses a valid tool call entry."""
        raw = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "test"}'},
            }
        ]
        result = _parse_tool_calls(raw)

        assert len(result) == 1
        assert result[0].id == "call_1"
        assert result[0].function.name == "search"
        assert result[0].function.arguments == '{"q": "test"}'

    def test_multiple_tool_calls(self) -> None:
        """Parses multiple valid tool call entries."""
        raw = [
            {
                "id": "call_1",
                "function": {"name": "search", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "function": {"name": "write", "arguments": '{"text": "hi"}'},
            },
        ]
        result = _parse_tool_calls(raw)

        assert len(result) == 2

    def test_skips_non_dict_entries(self) -> None:
        """Non-dict entries are skipped."""
        raw = ["not a dict", {"id": "call_1", "function": {"name": "search"}}]
        result = _parse_tool_calls(raw)

        assert len(result) == 1

    def test_skips_missing_id(self) -> None:
        """Entries without 'id' are skipped."""
        raw = [{"function": {"name": "search"}}]
        result = _parse_tool_calls(raw)

        assert len(result) == 0

    def test_skips_missing_function(self) -> None:
        """Entries without 'function' are skipped."""
        raw = [{"id": "call_1"}]
        result = _parse_tool_calls(raw)

        assert len(result) == 0

    def test_skips_missing_function_name(self) -> None:
        """Entries where function has no 'name' are skipped."""
        raw = [{"id": "call_1", "function": {"arguments": "{}"}}]
        result = _parse_tool_calls(raw)

        assert len(result) == 0

    def test_empty_list(self) -> None:
        """Empty list returns empty tuple."""
        result = _parse_tool_calls([])

        assert result == ()

    def test_default_arguments(self) -> None:
        """Missing arguments defaults to empty string."""
        raw = [{"id": "call_1", "function": {"name": "search"}}]
        result = _parse_tool_calls(raw)

        assert len(result) == 1
        assert result[0].function.arguments == ""
