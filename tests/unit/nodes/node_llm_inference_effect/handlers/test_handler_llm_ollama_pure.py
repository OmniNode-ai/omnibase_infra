# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for the module-level pure functions in handler_llm_ollama.

Tests the four pure functions extracted to module scope:
    - ``_build_ollama_options``: Maps request generation params to Ollama options.
    - ``_serialize_ollama_messages``: Serializes chat messages to Ollama format.
    - ``_serialize_ollama_tools``: Serializes tool definitions to Ollama format.
    - ``_parse_ollama_tool_calls``: Parses raw tool calls from Ollama responses.

These functions contain no I/O and no class-level state, making them ideal for
fast, isolated unit tests with real Pydantic model instances.

Related:
    - handler_llm_ollama.py: Source of the functions under test
    - OMN-2108: Phase 8 Ollama inference handler
"""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.types import JsonType
from omnibase_infra.nodes.effects.models import (
    ModelLlmFunctionCall,
    ModelLlmFunctionDef,
    ModelLlmInferenceRequest,
    ModelLlmMessage,
    ModelLlmToolCall,
    ModelLlmToolDefinition,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_ollama import (
    _build_ollama_options,
    _parse_ollama_tool_calls,
    _serialize_ollama_messages,
    _serialize_ollama_tools,
)

# Fixed UUID for deterministic ID generation in _parse_ollama_tool_calls tests.
FIXED_UUID = UUID("12345678-1234-5678-1234-567812345678")


def _make_chat_request(
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop: tuple[str, ...] = (),
) -> ModelLlmInferenceRequest:
    """Build a minimal CHAT_COMPLETION request with optional generation params.

    Uses a single user message to satisfy the CHAT_COMPLETION invariant.
    """
    return ModelLlmInferenceRequest(
        base_url="http://localhost:11434",
        model="test-model",
        operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        messages=(ModelLlmMessage(role="user", content="hello"),),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
    )


# ── TestBuildOllamaOptions ──────────────────────────────────────────────


@pytest.mark.unit
class TestBuildOllamaOptions:
    """Tests for ``_build_ollama_options``."""

    def test_empty_options_when_all_none(self) -> None:
        """Request with no temperature/top_p/max_tokens/stop returns empty dict."""
        request = _make_chat_request()
        result = _build_ollama_options(request)
        assert result == {}

    def test_temperature_only(self) -> None:
        """Sets 'temperature' key when temperature is provided."""
        request = _make_chat_request(temperature=0.7)
        result = _build_ollama_options(request)
        assert result == {"temperature": 0.7}

    def test_top_p_only(self) -> None:
        """Sets 'top_p' key when top_p is provided."""
        request = _make_chat_request(top_p=0.9)
        result = _build_ollama_options(request)
        assert result == {"top_p": 0.9}

    def test_max_tokens_maps_to_num_predict(self) -> None:
        """Maps max_tokens to 'num_predict' in Ollama options."""
        request = _make_chat_request(max_tokens=512)
        result = _build_ollama_options(request)
        assert result == {"num_predict": 512}

    def test_stop_sequences_converted_to_list(self) -> None:
        """Tuple of stop strings becomes list under 'stop' key."""
        request = _make_chat_request(stop=("<|end|>", "STOP"))
        result = _build_ollama_options(request)
        assert result == {"stop": ["<|end|>", "STOP"]}

    def test_empty_stop_tuple_excluded(self) -> None:
        """Empty stop=() does not produce 'stop' key."""
        request = _make_chat_request(stop=())
        result = _build_ollama_options(request)
        assert "stop" not in result

    def test_all_options_present(self) -> None:
        """All params set produces complete dict with all keys."""
        request = _make_chat_request(
            temperature=0.5,
            top_p=0.8,
            max_tokens=1024,
            stop=("END",),
        )
        result = _build_ollama_options(request)
        assert result == {
            "temperature": 0.5,
            "top_p": 0.8,
            "num_predict": 1024,
            "stop": ["END"],
        }


# ── TestSerializeOllamaMessages ─────────────────────────────────────────


@pytest.mark.unit
class TestSerializeOllamaMessages:
    """Tests for ``_serialize_ollama_messages``."""

    def test_simple_user_message(self) -> None:
        """Single user message with content serializes correctly."""
        messages = (ModelLlmMessage(role="user", content="What is 2+2?"),)
        result = _serialize_ollama_messages(messages)
        assert result == [{"role": "user", "content": "What is 2+2?"}]

    def test_message_with_none_content(self) -> None:
        """Content=None is omitted from the serialized dict.

        An assistant message with tool_calls but no content is valid.
        """
        tc = ModelLlmToolCall(
            id="call_1",
            function=ModelLlmFunctionCall(name="search", arguments='{"q":"x"}'),
        )
        messages = (ModelLlmMessage(role="assistant", tool_calls=(tc,)),)
        result = _serialize_ollama_messages(messages)
        assert len(result) == 1
        assert "content" not in result[0]

    def test_assistant_message_with_tool_calls(self) -> None:
        """Tool_calls serialized with function.name and parsed arguments dict."""
        tc = ModelLlmToolCall(
            id="call_abc",
            function=ModelLlmFunctionCall(
                name="get_weather",
                arguments='{"city": "London", "units": "metric"}',
            ),
        )
        messages = (
            ModelLlmMessage(
                role="assistant", content="Let me check.", tool_calls=(tc,)
            ),
        )
        result = _serialize_ollama_messages(messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check."
        tool_calls_val = msg["tool_calls"]
        assert isinstance(tool_calls_val, list)
        assert len(tool_calls_val) == 1
        tc_item = tool_calls_val[0]
        assert isinstance(tc_item, dict)
        assert tc_item == {
            "function": {
                "name": "get_weather",
                "arguments": {"city": "London", "units": "metric"},
            },
        }

    def test_tool_message_with_tool_call_id(self) -> None:
        """tool_call_id field included for tool role messages."""
        messages = (
            ModelLlmMessage(
                role="tool",
                content="The weather is sunny.",
                tool_call_id="call_abc",
            ),
        )
        result = _serialize_ollama_messages(messages)
        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_abc"

    def test_malformed_tool_call_arguments_raises_protocol_config_error(self) -> None:
        """Non-JSON arguments raise ProtocolConfigurationError."""
        tc = ModelLlmToolCall(
            id="call_bad",
            function=ModelLlmFunctionCall(name="foo", arguments="not-valid-json{"),
        )
        messages = (ModelLlmMessage(role="assistant", tool_calls=(tc,)),)
        with pytest.raises(ProtocolConfigurationError, match="Malformed JSON"):
            _serialize_ollama_messages(messages)

    def test_multiple_messages_preserve_order(self) -> None:
        """List preserves tuple ordering of input messages."""
        messages = (
            ModelLlmMessage(role="user", content="first"),
            ModelLlmMessage(role="assistant", content="second"),
            ModelLlmMessage(role="user", content="third"),
        )
        result = _serialize_ollama_messages(messages)
        assert len(result) == 3
        assert result[0]["content"] == "first"
        assert result[1]["content"] == "second"
        assert result[2]["content"] == "third"

    def test_empty_messages_tuple(self) -> None:
        """Empty input returns empty list."""
        result = _serialize_ollama_messages(())
        assert result == []


# ── TestSerializeOllamaTools ────────────────────────────────────────────


@pytest.mark.unit
class TestSerializeOllamaTools:
    """Tests for ``_serialize_ollama_tools``."""

    def test_single_tool_with_parameters(self) -> None:
        """Produces type=function with name, description, parameters."""
        params = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        tool = ModelLlmToolDefinition(
            function=ModelLlmFunctionDef(
                name="get_weather",
                description="Get current weather.",
                parameters=params,
            ),
        )
        result = _serialize_ollama_tools((tool,))
        assert len(result) == 1
        assert result[0] == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather.",
                "parameters": params,
            },
        }

    def test_tool_with_empty_description(self) -> None:
        """Description defaults to empty string when not provided."""
        tool = ModelLlmToolDefinition(
            function=ModelLlmFunctionDef(name="noop"),
        )
        result = _serialize_ollama_tools((tool,))
        func_val = result[0]["function"]
        assert isinstance(func_val, dict)
        assert func_val["description"] == ""

    def test_tool_with_empty_parameters(self) -> None:
        """Parameters defaults to empty dict when not provided."""
        tool = ModelLlmToolDefinition(
            function=ModelLlmFunctionDef(name="ping"),
        )
        result = _serialize_ollama_tools((tool,))
        func_val = result[0]["function"]
        assert isinstance(func_val, dict)
        assert func_val["parameters"] == {}

    def test_multiple_tools(self) -> None:
        """List preserves order of multiple tools."""
        tool_a = ModelLlmToolDefinition(
            function=ModelLlmFunctionDef(name="alpha"),
        )
        tool_b = ModelLlmToolDefinition(
            function=ModelLlmFunctionDef(name="beta"),
        )
        result = _serialize_ollama_tools((tool_a, tool_b))
        assert len(result) == 2
        func_a = result[0]["function"]
        assert isinstance(func_a, dict)
        assert func_a["name"] == "alpha"
        func_b = result[1]["function"]
        assert isinstance(func_b, dict)
        assert func_b["name"] == "beta"

    def test_empty_tools_tuple(self) -> None:
        """Empty input returns empty list."""
        result = _serialize_ollama_tools(())
        assert result == []


# ── TestParseOllamaToolCalls ────────────────────────────────────────────


@pytest.mark.unit
class TestParseOllamaToolCalls:
    """Tests for ``_parse_ollama_tool_calls``."""

    def test_none_input_returns_empty_tuple(self) -> None:
        """None returns ()."""
        result = _parse_ollama_tool_calls(None, FIXED_UUID)
        assert result == ()

    def test_empty_list_returns_empty_tuple(self) -> None:
        """[] returns ()."""
        result = _parse_ollama_tool_calls([], FIXED_UUID)
        assert result == ()

    def test_single_tool_call_with_dict_arguments(self) -> None:
        """Dict arguments serialized to compact JSON string."""
        raw: list[dict[str, JsonType]] = [
            {
                "function": {
                    "name": "search",
                    "arguments": {"query": "hello world"},
                },
            },
        ]
        result = _parse_ollama_tool_calls(raw, FIXED_UUID)
        assert len(result) == 1
        tc = result[0]
        assert tc.function.name == "search"
        # Compact JSON (no spaces after separators)
        assert tc.function.arguments == '{"query":"hello world"}'

    def test_tool_call_with_string_arguments(self) -> None:
        """String arguments passed through unchanged."""
        raw: list[dict[str, JsonType]] = [
            {
                "function": {
                    "name": "lookup",
                    "arguments": '{"key": "value"}',
                },
            },
        ]
        result = _parse_ollama_tool_calls(raw, FIXED_UUID)
        assert len(result) == 1
        assert result[0].function.arguments == '{"key": "value"}'

    def test_tool_call_with_none_arguments(self) -> None:
        """None arguments become '{}'."""
        raw: list[dict[str, JsonType]] = [
            {
                "function": {
                    "name": "no_args",
                    "arguments": None,
                },
            },
        ]
        result = _parse_ollama_tool_calls(raw, FIXED_UUID)
        assert len(result) == 1
        assert result[0].function.arguments == "{}"

    def test_tool_call_id_present(self) -> None:
        """Explicit id field is used when provided."""
        raw: list[dict[str, JsonType]] = [
            {
                "id": "call_explicit_42",
                "function": {
                    "name": "action",
                    "arguments": "{}",
                },
            },
        ]
        result = _parse_ollama_tool_calls(raw, FIXED_UUID)
        assert len(result) == 1
        assert result[0].id == "call_explicit_42"

    def test_tool_call_id_missing_generates_deterministic_id(self) -> None:
        """Missing id generates 'ollama-{corr_hex[:8]}-{index}'."""
        raw: list[dict[str, JsonType]] = [
            {
                "function": {
                    "name": "action",
                    "arguments": "{}",
                },
            },
        ]
        result = _parse_ollama_tool_calls(raw, FIXED_UUID)
        assert len(result) == 1
        expected_id = f"ollama-{FIXED_UUID.hex[:8]}-0"
        assert result[0].id == expected_id

    def test_tool_call_with_empty_function_name_skipped(self) -> None:
        """Empty name logs warning and skips the tool call."""
        raw: list[dict[str, JsonType]] = [
            {
                "function": {
                    "name": "",
                    "arguments": "{}",
                },
            },
        ]
        result = _parse_ollama_tool_calls(raw, FIXED_UUID)
        assert result == ()

    def test_tool_call_with_missing_function_key_skipped(self) -> None:
        """Missing 'function' key results in skipped tool call."""
        raw: list[dict[str, JsonType]] = [
            {
                "id": "call_no_func",
                # No "function" key at all
            },
        ]
        result = _parse_ollama_tool_calls(raw, FIXED_UUID)
        assert result == ()

    def test_tool_call_with_non_dict_function_skipped(self) -> None:
        """Non-dict function value results in skipped tool call (empty name)."""
        raw: list[dict[str, JsonType]] = [
            {
                "function": "not_a_dict",
            },
        ]
        result = _parse_ollama_tool_calls(raw, FIXED_UUID)
        assert result == ()

    def test_multiple_tool_calls_indexed_correctly(self) -> None:
        """Index increments correctly for multiple tool calls without ids."""
        raw: list[dict[str, JsonType]] = [
            {
                "function": {
                    "name": "first",
                    "arguments": '{"a":1}',
                },
            },
            {
                "function": {
                    "name": "second",
                    "arguments": '{"b":2}',
                },
            },
            {
                "function": {
                    "name": "third",
                    "arguments": '{"c":3}',
                },
            },
        ]
        result = _parse_ollama_tool_calls(raw, FIXED_UUID)
        assert len(result) == 3
        hex_prefix = FIXED_UUID.hex[:8]
        assert result[0].id == f"ollama-{hex_prefix}-0"
        assert result[0].function.name == "first"
        assert result[1].id == f"ollama-{hex_prefix}-1"
        assert result[1].function.name == "second"
        assert result[2].id == f"ollama-{hex_prefix}-2"
        assert result[2].function.name == "third"
