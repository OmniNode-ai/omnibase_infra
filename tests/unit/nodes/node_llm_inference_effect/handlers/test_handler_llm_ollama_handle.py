# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration-style unit tests for HandlerLlmOllama.handle().

Tests cover CHAT_COMPLETION, COMPLETION, and error paths by mocking at the
``_execute_llm_http_call`` boundary (the single HTTP call point from
MixinLlmHttpTransport). This validates the handler's request building,
response parsing, XOR invariant enforcement, usage parsing, and metadata
propagation without requiring a live Ollama server.

Related:
    - HandlerLlmOllama: The handler under test
    - MixinLlmHttpTransport: HTTP transport mixin mocked at its boundary
    - OMN-2108: Phase 8 Ollama inference handler
"""

from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.nodes.effects.models import (
    ModelLlmInferenceRequest,
    ModelLlmMessage,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_ollama import (
    HandlerLlmOllama,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "http://localhost:11434"
_MODEL = "llama3.2"
_CORRELATION_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_EXECUTION_ID = UUID("11111111-2222-3333-4444-555555555555")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_chat_request(**overrides: Any) -> ModelLlmInferenceRequest:
    """Build a valid CHAT_COMPLETION request with sensible defaults.

    All keyword arguments are forwarded to ``ModelLlmInferenceRequest``,
    overriding the defaults below.
    """
    defaults: dict[str, Any] = {
        "base_url": _BASE_URL,
        "model": _MODEL,
        "operation_type": EnumLlmOperationType.CHAT_COMPLETION,
        "messages": (ModelLlmMessage(role="user", content="Hello"),),
        "correlation_id": _CORRELATION_ID,
        "execution_id": _EXECUTION_ID,
    }
    defaults.update(overrides)
    return ModelLlmInferenceRequest(**defaults)


def _make_completion_request(**overrides: Any) -> ModelLlmInferenceRequest:
    """Build a valid COMPLETION request with sensible defaults.

    COMPLETION requests require ``prompt`` and must NOT have ``messages``,
    ``system_prompt``, ``tools``, or ``tool_choice``.
    """
    defaults: dict[str, Any] = {
        "base_url": _BASE_URL,
        "model": _MODEL,
        "operation_type": EnumLlmOperationType.COMPLETION,
        "prompt": "Once upon a time",
        "correlation_id": _CORRELATION_ID,
        "execution_id": _EXECUTION_ID,
    }
    defaults.update(overrides)
    return ModelLlmInferenceRequest(**defaults)


def _make_handler() -> HandlerLlmOllama:
    """Create a HandlerLlmOllama and replace _execute_llm_http_call with AsyncMock."""
    handler = HandlerLlmOllama()
    handler._execute_llm_http_call = AsyncMock(return_value={})  # type: ignore[method-assign]
    return handler


def _mock_call(handler: HandlerLlmOllama) -> AsyncMock:
    """Return the AsyncMock for _execute_llm_http_call with proper typing."""
    return cast("AsyncMock", handler._execute_llm_http_call)


# ---------------------------------------------------------------------------
# CHAT_COMPLETION tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandleChatCompletion:
    """Tests for the CHAT_COMPLETION operation type."""

    @pytest.mark.asyncio
    async def test_chat_completion_success_text_response(self) -> None:
        """Valid chat response with message.content produces correct fields."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "Hello back!"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 10,
            "prompt_eval_count": 5,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.5]):
            resp = await handler.handle(_make_chat_request())

        assert resp.generated_text == "Hello back!"
        assert resp.model_used == _MODEL
        assert resp.provider_id == "ollama"
        assert resp.finish_reason == EnumLlmFinishReason.STOP
        assert resp.truncated is False
        assert resp.tool_calls == ()
        assert resp.operation_type == EnumLlmOperationType.CHAT_COMPLETION
        assert resp.status == "success"

    @pytest.mark.asyncio
    async def test_chat_completion_success_tool_calls_response(self) -> None:
        """Response with tool_calls in message produces parsed tool calls."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
            },
            "model": _MODEL,
            "eval_count": 8,
            "prompt_eval_count": 12,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.generated_text is None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].function.name == "get_weather"
        assert resp.tool_calls[0].function.arguments == '{"city":"London"}'
        assert resp.finish_reason == EnumLlmFinishReason.TOOL_CALLS

    @pytest.mark.asyncio
    async def test_chat_completion_tool_calls_discard_text_xor(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When BOTH content and tool_calls are present, content is discarded with warning."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {
                "role": "assistant",
                "content": "I will call a tool",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": {"q": "test"},
                        },
                    }
                ],
            },
            "model": _MODEL,
            "eval_count": 5,
            "prompt_eval_count": 3,
        }

        with (
            patch("time.perf_counter", side_effect=[0.0, 0.1]),
            caplog.at_level(logging.WARNING),
        ):
            resp = await handler.handle(_make_chat_request())

        assert resp.generated_text is None
        assert len(resp.tool_calls) == 1
        assert resp.finish_reason == EnumLlmFinishReason.TOOL_CALLS
        assert "Discarding non-empty text content" in caplog.text

    @pytest.mark.asyncio
    async def test_chat_completion_empty_content_string(self) -> None:
        """Empty string content produces generated_text=''."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": ""},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 0,
            "prompt_eval_count": 5,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.generated_text == ""
        assert resp.tool_calls == ()

    @pytest.mark.asyncio
    async def test_chat_completion_none_content(self) -> None:
        """Absent message.content produces generated_text=None, no tool_calls."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 0,
            "prompt_eval_count": 5,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.generated_text is None
        assert resp.tool_calls == ()
        assert resp.finish_reason == EnumLlmFinishReason.STOP

    @pytest.mark.asyncio
    async def test_chat_completion_non_string_content_coerced(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-string content (e.g. int) is coerced to str with a warning."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": 123},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with (
            patch("time.perf_counter", side_effect=[0.0, 0.1]),
            caplog.at_level(logging.WARNING),
        ):
            resp = await handler.handle(_make_chat_request())

        assert resp.generated_text == "123"
        assert "Unexpected content type" in caplog.text

    @pytest.mark.asyncio
    async def test_chat_completion_usage_parsing(self) -> None:
        """eval_count and prompt_eval_count are mapped to ModelLlmUsage."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 42,
            "prompt_eval_count": 100,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.usage.tokens_output == 42
        assert resp.usage.tokens_input == 100
        assert resp.usage.tokens_total == 142

    @pytest.mark.asyncio
    async def test_chat_completion_non_numeric_usage_defaults_to_zero(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """String eval_count/prompt_eval_count default to 0 with debug log."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": "not_a_number",
            "prompt_eval_count": "also_not",
        }

        with (
            patch("time.perf_counter", side_effect=[0.0, 0.1]),
            caplog.at_level(logging.DEBUG),
        ):
            resp = await handler.handle(_make_chat_request())

        assert resp.usage.tokens_output == 0
        assert resp.usage.tokens_input == 0
        assert "Non-numeric usage value" in caplog.text

    @pytest.mark.asyncio
    async def test_chat_completion_model_used_from_response(self) -> None:
        """The 'model' field in the Ollama response is used for model_used."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "hi"},
            "model": "custom-model:latest",
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.model_used == "custom-model:latest"

    @pytest.mark.asyncio
    async def test_chat_completion_model_used_fallback(self) -> None:
        """Missing 'model' in response falls back to request.model."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "hi"},
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.model_used == _MODEL

    @pytest.mark.asyncio
    async def test_chat_completion_done_reason_mapped(self) -> None:
        """Raw done_reason is mapped via _map_finish_reason."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "hi"},
            "model": _MODEL,
            "done_reason": "content_filter",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.finish_reason == EnumLlmFinishReason.CONTENT_FILTER

    @pytest.mark.asyncio
    async def test_chat_completion_truncated_flag_on_length(self) -> None:
        """finish_reason=LENGTH sets truncated=True."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "partial output"},
            "model": _MODEL,
            "done_reason": "length",
            "eval_count": 50,
            "prompt_eval_count": 10,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.finish_reason == EnumLlmFinishReason.LENGTH
        assert resp.truncated is True

    @pytest.mark.asyncio
    async def test_chat_completion_latency_measured(self) -> None:
        """latency_ms is computed from perf_counter difference."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "hi"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with patch("time.perf_counter", side_effect=[1.0, 1.5]):
            resp = await handler.handle(_make_chat_request())

        assert resp.latency_ms == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_chat_completion_correlation_propagated(self) -> None:
        """response.correlation_id matches request.correlation_id."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "hi"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.correlation_id == _CORRELATION_ID

    @pytest.mark.asyncio
    async def test_chat_completion_execution_id_propagated(self) -> None:
        """response.execution_id matches request.execution_id."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "hi"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.execution_id == _EXECUTION_ID


# ---------------------------------------------------------------------------
# COMPLETION tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandleCompletion:
    """Tests for the COMPLETION operation type."""

    @pytest.mark.asyncio
    async def test_completion_success(self) -> None:
        """Mock returns {response: 'text'}, verify generated_text."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "response": "Once upon a time, there was a dragon.",
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 15,
            "prompt_eval_count": 5,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.2]):
            resp = await handler.handle(_make_completion_request())

        assert resp.generated_text == "Once upon a time, there was a dragon."
        assert resp.operation_type == EnumLlmOperationType.COMPLETION
        assert resp.finish_reason == EnumLlmFinishReason.STOP
        assert resp.provider_id == "ollama"

    @pytest.mark.asyncio
    async def test_completion_url_uses_api_generate(self) -> None:
        """Verify URL passed to _execute_llm_http_call ends with /api/generate."""
        handler = _make_handler()
        mock = _mock_call(handler)
        mock.return_value = {
            "response": "text",
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            await handler.handle(_make_completion_request())

        call_args = mock.call_args
        assert call_args is not None
        url = call_args.kwargs.get("url") or call_args.args[0]
        assert url.endswith("/api/generate")
        assert url == f"{_BASE_URL}/api/generate"

    @pytest.mark.asyncio
    async def test_completion_none_response_field(self) -> None:
        """Absent response.response field produces generated_text=None."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 0,
            "prompt_eval_count": 5,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_completion_request())

        assert resp.generated_text is None
        assert resp.tool_calls == ()

    @pytest.mark.asyncio
    async def test_completion_no_tool_calls(self) -> None:
        """COMPLETION responses never have tool_calls (raw_tool_calls is always None)."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "response": "some text",
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 5,
            "prompt_eval_count": 3,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_completion_request())

        assert resp.tool_calls == ()
        assert resp.generated_text == "some text"


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandleErrors:
    """Tests for error handling paths."""

    @pytest.mark.asyncio
    async def test_embedding_operation_raises_protocol_config_error(self) -> None:
        """EMBEDDING operation raises ProtocolConfigurationError with correlation_id."""
        handler = _make_handler()
        corr_id = uuid4()
        request = ModelLlmInferenceRequest(
            base_url=_BASE_URL,
            model=_MODEL,
            operation_type=EnumLlmOperationType.EMBEDDING,
            prompt="embed this",
            correlation_id=corr_id,
        )

        with pytest.raises(
            ProtocolConfigurationError, match="does not support EMBEDDING"
        ):
            await handler.handle(request)

        # Verify _execute_llm_http_call was never called
        _mock_call(handler).assert_not_called()

    @pytest.mark.asyncio
    async def test_base_url_trailing_slash_stripped(self) -> None:
        """base_url with trailing slash produces correct URL (no double slash)."""
        handler = _make_handler()
        mock = _mock_call(handler)
        mock.return_value = {
            "message": {"role": "assistant", "content": "hi"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        request = _make_chat_request(base_url="http://host:11434/")

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            await handler.handle(request)

        call_args = mock.call_args
        assert call_args is not None
        url = call_args.kwargs.get("url") or call_args.args[0]
        assert url == "http://host:11434/api/chat"
        assert "//" not in url.split("://")[1]
