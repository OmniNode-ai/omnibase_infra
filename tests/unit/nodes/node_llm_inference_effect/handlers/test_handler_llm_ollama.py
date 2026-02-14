# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerLlmOllama error mapping and edge cases.

Tests cover:
    - Error propagation from transport layer (all HTTP status -> exception pairs)
    - Unknown finish_reason string -> UNKNOWN enum fallback
    - EMBEDDING operation rejection
    - Correlation ID generation when missing from request

These tests complement the existing Ollama handler test files:
    - test_handler_llm_ollama_class.py: Initialization, properties, payload building
    - test_handler_llm_ollama_handle.py: Handle method happy paths
    - test_handler_llm_ollama_pure.py: Module-level pure functions

All tests mock ``_execute_llm_http_call`` on the handler to control HTTP
responses and isolate the handler's translation logic.

Related:
    - HandlerLlmOllama: The handler under test
    - MixinLlmHttpTransport: Transport mocked at its boundary
    - OMN-2109: Phase 9 inference handler tests
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import UUID

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
from omnibase_infra.models.errors.model_timeout_error_context import (
    ModelTimeoutErrorContext,
)
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
    """Build a valid CHAT_COMPLETION request with sensible defaults."""
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
    """Build a valid COMPLETION request with sensible defaults."""
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


def _make_error_context() -> ModelInfraErrorContext:
    """Create a minimal error context for exception construction."""
    return ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.HTTP,
        operation="test",
    )


# ---------------------------------------------------------------------------
# Error Mapping Tests (HTTP status -> exception pairs)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOllamaErrorPropagation:
    """Tests for error mapping: HTTP status -> exception type.

    The transport layer (_execute_llm_http_call) raises typed exceptions
    based on HTTP status codes. These tests verify the Ollama handler
    propagates each exception type correctly without swallowing or
    transforming them.
    """

    @pytest.mark.asyncio
    async def test_401_raises_auth_error(self) -> None:
        """401 from transport -> InfraAuthenticationError."""
        handler = _make_handler()
        _mock_call(handler).side_effect = InfraAuthenticationError(
            "Auth failed (401)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraAuthenticationError):
            await handler.handle(_make_chat_request())

    @pytest.mark.asyncio
    async def test_403_raises_auth_error(self) -> None:
        """403 from transport -> InfraAuthenticationError."""
        handler = _make_handler()
        _mock_call(handler).side_effect = InfraAuthenticationError(
            "Forbidden (403)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraAuthenticationError):
            await handler.handle(_make_chat_request())

    @pytest.mark.asyncio
    async def test_429_raises_rate_limited_error(self) -> None:
        """429 from transport -> InfraRateLimitedError."""
        handler = _make_handler()
        _mock_call(handler).side_effect = InfraRateLimitedError(
            "Rate limited (429)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraRateLimitedError):
            await handler.handle(_make_chat_request())

    @pytest.mark.asyncio
    async def test_400_raises_request_rejected_error(self) -> None:
        """400 from transport -> InfraRequestRejectedError."""
        handler = _make_handler()
        _mock_call(handler).side_effect = InfraRequestRejectedError(
            "Bad request (400)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraRequestRejectedError):
            await handler.handle(_make_chat_request())

    @pytest.mark.asyncio
    async def test_422_raises_request_rejected_error(self) -> None:
        """422 from transport -> InfraRequestRejectedError."""
        handler = _make_handler()
        _mock_call(handler).side_effect = InfraRequestRejectedError(
            "Unprocessable (422)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraRequestRejectedError):
            await handler.handle(_make_chat_request())

    @pytest.mark.asyncio
    async def test_404_raises_config_error(self) -> None:
        """404 from transport -> ProtocolConfigurationError."""
        handler = _make_handler()
        _mock_call(handler).side_effect = ProtocolConfigurationError(
            "Not found (404)",
            context=_make_error_context(),
        )

        with pytest.raises(ProtocolConfigurationError):
            await handler.handle(_make_chat_request())

    @pytest.mark.asyncio
    async def test_500_raises_unavailable_error(self) -> None:
        """500 from transport -> InfraUnavailableError."""
        handler = _make_handler()
        _mock_call(handler).side_effect = InfraUnavailableError(
            "Internal server error (500)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraUnavailableError):
            await handler.handle(_make_chat_request())

    @pytest.mark.asyncio
    async def test_502_raises_unavailable_error(self) -> None:
        """502 from transport -> InfraUnavailableError."""
        handler = _make_handler()
        _mock_call(handler).side_effect = InfraUnavailableError(
            "Bad gateway (502)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraUnavailableError):
            await handler.handle(_make_chat_request())

    @pytest.mark.asyncio
    async def test_503_raises_unavailable_error(self) -> None:
        """503 from transport -> InfraUnavailableError."""
        handler = _make_handler()
        _mock_call(handler).side_effect = InfraUnavailableError(
            "Service unavailable (503)",
            context=_make_error_context(),
        )

        with pytest.raises(InfraUnavailableError):
            await handler.handle(_make_chat_request())

    @pytest.mark.asyncio
    async def test_connection_error_propagated(self) -> None:
        """Connection failure from transport -> InfraConnectionError."""
        handler = _make_handler()
        _mock_call(handler).side_effect = InfraConnectionError(
            "Connection refused",
            context=_make_error_context(),
        )

        with pytest.raises(InfraConnectionError):
            await handler.handle(_make_chat_request())

    @pytest.mark.asyncio
    async def test_timeout_error_propagated(self) -> None:
        """Timeout from transport -> InfraTimeoutError."""
        handler = _make_handler()
        _mock_call(handler).side_effect = InfraTimeoutError(
            "Request timed out",
            context=ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation="test",
            ),
        )

        with pytest.raises(InfraTimeoutError):
            await handler.handle(_make_chat_request())

    @pytest.mark.asyncio
    async def test_embedding_operation_raises_config_error(self) -> None:
        """EMBEDDING operation raises ProtocolConfigurationError before HTTP call."""
        handler = _make_handler()
        request = ModelLlmInferenceRequest(
            base_url=_BASE_URL,
            model=_MODEL,
            operation_type=EnumLlmOperationType.EMBEDDING,
            prompt="embed this",
            correlation_id=_CORRELATION_ID,
        )

        with pytest.raises(
            ProtocolConfigurationError, match="does not support EMBEDDING"
        ):
            await handler.handle(request)

        # Verify _execute_llm_http_call was never called
        _mock_call(handler).assert_not_called()


# ---------------------------------------------------------------------------
# Unknown Finish Reason Fallback Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOllamaUnknownFinishReason:
    """Tests for unknown finish_reason string -> UNKNOWN enum fallback."""

    @pytest.mark.asyncio
    async def test_unrecognized_done_reason_maps_to_UNKNOWN(self) -> None:
        """An unrecognized done_reason string maps to UNKNOWN, not crash."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": _MODEL,
            "done_reason": "some_future_provider_reason",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.finish_reason == EnumLlmFinishReason.UNKNOWN

    @pytest.mark.asyncio
    async def test_none_done_reason_maps_to_UNKNOWN(self) -> None:
        """None done_reason maps to UNKNOWN."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": _MODEL,
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.finish_reason == EnumLlmFinishReason.UNKNOWN

    @pytest.mark.asyncio
    async def test_empty_string_done_reason_maps_to_UNKNOWN(self) -> None:
        """Empty string done_reason maps to UNKNOWN."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": _MODEL,
            "done_reason": "",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(_make_chat_request())

        assert resp.finish_reason == EnumLlmFinishReason.UNKNOWN


# ---------------------------------------------------------------------------
# Correlation ID Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOllamaCorrelationId:
    """Tests for correlation ID handling."""

    @pytest.mark.asyncio
    async def test_correlation_id_auto_generated_when_none(self) -> None:
        """When request has no correlation_id, one is auto-generated."""
        handler = _make_handler()
        _mock_call(handler).return_value = {
            "message": {"role": "assistant", "content": "hi"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        request = ModelLlmInferenceRequest(
            base_url=_BASE_URL,
            model=_MODEL,
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            messages=(ModelLlmMessage(role="user", content="Hello"),),
            # correlation_id is auto-generated by default_factory=uuid4
        )

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            resp = await handler.handle(request)

        assert resp.correlation_id is not None


# ---------------------------------------------------------------------------
# URL Routing Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOllamaUrlRouting:
    """Tests for CHAT_COMPLETION vs COMPLETION URL routing."""

    @pytest.mark.asyncio
    async def test_chat_completion_uses_api_chat(self) -> None:
        """CHAT_COMPLETION routes to /api/chat."""
        handler = _make_handler()
        mock = _mock_call(handler)
        mock.return_value = {
            "message": {"role": "assistant", "content": "hi"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            await handler.handle(_make_chat_request())

        call_args = mock.call_args
        assert call_args is not None
        url = call_args.kwargs.get("url") or call_args.args[0]
        assert url == f"{_BASE_URL}/api/chat"

    @pytest.mark.asyncio
    async def test_completion_uses_api_generate(self) -> None:
        """COMPLETION routes to /api/generate."""
        handler = _make_handler()
        mock = _mock_call(handler)
        mock.return_value = {
            "response": "some text",
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
        assert url == f"{_BASE_URL}/api/generate"


# ---------------------------------------------------------------------------
# System Prompt Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOllamaSystemPrompt:
    """Tests for system_prompt injection in Ollama handler."""

    @pytest.mark.asyncio
    async def test_system_prompt_included_in_payload(self) -> None:
        """system_prompt produces 'system' key in the payload."""
        handler = _make_handler()
        mock = _mock_call(handler)
        mock.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        request = _make_chat_request(system_prompt="You are a helpful assistant.")

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            await handler.handle(request)

        call_args = mock.call_args
        assert call_args is not None
        payload = call_args.kwargs.get("payload") or call_args.args[1]
        assert isinstance(payload, dict)
        assert payload.get("system") == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_no_system_prompt_no_system_key(self) -> None:
        """No 'system' key when system_prompt is None."""
        handler = _make_handler()
        mock = _mock_call(handler)
        mock.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": _MODEL,
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 1,
        }

        request = _make_chat_request(system_prompt=None)

        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            await handler.handle(request)

        call_args = mock.call_args
        assert call_args is not None
        payload = call_args.kwargs.get("payload") or call_args.args[1]
        assert isinstance(payload, dict)
        assert "system" not in payload
