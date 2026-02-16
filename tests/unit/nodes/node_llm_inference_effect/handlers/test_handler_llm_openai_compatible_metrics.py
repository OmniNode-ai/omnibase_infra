# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerLlmOpenaiCompatible usage extraction and metrics building.

Tests cover:
    - Usage extraction from responses with all 5 fallback cases
    - ContractLlmCallMetrics population via last_call_metrics
    - Fire-and-forget behavior (metrics errors don't break inference)
    - Input hash computation
    - Prompt text building for estimation fallback

Related:
    - OMN-2238: Extract and normalize token usage from LLM API responses
    - handler_llm_openai_compatible.py: Handler under test
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
    _compute_input_hash,
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


def _make_handler(
    transport: MagicMock | None = None,
) -> HandlerLlmOpenaiCompatible:
    """Create a handler with mock transport."""
    if transport is None:
        transport = _make_transport()
    return HandlerLlmOpenaiCompatible(transport)


def _make_chat_request(**overrides: Any) -> ModelLlmInferenceRequest:
    """Build a valid CHAT_COMPLETION request."""
    defaults: dict[str, Any] = {
        "base_url": _BASE_URL,
        "model": _MODEL,
        "operation_type": EnumLlmOperationType.CHAT_COMPLETION,
        "messages": ({"role": "user", "content": "Hello"},),
    }
    defaults.update(overrides)
    return ModelLlmInferenceRequest(**defaults)


def _make_completion_request(**overrides: Any) -> ModelLlmInferenceRequest:
    """Build a valid COMPLETION request."""
    defaults: dict[str, Any] = {
        "base_url": _BASE_URL,
        "model": _MODEL,
        "operation_type": EnumLlmOperationType.COMPLETION,
        "prompt": "Once upon a time",
    }
    defaults.update(overrides)
    return ModelLlmInferenceRequest(**defaults)


def _make_response_with_usage(
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict[str, Any]:
    """Build a response with complete usage data."""
    return {
        "id": "chatcmpl-abc",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_response_without_usage() -> dict[str, Any]:
    """Build a response without usage data."""
    return {
        "id": "chatcmpl-nousage",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop",
            },
        ],
    }


def _make_response_partial_usage() -> dict[str, Any]:
    """Build a response with partial usage data (missing completion_tokens)."""
    return {
        "id": "chatcmpl-partial",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 20,
        },
    }


# ---------------------------------------------------------------------------
# Metrics Building Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsBuilding:
    """Tests for metrics building via handler.last_call_metrics."""

    @pytest.mark.asyncio
    async def test_metrics_built_with_complete_usage(self) -> None:
        """Complete usage response populates last_call_metrics correctly."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_response_with_usage(
            prompt_tokens=100, completion_tokens=50
        )

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        metrics = handler.last_call_metrics
        assert metrics is not None
        assert metrics.model_id == _MODEL
        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.usage_is_estimated is False

    @pytest.mark.asyncio
    async def test_metrics_built_with_partial_usage(self) -> None:
        """Partial usage response builds estimated metrics."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_response_partial_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        metrics = handler.last_call_metrics
        assert metrics is not None
        assert metrics.prompt_tokens == 20
        assert metrics.usage_is_estimated is True

    @pytest.mark.asyncio
    async def test_metrics_built_with_absent_usage(self) -> None:
        """Absent usage response builds estimated metrics."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_response_without_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        metrics = handler.last_call_metrics
        assert metrics is not None
        # Should be estimated from the text.
        assert metrics.usage_is_estimated is True

    @pytest.mark.asyncio
    async def test_metrics_available_after_handle(self) -> None:
        """Handler stores metrics for caller retrieval after handle()."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        # Before handle, metrics should be None.
        assert handler.last_call_metrics is None

        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )
        assert resp.status == "success"

        # After handle, metrics should be populated.
        assert handler.last_call_metrics is not None

    @pytest.mark.asyncio
    async def test_metrics_include_latency(self) -> None:
        """Metrics include latency_ms."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        metrics = handler.last_call_metrics
        assert metrics is not None
        assert metrics.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_metrics_include_input_hash(self) -> None:
        """Metrics include input_hash."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        metrics = handler.last_call_metrics
        assert metrics is not None
        assert metrics.input_hash.startswith("sha256-")

    @pytest.mark.asyncio
    async def test_metrics_include_timestamp(self) -> None:
        """Metrics include ISO timestamp."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        metrics = handler.last_call_metrics
        assert metrics is not None
        assert metrics.timestamp_iso != ""

    @pytest.mark.asyncio
    async def test_metrics_include_reporting_source(self) -> None:
        """Metrics include reporting_source."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        metrics = handler.last_call_metrics
        assert metrics is not None
        assert metrics.reporting_source == "handler-llm-openai-compatible"

    @pytest.mark.asyncio
    async def test_metrics_include_raw_and_normalized_usage(self) -> None:
        """Metrics include both raw and normalized usage."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        metrics = handler.last_call_metrics
        assert metrics is not None
        assert metrics.usage_raw is not None
        assert metrics.usage_normalized is not None
        assert metrics.usage_raw.provider == "openai_compatible"

    @pytest.mark.asyncio
    async def test_metrics_reset_between_calls(self) -> None:
        """last_call_metrics is reset at the start of each handle() call."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_response_with_usage(
            prompt_tokens=10, completion_tokens=5
        )

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        first_metrics = handler.last_call_metrics

        transport._execute_llm_http_call.return_value = _make_response_with_usage(
            prompt_tokens=200, completion_tokens=100
        )

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        second_metrics = handler.last_call_metrics

        assert first_metrics is not None
        assert second_metrics is not None
        assert first_metrics.prompt_tokens == 10
        assert second_metrics.prompt_tokens == 200


# ---------------------------------------------------------------------------
# Fire-and-Forget Behavior Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsFireAndForget:
    """Tests that metrics errors never break inference flow."""

    @pytest.mark.asyncio
    async def test_handle_succeeds_when_normalize_llm_usage_raises(self) -> None:
        """When normalize_llm_usage raises, handle() still returns a valid response."""
        transport = _make_transport()
        handler = _make_handler(transport)
        transport._execute_llm_http_call.return_value = _make_response_with_usage(
            prompt_tokens=10, completion_tokens=5
        )

        with patch(
            "omnibase_infra.nodes.node_llm_inference_effect.handlers"
            ".handler_llm_openai_compatible.normalize_llm_usage",
            side_effect=ValueError("normalizer exploded"),
        ):
            resp = await handler.handle(
                _make_chat_request(), correlation_id=_CORRELATION_ID
            )

        # Response should still be valid despite metrics failure.
        assert resp is not None
        assert resp.generated_text == "Hello!"
        assert resp.model_used == _MODEL
        assert resp.correlation_id == _CORRELATION_ID

        # Metrics should be None because building failed.
        assert handler.last_call_metrics is None


# ---------------------------------------------------------------------------
# Input Hash Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeInputHash:
    """Tests for _compute_input_hash."""

    def test_deterministic(self) -> None:
        """Same request produces same hash."""
        req = _make_chat_request()
        assert _compute_input_hash(req) == _compute_input_hash(req)

    def test_different_messages_different_hash(self) -> None:
        """Different messages produce different hashes."""
        req1 = _make_chat_request(messages=({"role": "user", "content": "Hello"},))
        req2 = _make_chat_request(messages=({"role": "user", "content": "Goodbye"},))
        assert _compute_input_hash(req1) != _compute_input_hash(req2)

    def test_prefix(self) -> None:
        """Hash is prefixed with sha256-."""
        req = _make_chat_request()
        assert _compute_input_hash(req).startswith("sha256-")

    def test_completion_request_hash(self) -> None:
        """COMPLETION request produces valid hash."""
        req = _make_completion_request()
        h = _compute_input_hash(req)
        assert h.startswith("sha256-")
        assert len(h) == len("sha256-") + 64


# ---------------------------------------------------------------------------
# Prompt Text Building Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildPromptText:
    """Tests for HandlerLlmOpenaiCompatible._build_prompt_text."""

    def test_completion_returns_prompt(self) -> None:
        """COMPLETION request returns the prompt field."""
        req = _make_completion_request(prompt="Complete this sentence")
        text = HandlerLlmOpenaiCompatible._build_prompt_text(req)
        assert text == "Complete this sentence"

    def test_chat_concatenates_messages(self) -> None:
        """CHAT_COMPLETION request concatenates message contents."""
        req = _make_chat_request(
            system_prompt="You are helpful",
            messages=(
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "World"},
            ),
        )
        text = HandlerLlmOpenaiCompatible._build_prompt_text(req)
        assert text is not None
        assert "You are helpful" in text
        assert "Hello" in text
        assert "World" in text

    def test_chat_with_system_prompt_includes_system_prompt(self) -> None:
        """CHAT_COMPLETION with system_prompt includes it in returned text."""
        req = _make_chat_request(
            system_prompt="System",
            messages=({"role": "user", "content": "Hi"},),
        )
        text = HandlerLlmOpenaiCompatible._build_prompt_text(req)
        assert text is not None
        assert "System" in text

    def test_chat_with_content_returns_text(self) -> None:
        """CHAT_COMPLETION with message content returns concatenated text."""
        req = _make_chat_request(
            messages=({"role": "user", "content": "test"},),
        )
        text = HandlerLlmOpenaiCompatible._build_prompt_text(req)
        assert text is not None


# ---------------------------------------------------------------------------
# Backward Compatibility Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBackwardCompatibility:
    """Ensure existing handler behavior is preserved after metrics addition."""

    @pytest.mark.asyncio
    async def test_handler_works(self) -> None:
        """Handler constructs and operates correctly."""
        transport = _make_transport()
        handler = HandlerLlmOpenaiCompatible(transport)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        assert resp.status == "success"
        assert resp.usage.tokens_input == 10
        assert resp.usage.tokens_output == 5

    @pytest.mark.asyncio
    async def test_response_content_unchanged(self) -> None:
        """Metrics building does not change the response content."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage(
            prompt_tokens=100, completion_tokens=50
        )

        handler = HandlerLlmOpenaiCompatible(transport)
        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        assert resp.generated_text == "Hello!"
        assert resp.usage.tokens_input == 100
        assert resp.usage.tokens_output == 50
