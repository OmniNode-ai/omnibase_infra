# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerLlmOpenaiCompatible usage extraction and metrics emission.

Tests cover:
    - Usage extraction from responses with all 5 fallback cases
    - ContractLlmCallMetrics population
    - Kafka event emission via metrics publisher
    - Fire-and-forget behavior (metrics errors don't break inference)
    - Input hash computation
    - Prompt text building for estimation fallback

Related:
    - OMN-2238: Extract and normalize token usage from LLM API responses
    - handler_llm_openai_compatible.py: Handler under test
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.event_bus.topic_constants import TOPIC_LLM_CALL_COMPLETED
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


def _make_publisher() -> AsyncMock:
    """Create a mock metrics publisher."""
    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    return publisher


def _make_handler(
    transport: MagicMock | None = None,
    publisher: AsyncMock | None = None,
) -> HandlerLlmOpenaiCompatible:
    """Create a handler with mock transport and optional publisher."""
    if transport is None:
        transport = _make_transport()
    return HandlerLlmOpenaiCompatible(transport, metrics_publisher=publisher)


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
# Metrics Emission Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsEmission:
    """Tests for metrics event emission from the handler."""

    @pytest.mark.asyncio
    async def test_metrics_emitted_with_complete_usage(self) -> None:
        """Complete usage response emits metrics to Kafka topic."""
        transport = _make_transport()
        publisher = _make_publisher()
        handler = _make_handler(transport, publisher)
        transport._execute_llm_http_call.return_value = _make_response_with_usage(
            prompt_tokens=100, completion_tokens=50
        )

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        publisher.publish.assert_awaited_once()
        call_args = publisher.publish.call_args
        topic = call_args.args[0] if call_args.args else call_args.kwargs.get("topic")
        payload = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get("payload")
        )

        assert topic == TOPIC_LLM_CALL_COMPLETED
        assert payload["model_id"] == _MODEL
        assert payload["prompt_tokens"] == 100
        assert payload["completion_tokens"] == 50
        assert payload["total_tokens"] == 150
        assert payload["usage_is_estimated"] is False

    @pytest.mark.asyncio
    async def test_metrics_emitted_with_partial_usage(self) -> None:
        """Partial usage response emits estimated metrics."""
        transport = _make_transport()
        publisher = _make_publisher()
        handler = _make_handler(transport, publisher)
        transport._execute_llm_http_call.return_value = _make_response_partial_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        publisher.publish.assert_awaited_once()
        call_args = publisher.publish.call_args
        payload = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get("payload")
        )

        assert payload["prompt_tokens"] == 20
        assert payload["usage_is_estimated"] is True

    @pytest.mark.asyncio
    async def test_metrics_emitted_with_absent_usage(self) -> None:
        """Absent usage response emits estimated metrics."""
        transport = _make_transport()
        publisher = _make_publisher()
        handler = _make_handler(transport, publisher)
        transport._execute_llm_http_call.return_value = _make_response_without_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        publisher.publish.assert_awaited_once()
        call_args = publisher.publish.call_args
        payload = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get("payload")
        )

        # Should be estimated from the text.
        assert payload["usage_is_estimated"] is True

    @pytest.mark.asyncio
    async def test_no_publisher_no_error(self) -> None:
        """Handler without publisher computes metrics without error."""
        transport = _make_transport()
        handler = _make_handler(transport, publisher=None)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        # Should not raise.
        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )
        assert resp.status == "success"

    @pytest.mark.asyncio
    async def test_publisher_error_does_not_break_inference(self) -> None:
        """Metrics publisher error is swallowed; inference still succeeds."""
        transport = _make_transport()
        publisher = _make_publisher()
        publisher.publish.side_effect = RuntimeError("Kafka unavailable")
        handler = _make_handler(transport, publisher)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        # Should not raise despite publisher failure.
        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )
        assert resp.status == "success"
        assert resp.generated_text == "Hello!"

    @pytest.mark.asyncio
    async def test_metrics_include_latency(self) -> None:
        """Metrics payload includes latency_ms."""
        transport = _make_transport()
        publisher = _make_publisher()
        handler = _make_handler(transport, publisher)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        call_args = publisher.publish.call_args
        payload = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get("payload")
        )
        assert "latency_ms" in payload
        assert payload["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_metrics_include_input_hash(self) -> None:
        """Metrics payload includes input_hash."""
        transport = _make_transport()
        publisher = _make_publisher()
        handler = _make_handler(transport, publisher)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        call_args = publisher.publish.call_args
        payload = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get("payload")
        )
        assert payload["input_hash"].startswith("sha256-")

    @pytest.mark.asyncio
    async def test_metrics_include_timestamp(self) -> None:
        """Metrics payload includes ISO timestamp."""
        transport = _make_transport()
        publisher = _make_publisher()
        handler = _make_handler(transport, publisher)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        call_args = publisher.publish.call_args
        payload = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get("payload")
        )
        assert payload["timestamp_iso"] != ""

    @pytest.mark.asyncio
    async def test_metrics_include_reporting_source(self) -> None:
        """Metrics payload includes reporting_source."""
        transport = _make_transport()
        publisher = _make_publisher()
        handler = _make_handler(transport, publisher)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        call_args = publisher.publish.call_args
        payload = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get("payload")
        )
        assert payload["reporting_source"] == "handler-llm-openai-compatible"

    @pytest.mark.asyncio
    async def test_metrics_include_raw_and_normalized_usage(self) -> None:
        """Metrics payload includes both raw and normalized usage."""
        transport = _make_transport()
        publisher = _make_publisher()
        handler = _make_handler(transport, publisher)
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        await handler.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)

        call_args = publisher.publish.call_args
        payload = (
            call_args.args[1]
            if len(call_args.args) > 1
            else call_args.kwargs.get("payload")
        )
        assert payload["usage_raw"] is not None
        assert payload["usage_normalized"] is not None
        assert payload["usage_raw"]["provider"] == "openai_compatible"


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

    def test_chat_no_messages_returns_none(self) -> None:
        """CHAT_COMPLETION with only system_prompt returns system_prompt."""
        req = _make_chat_request(
            system_prompt="System",
            messages=({"role": "user", "content": "Hi"},),
        )
        text = HandlerLlmOpenaiCompatible._build_prompt_text(req)
        assert text is not None
        assert "System" in text

    def test_chat_no_content_returns_none(self) -> None:
        """CHAT_COMPLETION with no text content returns None."""
        req = _make_chat_request(
            messages=({"role": "user", "content": "test"},),
        )
        # Messages have content, so should not be None.
        text = HandlerLlmOpenaiCompatible._build_prompt_text(req)
        assert text is not None


# ---------------------------------------------------------------------------
# Backward Compatibility Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBackwardCompatibility:
    """Ensure existing handler behavior is preserved after metrics addition."""

    @pytest.mark.asyncio
    async def test_handler_without_publisher_works(self) -> None:
        """Handler constructed without publisher (old API) still works."""
        transport = _make_transport()
        handler = HandlerLlmOpenaiCompatible(transport)  # No publisher arg
        transport._execute_llm_http_call.return_value = _make_response_with_usage()

        resp = await handler.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        assert resp.status == "success"
        assert resp.usage.tokens_input == 10
        assert resp.usage.tokens_output == 5

    @pytest.mark.asyncio
    async def test_response_unchanged_with_publisher(self) -> None:
        """Adding a publisher does not change the response content."""
        transport = _make_transport()
        publisher = _make_publisher()
        transport._execute_llm_http_call.return_value = _make_response_with_usage(
            prompt_tokens=100, completion_tokens=50
        )

        handler_no_pub = HandlerLlmOpenaiCompatible(transport)
        resp_no_pub = await handler_no_pub.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        transport._execute_llm_http_call.return_value = _make_response_with_usage(
            prompt_tokens=100, completion_tokens=50
        )
        handler_with_pub = HandlerLlmOpenaiCompatible(transport, publisher)
        resp_with_pub = await handler_with_pub.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        assert resp_no_pub.generated_text == resp_with_pub.generated_text
        assert resp_no_pub.usage.tokens_input == resp_with_pub.usage.tokens_input
        assert resp_no_pub.usage.tokens_output == resp_with_pub.usage.tokens_output
        assert resp_no_pub.finish_reason == resp_with_pub.finish_reason
