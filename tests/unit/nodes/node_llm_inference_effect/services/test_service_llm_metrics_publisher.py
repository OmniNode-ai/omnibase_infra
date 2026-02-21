# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceLlmMetricsPublisher.

Tests cover:
    - Metrics are published after a successful inference call
    - Publisher is called with the correct topic and payload fields
    - Publish failures are swallowed and never break inference
    - When last_call_metrics is None, no publish is attempted
    - Correlation ID is threaded through to the publisher

Related:
    - OMN-2443: Wire NodeLlmInferenceEffect to emit llm-call-completed events
    - service_llm_metrics_publisher.py: Module under test
    - TOPIC_LLM_CALL_COMPLETED: onex.evt.omniintelligence.llm-call-completed.v1
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.event_bus.topic_constants import TOPIC_LLM_CALL_COMPLETED
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_ollama import (
    HandlerLlmOllama,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_infra.nodes.node_llm_inference_effect.services.service_llm_metrics_publisher import (
    ServiceLlmMetricsPublisher,
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
    transport._execute_llm_http_call = AsyncMock()
    transport._http_client = None
    transport._owns_http_client = True
    return transport


def _make_response_with_usage(
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict[str, Any]:
    """Build an OpenAI-compatible response with complete usage data."""
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


def _make_publisher() -> AsyncMock:
    """Create an AsyncMock publisher that returns True."""
    pub = AsyncMock(return_value=True)
    return pub


def _make_service(
    transport: MagicMock | None = None,
    publisher: AsyncMock | None = None,
) -> tuple[ServiceLlmMetricsPublisher, HandlerLlmOpenaiCompatible, AsyncMock]:
    """Build a ServiceLlmMetricsPublisher with mock inner handler and publisher."""
    if transport is None:
        transport = _make_transport()
    if publisher is None:
        publisher = _make_publisher()
    handler = HandlerLlmOpenaiCompatible(transport=transport)
    service = ServiceLlmMetricsPublisher(handler=handler, publisher=publisher)
    return service, handler, publisher


# ---------------------------------------------------------------------------
# Core emission tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsEmission:
    """Tests that metrics are published after inference calls."""

    @pytest.mark.asyncio
    async def test_publisher_called_after_successful_inference(self) -> None:
        """Publisher is called once after a successful inference call."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        service, _, publisher = _make_service(transport=transport)

        await service.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        await asyncio.sleep(0)

        publisher.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publisher_called_with_correct_topic(self) -> None:
        """Publisher receives the canonical LLM call completed topic."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        service, _, publisher = _make_service(transport=transport)

        await service.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        await asyncio.sleep(0)

        call_args = publisher.call_args
        assert call_args[0][0] == TOPIC_LLM_CALL_COMPLETED

    @pytest.mark.asyncio
    async def test_publisher_payload_contains_model_id(self) -> None:
        """Published payload contains the model_id field."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        service, _, publisher = _make_service(transport=transport)

        await service.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        await asyncio.sleep(0)

        payload = publisher.call_args[0][1]
        assert isinstance(payload, dict)
        assert payload["model_id"] == _MODEL

    @pytest.mark.asyncio
    async def test_publisher_payload_contains_token_counts(self) -> None:
        """Published payload contains prompt_tokens and completion_tokens."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage(
            prompt_tokens=42, completion_tokens=17
        )
        service, _, publisher = _make_service(transport=transport)

        await service.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        await asyncio.sleep(0)

        payload = publisher.call_args[0][1]
        assert payload["prompt_tokens"] == 42
        assert payload["completion_tokens"] == 17
        assert payload["total_tokens"] == 59

    @pytest.mark.asyncio
    async def test_publisher_payload_contains_timestamp_iso(self) -> None:
        """Published payload contains a non-empty timestamp_iso."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        service, _, publisher = _make_service(transport=transport)

        await service.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        await asyncio.sleep(0)

        payload = publisher.call_args[0][1]
        assert isinstance(payload.get("timestamp_iso"), str)
        assert payload["timestamp_iso"] != ""

    @pytest.mark.asyncio
    async def test_publisher_payload_contains_reporting_source(self) -> None:
        """Published payload contains reporting_source from the handler."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        service, _, publisher = _make_service(transport=transport)

        await service.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        await asyncio.sleep(0)

        payload = publisher.call_args[0][1]
        assert payload.get("reporting_source") == "handler-llm-openai-compatible"

    @pytest.mark.asyncio
    async def test_publisher_receives_correlation_id(self) -> None:
        """Publisher receives the correlation_id as third positional argument."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        service, _, publisher = _make_service(transport=transport)

        await service.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        await asyncio.sleep(0)

        call_args = publisher.call_args
        assert call_args[0][2] == str(_CORRELATION_ID)

    @pytest.mark.asyncio
    async def test_inference_response_unchanged_by_emission(self) -> None:
        """Response returned by handle() is unchanged by metrics emission."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage(
            prompt_tokens=100, completion_tokens=50
        )
        service, _, _ = _make_service(transport=transport)

        response = await service.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        assert response.generated_text == "Hello!"
        assert response.usage.tokens_input == 100
        assert response.usage.tokens_output == 50

    @pytest.mark.asyncio
    async def test_publisher_payload_is_json_serializable(self) -> None:
        """Published payload dict is JSON-serializable (no Pydantic models)."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        service, _, publisher = _make_service(transport=transport)

        await service.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        await asyncio.sleep(0)

        payload = publisher.call_args[0][1]
        # Must not raise
        json.dumps(payload)

    @pytest.mark.asyncio
    async def test_auto_generates_correlation_id_when_none(self) -> None:
        """When correlation_id is None, one is generated and passed to publisher."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        service, _, publisher = _make_service(transport=transport)

        await service.handle(_make_chat_request())
        await asyncio.sleep(0)

        call_args = publisher.call_args
        received_corr_id = call_args[0][2]
        # Must be a valid UUID string
        UUID(received_corr_id)  # raises ValueError if invalid


# ---------------------------------------------------------------------------
# Fire-and-forget / resilience tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPublisherResilience:
    """Tests that publish failures never break inference."""

    @pytest.mark.asyncio
    async def test_inference_succeeds_when_publisher_raises(self) -> None:
        """When publisher raises, handle() still returns the response."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        publisher = AsyncMock(side_effect=RuntimeError("Kafka unavailable"))
        service, _, _ = _make_service(transport=transport, publisher=publisher)

        response = await service.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        assert response.generated_text == "Hello!"

    @pytest.mark.asyncio
    async def test_inference_succeeds_when_publisher_returns_false(self) -> None:
        """When publisher returns False (transient failure), handle() still succeeds."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        publisher = AsyncMock(return_value=False)
        service, _, _ = _make_service(transport=transport, publisher=publisher)

        response = await service.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        assert response.generated_text == "Hello!"

    @pytest.mark.asyncio
    async def test_no_publish_when_last_call_metrics_is_none(self) -> None:
        """When last_call_metrics is None (normalizer failed), publisher is not called."""
        publisher = _make_publisher()

        # Use a mock handler whose last_call_metrics is None.  This tests the
        # public contract of ServiceLlmMetricsPublisher — it must skip publish
        # when the handler exposes no metrics — without coupling to any private
        # implementation detail of HandlerLlmOpenaiCompatible.
        mock_response = MagicMock()
        mock_response.generated_text = "Hello!"

        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(return_value=mock_response)
        mock_handler.last_call_metrics = None

        service = ServiceLlmMetricsPublisher(handler=mock_handler, publisher=publisher)

        response = await service.handle(
            _make_chat_request(), correlation_id=_CORRELATION_ID
        )

        # Response still valid
        assert response.generated_text == "Hello!"
        await asyncio.sleep(0)
        # Publisher must NOT have been called (metrics were None)
        publisher.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handler_exception_propagates(self) -> None:
        """When the inner handler raises, the exception is propagated unchanged."""
        transport = _make_transport()
        transport._execute_llm_http_call.side_effect = ConnectionError("timeout")
        service, _, publisher = _make_service(transport=transport)

        with pytest.raises(ConnectionError, match="timeout"):
            await service.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        await asyncio.sleep(0)

        # Publisher must NOT have been called (handler never returned)
        publisher.assert_not_awaited()


# ---------------------------------------------------------------------------
# Multiple calls
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultipleCalls:
    """Tests for correct behaviour across multiple handle() calls."""

    @pytest.mark.asyncio
    async def test_publisher_called_for_each_call(self) -> None:
        """Publisher is called once per handle() invocation."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        service, _, publisher = _make_service(transport=transport)

        await service.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        await asyncio.sleep(0)
        await service.handle(_make_chat_request(), correlation_id=_CORRELATION_ID)
        await asyncio.sleep(0)

        assert publisher.await_count == 2

    @pytest.mark.asyncio
    async def test_each_call_uses_own_correlation_id(self) -> None:
        """Each handle() call passes its own correlation_id to the publisher."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = _make_response_with_usage()
        service, _, publisher = _make_service(transport=transport)

        corr1 = uuid4()
        corr2 = uuid4()
        await service.handle(_make_chat_request(), correlation_id=corr1)
        await asyncio.sleep(0)
        await service.handle(_make_chat_request(), correlation_id=corr2)
        await asyncio.sleep(0)

        first_call_corr = publisher.call_args_list[0][0][2]
        second_call_corr = publisher.call_args_list[1][0][2]
        assert first_call_corr == str(corr1)
        assert second_call_corr == str(corr2)
        assert first_call_corr != second_call_corr


# ---------------------------------------------------------------------------
# Protocol structural compatibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_protocol_structural_compatibility() -> None:
    """Verify HandlerLlmOllama and HandlerLlmOpenaiCompatible satisfy ProtocolLlmHandler structurally.

    ProtocolLlmHandler is not runtime-checkable, so compatibility is verified
    via hasattr checks on the required ``handle`` method.  If either handler
    drops or renames ``handle``, this test will catch the divergence before it
    silently breaks ServiceLlmMetricsPublisher at runtime.
    """
    from omnibase_infra.nodes.node_llm_inference_effect.services.protocol_llm_handler import (
        ProtocolLlmHandler,
    )

    # Confirm the protocol itself declares handle
    assert hasattr(ProtocolLlmHandler, "handle"), "ProtocolLlmHandler missing handle()"

    for handler_cls in [HandlerLlmOllama, HandlerLlmOpenaiCompatible]:
        assert hasattr(handler_cls, "handle"), (
            f"{handler_cls.__name__} missing handle()"
        )
        assert callable(handler_cls.handle), (
            f"{handler_cls.__name__}.handle is not callable"
        )
