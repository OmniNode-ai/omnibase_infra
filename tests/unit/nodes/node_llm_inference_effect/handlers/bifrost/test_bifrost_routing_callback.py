# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for Bifrost routing decision callback (OMN-7040 Task 8).

Tests that the optional on_routing_decision callback is invoked with
correct ModelBifrostResponse data on both success and failure paths,
and that None callback is a safe no-op.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.enums.enum_cost_tier import EnumCostTier
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.models.llm.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.models.llm.model_llm_usage import ModelLlmUsage
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost import (
    HandlerBifrostGateway,
    ModelBifrostConfig,
    ModelBifrostRequest,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_config import (
    ModelBifrostBackendConfig,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)

pytestmark = pytest.mark.unit


def _make_inference_response() -> ModelLlmInferenceResponse:
    return ModelLlmInferenceResponse(
        generated_text="Hello from bifrost",
        model_used="test-model",
        operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        finish_reason=EnumLlmFinishReason.STOP,
        usage=ModelLlmUsage(),
        latency_ms=100.0,
        backend_result=ModelBackendResult(success=True, duration_ms=100.0),
        correlation_id=uuid4(),
        execution_id=uuid4(),
        timestamp=datetime.now(UTC),
    )


def _make_handler(
    inference_response: ModelLlmInferenceResponse | None = None,
) -> HandlerLlmOpenaiCompatible:
    transport = MagicMock(spec=MixinLlmHttpTransport)
    handler = HandlerLlmOpenaiCompatible(transport=transport)
    if inference_response is None:
        inference_response = _make_inference_response()
    handler.handle = AsyncMock(return_value=inference_response)
    return handler


def _make_config() -> ModelBifrostConfig:
    return ModelBifrostConfig(
        backends={
            "backend-a": ModelBifrostBackendConfig(
                backend_id="backend-a",
                base_url="http://backend-a:8000",
                model_name="model-a",
            ),
        },
        default_backends=("backend-a",),
        failover_attempts=1,
        failover_backoff_base_ms=0,
    )


def _make_request() -> ModelBifrostRequest:
    return ModelBifrostRequest(
        operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        cost_tier=EnumCostTier.MID,
        tenant_id=uuid4(),
        messages=({"role": "user", "content": "test"},),
    )


@pytest.mark.asyncio
async def test_callback_invoked_on_success() -> None:
    """Callback receives ModelBifrostResponse on successful routing."""
    callback = MagicMock()
    handler = _make_handler()
    config = _make_config()
    gateway = HandlerBifrostGateway(
        config=config,
        inference_handler=handler,
        on_routing_decision=callback,
    )
    request = _make_request()

    response = await gateway.handle(request)

    callback.assert_called_once()
    invoked_response = callback.call_args[0][0]
    assert invoked_response is response
    assert invoked_response.success is True
    assert invoked_response.backend_selected == "backend-a"


@pytest.mark.asyncio
async def test_callback_invoked_on_all_backends_failed() -> None:
    """Callback receives ModelBifrostResponse even when all backends fail."""
    callback = MagicMock()
    handler = _make_handler()
    handler.handle = AsyncMock(side_effect=RuntimeError("backend down"))
    config = _make_config()
    gateway = HandlerBifrostGateway(
        config=config,
        inference_handler=handler,
        on_routing_decision=callback,
    )
    request = _make_request()

    response = await gateway.handle(request)

    callback.assert_called_once()
    invoked_response = callback.call_args[0][0]
    assert invoked_response is response
    assert invoked_response.success is False
    assert invoked_response.backend_selected == ""


@pytest.mark.asyncio
async def test_none_callback_is_noop() -> None:
    """None callback does not cause errors (default behavior)."""
    handler = _make_handler()
    config = _make_config()
    gateway = HandlerBifrostGateway(
        config=config,
        inference_handler=handler,
        on_routing_decision=None,
    )
    request = _make_request()

    response = await gateway.handle(request)

    assert response.success is True


@pytest.mark.asyncio
async def test_callback_exception_does_not_crash_gateway() -> None:
    """Gateway still returns a valid response even if callback raises."""
    callback = MagicMock(side_effect=RuntimeError("callback exploded"))
    handler = _make_handler()
    config = _make_config()
    gateway = HandlerBifrostGateway(
        config=config,
        inference_handler=handler,
        on_routing_decision=callback,
    )
    request = _make_request()

    response = await gateway.handle(request)

    assert response.success is True
    assert response.backend_selected == "backend-a"
    callback.assert_called_once()


@pytest.mark.asyncio
async def test_callback_receives_correct_routing_metadata() -> None:
    """Callback response includes matched_rule_id, latency_ms, retry_count."""
    callback = MagicMock()
    handler = _make_handler()
    config = _make_config()
    gateway = HandlerBifrostGateway(
        config=config,
        inference_handler=handler,
        on_routing_decision=callback,
    )
    request = _make_request()

    await gateway.handle(request)

    invoked_response = callback.call_args[0][0]
    assert invoked_response.latency_ms >= 0.0
    assert invoked_response.retry_count >= 0
    assert invoked_response.correlation_id is not None
    assert invoked_response.tenant_id == request.tenant_id


__all__: list[str] = []
