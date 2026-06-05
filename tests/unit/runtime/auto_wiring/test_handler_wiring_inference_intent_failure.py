# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for inference-intent validation failures.

Malformed delegation ``ModelInferenceIntent`` payloads used to be rejected by
auto-wiring before ``HandlerInferenceIntent.handle()`` could return its normal
``ModelInferenceResponseData(error_message=...)`` failure surface. That left the
delegation workflow waiting for a terminal event until the caller timed out.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from omnibase_core.models.delegation.wire import ModelInferenceResponseData
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_dispatch_callback,
    _make_payload_type_matcher,
)
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

pytestmark = pytest.mark.unit

_INFERENCE_RESPONSE_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"


class _MustNotRunInferenceHandler:
    def __init__(self) -> None:
        self.called = False

    def handle(self, payload: object) -> object:
        self.called = True
        raise AssertionError("validation failures must not call the handler")


def _inference_event_model_ref() -> ModelHandlerRef:
    return ModelHandlerRef(
        name="ModelInferenceIntent",
        module="omnibase_core.models.delegation.wire",
    )


def _invalid_inference_payload(correlation_id: UUID) -> dict[str, object]:
    return {
        "intent": "llm_inference",
        "base_url": "http://127.0.0.1:8001",
        "model": "qwen-test",
        "system_prompt": "",
        "prompt": "produce a short result",
        "max_tokens": 128,
        "temperature": 0.3,
        "timeout_seconds": 30.0,
        "correlation_id": str(correlation_id),
        "api_key": None,
    }


def _envelope(
    *,
    correlation_id: UUID,
    payload: dict[str, object],
) -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object](
        payload=payload,
        correlation_id=correlation_id,
        envelope_timestamp=datetime.now(UTC),
        event_type="ModelInferenceIntent",
        payload_type="ModelInferenceIntent",
        source_tool="test-handler-wiring",
    )


def test_inference_intent_matcher_accepts_claimed_but_invalid_payload() -> None:
    matcher = _make_payload_type_matcher(_inference_event_model_ref())
    correlation_id = UUID("00000000-0000-0000-0000-000000000127")

    assert matcher(_invalid_inference_payload(correlation_id)) is True
    assert (
        matcher({"intent": "routing_reducer", "correlation_id": str(correlation_id)})
        is False
    )


@pytest.mark.asyncio
async def test_invalid_inference_intent_returns_correlated_error_response() -> None:
    correlation_id = UUID("00000000-0000-0000-0000-000000001270")
    handler = _MustNotRunInferenceHandler()
    callback = _make_dispatch_callback(
        handler,  # type: ignore[arg-type]
        event_model=_inference_event_model_ref(),
    )

    result = await callback(
        _envelope(
            correlation_id=correlation_id,
            payload=_invalid_inference_payload(correlation_id),
        )
    )

    assert handler.called is False
    assert result is not None
    assert result.status is EnumDispatchStatus.SUCCESS
    assert result.correlation_id == correlation_id
    assert result.output_count == 1
    assert len(result.output_events) == 1

    response = result.output_events[0]
    assert isinstance(response, ModelInferenceResponseData)
    assert response.correlation_id == correlation_id
    assert response.model_used == "qwen-test"
    assert response.content == ""
    assert "ModelInferenceIntent validation failed" in response.error_message
    assert "api_key" in response.error_message


@pytest.mark.asyncio
async def test_invalid_inference_intent_error_response_publishes_to_inference_topic() -> (
    None
):
    correlation_id = UUID("00000000-0000-0000-0000-000000012700")
    callback = _make_dispatch_callback(
        _MustNotRunInferenceHandler(),  # type: ignore[arg-type]
        event_model=_inference_event_model_ref(),
    )
    result = await callback(
        _envelope(
            correlation_id=correlation_id,
            payload=_invalid_inference_payload(correlation_id),
        )
    )
    assert result is not None

    bus = AsyncMock()
    applier = DispatchResultApplier(
        event_bus=bus,
        output_topic="onex.evt.omnibase-infra.delegation-completed.v1",
        output_topic_map={"InferenceResponseData": _INFERENCE_RESPONSE_TOPIC},
    )

    await applier.apply(result, correlation_id=correlation_id)

    bus.publish_envelope.assert_awaited_once()
    call_kwargs = bus.publish_envelope.call_args.kwargs
    assert call_kwargs["topic"] == _INFERENCE_RESPONSE_TOPIC
    assert call_kwargs["envelope"].correlation_id == correlation_id
    assert isinstance(call_kwargs["envelope"].payload, ModelInferenceResponseData)
