# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for LLM event-path wiring (OMN-6789).

Verifies the full data-flow chain for LLM dispatch:

1. A synthetic LLM inference request is constructed matching the
   ``onex.cmd.omnibase-infra.llm-inference-request.v1`` command topic.
2. ServiceLlmMetricsPublisher (wrapping a mocked handler) processes it and
   calls the publisher callable, simulating the emit of
   ``onex.evt.omniintelligence.llm-call-completed.v1``.
3. The in-memory event bus delivers the event to a subscriber that records it.
4. The captured event payload is asserted to contain usage metrics.

Covers both the inference path (CHAT_COMPLETION) and the embedding path
(EMBEDDING), proving that the data_flow_sweep LLM event paths are wired
end-to-end without requiring live LLM endpoints or Kafka.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.llm.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.models.llm.model_llm_usage import ModelLlmUsage
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_infra.nodes.node_llm_inference_effect.services.service_llm_metrics_publisher import (
    ServiceLlmMetricsPublisher,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

pytestmark = [pytest.mark.integration]

# Topics read from the canonical registry — never hardcoded.
_REGISTRY = ServiceTopicRegistry.from_defaults()
_TOPIC_INFERENCE_CMD = _REGISTRY.resolve(topic_keys.LLM_INFERENCE_REQUEST)
_TOPIC_LLM_COMPLETED = _REGISTRY.resolve(topic_keys.LLM_CALL_COMPLETED)
_TOPIC_LLM_COMPLETED_INFRA = _REGISTRY.resolve(topic_keys.LLM_CALL_COMPLETED_INFRA)
_TOPIC_EMBEDDING_CMD = _REGISTRY.resolve(topic_keys.LLM_EMBEDDING_REQUEST)


def _make_inference_response(
    *, operation_type: EnumLlmOperationType
) -> ModelLlmInferenceResponse:
    """Build a minimal successful ModelLlmInferenceResponse for mocking."""
    return ModelLlmInferenceResponse(
        generated_text="pong",
        model_used="test-model",
        operation_type=operation_type,
        finish_reason=EnumLlmFinishReason.STOP,
        usage=ModelLlmUsage(tokens_input=10, tokens_output=5),
        latency_ms=42.0,
        backend_result=ModelBackendResult(success=True, duration_ms=40.0),
        correlation_id=uuid4(),
        execution_id=uuid4(),
        timestamp=datetime.now(UTC),
    )


def _make_chat_request() -> ModelLlmInferenceRequest:
    return ModelLlmInferenceRequest(
        base_url="http://localhost:8000",
        model="test-model",
        operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        messages=({"role": "user", "content": "ping"},),
    )


def _make_embedding_request() -> ModelLlmInferenceRequest:
    return ModelLlmInferenceRequest(
        base_url="http://localhost:8100",
        model="test-embedding-model",
        operation_type=EnumLlmOperationType.EMBEDDING,
        prompt="embed this text",
    )


@pytest.mark.asyncio
async def test_inference_request_emits_llm_call_completed_event() -> None:
    """Chat-completion request publishes llm-call-completed with usage metrics.

    Flow:
      mock_handler.handle() -> ServiceLlmMetricsPublisher -> publisher callable
      -> EventBusInmemory.publish_envelope -> subscriber records event
    """
    bus = EventBusInmemory(environment="test", group="llm-dispatch-wiring")
    await bus.start()

    captured: list[ModelEventMessage] = []

    consumer_identity = ModelNodeIdentity(
        env="test",
        service="omnibase-infra",
        node_name="test-llm-wiring",
        version="v1",
    )
    unsubscribe = await bus.subscribe(
        _TOPIC_LLM_COMPLETED,
        consumer_identity,
        on_message=lambda msg: _append(captured, msg),
    )

    # Build a mock handler that returns a successful response and sets
    # last_call_metrics so ServiceLlmMetricsPublisher emits the event.
    response = _make_inference_response(
        operation_type=EnumLlmOperationType.CHAT_COMPLETION
    )
    mock_handler = AsyncMock()
    mock_handler.handle = AsyncMock(return_value=response)

    # ContractLlmCallMetrics minimal stub — publisher only calls model_dump_json().
    from omnibase_spi.contracts.measurement.contract_llm_call_metrics import (
        ContractLlmCallMetrics,
    )

    mock_metrics = ContractLlmCallMetrics(
        model_id="test-model",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=42.0,
    )
    mock_handler.last_call_metrics = mock_metrics

    async def _publisher(
        topic: str, payload: object, correlation_id: str | None = None
    ) -> bool:
        await bus.publish_envelope(payload, topic=topic)
        return True

    service = ServiceLlmMetricsPublisher(
        handler=mock_handler,
        publisher=_publisher,
        topic_registry=_REGISTRY,
    )

    request = _make_chat_request()
    await service.handle(request)

    # ServiceLlmMetricsPublisher schedules emit as a background task — drain it.
    await asyncio.sleep(0)

    assert len(captured) >= 1, (
        f"Expected at least one event on {_TOPIC_LLM_COMPLETED!r} but got {len(captured)}. "
        "ServiceLlmMetricsPublisher may not be publishing to the event bus."
    )

    event_payload = json.loads(captured[0].value.decode())
    assert "model_id" in event_payload, (
        "Published llm-call-completed payload missing 'model_id' field. "
        f"Payload keys: {list(event_payload.keys())}"
    )
    assert event_payload["prompt_tokens"] == 10
    assert event_payload["completion_tokens"] == 5
    assert event_payload["total_tokens"] == 15

    await unsubscribe()
    await bus.close()


def test_inference_subscribe_topic_matches_contract() -> None:
    """The contract subscribe topic string matches the canonical registry key.

    Parses the node contract YAML directly to verify the subscribe_topics
    declared in the contract match the canonical registry resolution.
    """
    from pathlib import Path

    import yaml

    contract_yaml = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "omnibase_infra"
        / "nodes"
        / "node_llm_inference_effect"
        / "contract.yaml"
    )
    contract = yaml.safe_load(contract_yaml.read_text(encoding="utf-8"))
    subscribe_topics = contract["event_bus"]["subscribe_topics"]

    assert _TOPIC_INFERENCE_CMD in subscribe_topics, (
        f"Expected {_TOPIC_INFERENCE_CMD!r} in contract event_bus.subscribe_topics. "
        f"Got: {subscribe_topics!r}"
    )


@pytest.mark.asyncio
async def test_embedding_request_path_publisher_invoked() -> None:
    """Embedding request path invokes the publisher callable.

    The ticket requires coverage of the embedding request path. This test
    verifies that a request with operation_type=EMBEDDING flows through
    ServiceLlmMetricsPublisher and the publisher is called with the
    expected completed-event topic.
    """
    published_topics: list[str] = []

    async def _capturing_publisher(
        topic: str, payload: object, correlation_id: str | None = None
    ) -> bool:
        published_topics.append(topic)
        return True

    response = _make_inference_response(operation_type=EnumLlmOperationType.EMBEDDING)
    mock_handler = AsyncMock()
    mock_handler.handle = AsyncMock(return_value=response)

    from omnibase_spi.contracts.measurement.contract_llm_call_metrics import (
        ContractLlmCallMetrics,
    )

    mock_handler.last_call_metrics = ContractLlmCallMetrics(
        model_id="test-embedding-model",
        prompt_tokens=20,
        completion_tokens=0,
        total_tokens=20,
        latency_ms=15.0,
    )

    service = ServiceLlmMetricsPublisher(
        handler=mock_handler,
        publisher=_capturing_publisher,
        topic_registry=_REGISTRY,
    )

    request = _make_embedding_request()
    await service.handle(request)

    # Drain background task.
    await asyncio.sleep(0)

    assert _TOPIC_LLM_COMPLETED in published_topics, (
        f"ServiceLlmMetricsPublisher did not publish to {_TOPIC_LLM_COMPLETED!r} "
        f"for embedding request. Published topics: {published_topics!r}"
    )


@pytest.mark.asyncio
async def test_llm_call_completed_infra_topic_also_published() -> None:
    """Both omniintelligence and omnibase-infra completed topics receive events."""
    published_topics: list[str] = []

    async def _capturing_publisher(
        topic: str, payload: object, correlation_id: str | None = None
    ) -> bool:
        published_topics.append(topic)
        return True

    response = _make_inference_response(
        operation_type=EnumLlmOperationType.CHAT_COMPLETION
    )
    mock_handler = AsyncMock()
    mock_handler.handle = AsyncMock(return_value=response)

    from omnibase_spi.contracts.measurement.contract_llm_call_metrics import (
        ContractLlmCallMetrics,
    )

    mock_handler.last_call_metrics = ContractLlmCallMetrics(
        model_id="test-model",
        prompt_tokens=8,
        completion_tokens=4,
        total_tokens=12,
        latency_ms=30.0,
    )

    service = ServiceLlmMetricsPublisher(
        handler=mock_handler,
        publisher=_capturing_publisher,
        topic_registry=_REGISTRY,
    )

    await service.handle(_make_chat_request())
    await asyncio.sleep(0)

    assert _TOPIC_LLM_COMPLETED in published_topics, (
        f"Missing omniintelligence topic. Published: {published_topics!r}"
    )
    assert _TOPIC_LLM_COMPLETED_INFRA in published_topics, (
        f"Missing omnibase-infra infra topic. Published: {published_topics!r}"
    )


async def _append(lst: list[ModelEventMessage], msg: ModelEventMessage) -> None:
    lst.append(msg)


__all__: list[str] = []
