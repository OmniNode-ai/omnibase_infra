# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.llm import ModelLlmInferenceResponse, ModelLlmUsage
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_llm_inference_effect.handlers import (
    handler_llm_inference_command as command_handler_module,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.helpers.util_kafka import KafkaTopicManager, wait_for_consumer_ready

pytestmark = pytest.mark.integration

COMMAND_TOPIC = "onex.cmd.omnibase-infra.llm-inference-request.v1"
RESPONSE_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"
INTELLIGENCE_TOPIC = "onex.evt.omniintelligence.llm-call-completed.v1"
INFRA_TOPIC = "onex.evt.omnibase-infra.llm-call-completed.v1"
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
KAFKA_AVAILABLE = (
    os.getenv("KAFKA_BOOTSTRAP_SERVERS") is not None
    and os.getenv("KAFKA_INTEGRATION_TESTS") == "1"
)


class _CapturedRequest(BaseModel):
    url: str


def _contract(
    *,
    command_topic: str = COMMAND_TOPIC,
    response_topic: str = RESPONSE_TOPIC,
    intelligence_topic: str = INTELLIGENCE_TOPIC,
    infra_topic: str = INFRA_TOPIC,
    contract_path: Path | None = None,
) -> ModelDiscoveredContract:
    effective_contract_path = contract_path or (
        Path(__file__).parents[3]
        / "src/omnibase_infra/nodes/node_llm_inference_effect/contract.yaml"
    )
    return ModelDiscoveredContract(
        name="node_llm_inference_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=4, patch=1),
        contract_path=effective_contract_path,
        entry_point_name="node_llm_inference_effect",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(command_topic,),
            publish_topics=(response_topic, intelligence_topic, infra_topic),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    event_model=ModelHandlerRef(
                        name="ModelLlmInferenceCommand",
                        module="omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_command",
                    ),
                    event_type="omnibase-infra.llm-inference-request",
                    message_category="COMMAND",
                    handler=ModelHandlerRef(
                        name="HandlerLlmInferenceCommand",
                        module="omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_inference_command",
                    ),
                ),
            ),
        ),
    )


async def _wire_runtime(bus: EventBusInmemory) -> MessageDispatchEngine:
    engine = MessageDispatchEngine()
    await wire_from_manifest(
        ModelAutoWiringManifest(contracts=(_contract(),)),
        engine,
        event_bus=bus,
        environment="test",
    )
    engine.freeze()
    return engine


def _write_topic_map_contract(
    tmp_path: Path,
    *,
    response_topic: str,
    intelligence_topic: str,
    infra_topic: str,
) -> Path:
    contract_path = tmp_path / "node_llm_inference_effect.contract.yaml"
    contract_path.write_text(
        "\n".join(
            [
                "published_events:",
                f'  - topic: "{response_topic}"',
                '    event_type: "LlmInferenceResponse"',
                f'  - topic: "{intelligence_topic}"',
                '    event_type: "LlmCallCompletedEvent"',
                f'  - topic: "{infra_topic}"',
                '    event_type: "LlmCallCompletedInfraEvent"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return contract_path


def _install_fake_http(
    monkeypatch: pytest.MonkeyPatch,
) -> asyncio.Queue[_CapturedRequest]:
    captured: asyncio.Queue[_CapturedRequest] = asyncio.Queue()

    async def fake_execute(
        self: object,
        *,
        url: str,
        payload: dict[str, object],
        correlation_id: UUID,
        timeout_seconds: float,
    ) -> dict[str, object]:
        await captured.put(_CapturedRequest(url=url))
        return {
            "id": "chatcmpl-test",
            "model": "gemini-2.5-pro",
            "choices": [
                {
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

    monkeypatch.setattr(
        command_handler_module.LlmInferenceCommandTransport,
        "_execute_llm_http_call",
        fake_execute,
    )
    return captured


@pytest.mark.asyncio
async def test_llm_inference_contract_runs_through_inmemory_runtime_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_http(monkeypatch)

    bus = EventBusInmemory(environment="test", group="llm-contract")
    await bus.start()
    try:
        response_events: asyncio.Queue[ModelEventEnvelope[object]] = asyncio.Queue()
        intelligence_events: asyncio.Queue[ModelEventEnvelope[object]] = asyncio.Queue()
        infra_events: asyncio.Queue[ModelEventEnvelope[object]] = asyncio.Queue()

        async def collect_response(message: ModelEventMessage) -> None:
            await response_events.put(
                ModelEventEnvelope[object].model_validate_json(message.value)
            )

        async def collect_intelligence(message: ModelEventMessage) -> None:
            await intelligence_events.put(
                ModelEventEnvelope[object].model_validate_json(message.value)
            )

        async def collect_infra(message: ModelEventMessage) -> None:
            await infra_events.put(
                ModelEventEnvelope[object].model_validate_json(message.value)
            )

        await bus.subscribe(
            RESPONSE_TOPIC,
            group_id="response-collector",
            on_message=collect_response,
        )
        await bus.subscribe(
            INTELLIGENCE_TOPIC,
            group_id="intelligence-collector",
            on_message=collect_intelligence,
        )
        await bus.subscribe(
            INFRA_TOPIC,
            group_id="infra-collector",
            on_message=collect_infra,
        )
        await _wire_runtime(bus)

        correlation_id = UUID("11111111-1111-4111-8111-111111111111")
        command = {
            "correlation_id": str(correlation_id),
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": "ping"}],
            "provider_config": {"endpoint_url": GEMINI_ENDPOINT},
        }
        await bus.publish(
            COMMAND_TOPIC,
            None,
            json.dumps(command).encode("utf-8"),
            None,
        )

        request = await asyncio.wait_for(captured.get(), timeout=2)
        response = await asyncio.wait_for(response_events.get(), timeout=2)
        intelligence = await asyncio.wait_for(intelligence_events.get(), timeout=2)
        infra = await asyncio.wait_for(infra_events.get(), timeout=2)

        assert request.url == GEMINI_ENDPOINT
        assert response.correlation_id == correlation_id
        assert response.payload["generated_text"] == "ok"
        assert response.payload["model_used"] == "gemini-2.5-pro"
        assert intelligence.correlation_id == correlation_id
        assert infra.correlation_id == correlation_id
        assert intelligence.payload["model_id"] == "gemini-2.5-pro"
        assert infra.payload["endpoint_url"] == GEMINI_ENDPOINT
    finally:
        await bus.close()


@pytest.mark.kafka
@pytest.mark.skipif(
    not KAFKA_AVAILABLE,
    reason="Kafka/Redpanda integration disabled; set KAFKA_INTEGRATION_TESTS=1",
)
@pytest.mark.asyncio
async def test_llm_inference_contract_runs_through_redpanda_runtime_bus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    captured = _install_fake_http(monkeypatch)
    bootstrap_servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    suffix = uuid4().hex[:12]
    command_topic = f"onex.cmd.omnibase-infra.llm-inference-request-{suffix}.v1"
    response_topic = f"onex.evt.omnibase-infra.inference-response-{suffix}.v1"
    intelligence_topic = f"onex.evt.omniintelligence.llm-call-completed-{suffix}.v1"
    infra_topic = f"onex.evt.omnibase-infra.llm-call-completed-{suffix}.v1"
    contract_path = _write_topic_map_contract(
        tmp_path,
        response_topic=response_topic,
        intelligence_topic=intelligence_topic,
        infra_topic=infra_topic,
    )
    contract = _contract(
        command_topic=command_topic,
        response_topic=response_topic,
        intelligence_topic=intelligence_topic,
        infra_topic=infra_topic,
        contract_path=contract_path,
    )

    async with KafkaTopicManager(bootstrap_servers) as topic_manager:
        await topic_manager.create_topic(command_topic)
        await topic_manager.create_topic(response_topic)
        await topic_manager.create_topic(intelligence_topic)
        await topic_manager.create_topic(infra_topic)

        bus = EventBusKafka(
            config=ModelKafkaEventBusConfig(
                bootstrap_servers=bootstrap_servers,
                environment="test",
                timeout_seconds=30,
                max_retry_attempts=2,
                retry_backoff_base=0.5,
            )
        )
        await bus.start()
        try:
            response_events: asyncio.Queue[ModelEventEnvelope[object]] = asyncio.Queue()
            intelligence_events: asyncio.Queue[ModelEventEnvelope[object]] = (
                asyncio.Queue()
            )
            infra_events: asyncio.Queue[ModelEventEnvelope[object]] = asyncio.Queue()

            async def collect_response(message: ModelEventMessage) -> None:
                await response_events.put(
                    ModelEventEnvelope[object].model_validate_json(message.value)
                )

            async def collect_intelligence(message: ModelEventMessage) -> None:
                await intelligence_events.put(
                    ModelEventEnvelope[object].model_validate_json(message.value)
                )

            async def collect_infra(message: ModelEventMessage) -> None:
                await infra_events.put(
                    ModelEventEnvelope[object].model_validate_json(message.value)
                )

            await bus.subscribe(
                response_topic,
                group_id=f"llm-response-{suffix}",
                on_message=collect_response,
            )
            await bus.subscribe(
                intelligence_topic,
                group_id=f"llm-intelligence-{suffix}",
                on_message=collect_intelligence,
            )
            await bus.subscribe(
                infra_topic,
                group_id=f"llm-infra-{suffix}",
                on_message=collect_infra,
            )
            engine = MessageDispatchEngine()
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(contract,)),
                engine,
                event_bus=bus,
                environment="test",
            )
            engine.freeze()
            await wait_for_consumer_ready(bus, command_topic)
            await wait_for_consumer_ready(bus, response_topic)
            await wait_for_consumer_ready(bus, intelligence_topic)
            await wait_for_consumer_ready(bus, infra_topic)

            correlation_id = UUID("22222222-2222-4222-8222-222222222222")
            command = {
                "correlation_id": str(correlation_id),
                "model": "gemini-2.5-pro",
                "messages": [{"role": "user", "content": "ping"}],
                "provider_config": {"endpoint_url": GEMINI_ENDPOINT},
            }
            await bus.publish(
                command_topic,
                None,
                json.dumps(command).encode("utf-8"),
                None,
            )

            request = await asyncio.wait_for(captured.get(), timeout=10)
            response = await asyncio.wait_for(response_events.get(), timeout=10)
            intelligence = await asyncio.wait_for(intelligence_events.get(), timeout=10)
            infra = await asyncio.wait_for(infra_events.get(), timeout=10)

            assert request.url == GEMINI_ENDPOINT
            assert response.correlation_id == correlation_id
            assert response.payload["generated_text"] == "ok"
            assert response.payload["model_used"] == "gemini-2.5-pro"
            assert intelligence.correlation_id == correlation_id
            assert infra.correlation_id == correlation_id
            assert intelligence.payload["model_id"] == "gemini-2.5-pro"
            assert infra.payload["endpoint_url"] == GEMINI_ENDPOINT
        finally:
            await bus.close()


def test_llm_inference_redpanda_runtime_bus_contract_is_declared() -> None:
    """Redpanda parity target uses the same contract route as in-memory wiring.

    This guard stays lightweight locally; the runtime Redpanda job owns broker
    availability and executes the same contract/topic path against Kafka.
    """
    contract = _contract()
    assert contract.event_bus is not None
    assert COMMAND_TOPIC in contract.event_bus.subscribe_topics
    assert RESPONSE_TOPIC in contract.event_bus.publish_topics
    assert INTELLIGENCE_TOPIC in contract.event_bus.publish_topics
    assert INFRA_TOPIC in contract.event_bus.publish_topics
