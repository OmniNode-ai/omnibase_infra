# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Receipt handler loaded by its adapter spec, exchanging real bus envelopes."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.handlers.handler_receipt_mode_publish_await import (
    HandlerReceiptModePublishAwait,
)
from omnibase_infra.models.model_receipt_mode_event import ModelReceiptModeEvent
from tests.helpers.service_env import require_service_env

pytestmark = pytest.mark.integration

_ADAPTER_SPEC = (
    "omnibase_infra.handlers.handler_receipt_mode_publish_await:"
    "HandlerReceiptModePublishAwait"
)


@pytest.mark.parametrize(
    "transport",
    ["inmemory", pytest.param("kafka", marks=pytest.mark.kafka)],
)
async def test_adapter_loaded_handler_exchanges_correlated_receipts(
    transport: str,
) -> None:
    """The synchronous adapter call receives progress then action and closes."""
    suffix = uuid4().hex[:12]
    command_topic = f"onex.cmd.receipt-mode.requested-{suffix}.v1"
    reply_topic = f"onex.evt.receipt-mode.action-{suffix}.v1"
    invocation_id = str(uuid4())
    responder: EventBusInmemory | EventBusKafka
    client_bus: EventBusInmemory | EventBusKafka
    if transport == "kafka":
        require_service_env(
            opt_in="KAFKA_INTEGRATION_TESTS",
            endpoint="KAFKA_BOOTSTRAP_SERVERS",
            workflow=".github/workflows/ci.yml (service-integration-suites)",
            service="Kafka/Redpanda",
        )
        config = ModelKafkaEventBusConfig(
            bootstrap_servers=os.environ["KAFKA_BOOTSTRAP_SERVERS"],
            environment="test",
            timeout_seconds=20,
            auto_offset_reset="earliest",
        )
        responder = EventBusKafka(config=config)
        client_bus = EventBusKafka(config=config)
    else:
        responder = client_bus = EventBusInmemory()

    commands: list[ModelEventEnvelope[dict[str, object]]] = []

    async def respond(message: ModelEventMessage) -> None:
        command = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        commands.append(command)
        # A foreign action must not finish the adapter's wait.
        replies = (
            {"correlation_id": str(uuid4()), "payload": {"action": "deny"}},
            {
                "correlation_id": str(command.correlation_id),
                "payload": {"status": "ready"},
            },
            {
                "correlation_id": str(uuid4()),
                "payload": {"invocation_id": invocation_id, "action": "allow"},
            },
        )
        for reply in replies:
            await responder.publish(
                reply_topic, key=None, value=json.dumps(reply).encode()
            )

    module_name, attribute = _ADAPTER_SPEC.split(":")
    handler_type = getattr(importlib.import_module(module_name), attribute)
    assert handler_type is HandlerReceiptModePublishAwait
    await responder.start()
    try:
        unsubscribe = await responder.subscribe(
            command_topic, group_id=f"receipt-responder.{suffix}", on_message=respond
        )
        try:
            events = await asyncio.to_thread(
                handler_type(lambda: client_bus).publish_and_await,
                topic=command_topic,
                command_type="receipt-mode.requested",
                payload={"invocation_id": invocation_id},
                awaited_events=(reply_topic,),
                deadline_ms=30000,
            )
            assert events == (
                ModelReceiptModeEvent(topic=reply_topic, payload={"status": "ready"}),
                ModelReceiptModeEvent(
                    topic=reply_topic,
                    payload={"invocation_id": invocation_id, "action": "allow"},
                ),
            )
            assert len(commands) == 1
            assert commands[0].event_type == "receipt-mode.requested"
            assert commands[0].payload == {"invocation_id": invocation_id}
            health = await client_bus.health_check()
            assert health["started"] is False
            assert not client_bus.get_consumer_groups()
        finally:
            await unsubscribe()
    finally:
        await responder.close()
