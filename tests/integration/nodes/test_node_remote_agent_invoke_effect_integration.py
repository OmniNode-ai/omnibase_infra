# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for node_remote_agent_invoke_effect contract wiring."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums.enum_agent_protocol import EnumAgentProtocol
from omnibase_core.enums.enum_agent_task_lifecycle_type import (
    EnumAgentTaskLifecycleType,
)
from omnibase_core.enums.enum_invocation_kind import EnumInvocationKind
from omnibase_core.models.common.model_schema_value import ModelSchemaValue
from omnibase_core.models.delegation import (
    ModelAgentTaskLifecycleEvent,
    ModelInvocationCommand,
    ModelRemoteTaskState,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
    load_event_bus_subcontract,
)
from omnibase_infra.utils import compute_consumer_group_id
from tests.helpers.util_kafka import KafkaTopicManager, wait_for_consumer_ready

CONTRACT_PATH = (
    Path("src")
    / "omnibase_infra"
    / "nodes"
    / "node_remote_agent_invoke_effect"
    / "contract.yaml"
)
NODE_NAME = "node_remote_agent_invoke_effect"
SERVICE_NAME = "omnibase-infra"
NODE_VERSION = "v0.1.0"


class RecordingDispatchEngine:
    """Small async dispatch double used to observe real Kafka delivery."""

    def __init__(self) -> None:
        self.received: list[tuple[str, ModelEventEnvelope[object]]] = []
        self.message_received = asyncio.Event()

    async def dispatch(
        self,
        topic: str,
        envelope: ModelEventEnvelope[object],
    ) -> None:
        self.received.append((topic, envelope))
        self.message_received.set()


class LocalRemoteTaskStateStore:
    """In-memory state store used by the local P12 run."""

    def __init__(self) -> None:
        self._rows: dict[UUID, ModelRemoteTaskState] = {}

    async def upsert(self, state: ModelRemoteTaskState) -> None:
        self._rows[state.task_id] = state

    async def get(self, task_id: UUID) -> ModelRemoteTaskState | None:
        return self._rows.get(task_id)


class LocalRemoteAgentInvokeDispatchEngine:
    """Local dispatch implementation for the P12 in-memory end-to-end run."""

    def __init__(
        self,
        event_bus: EventBusInmemory,
        state_store: LocalRemoteTaskStateStore,
        lifecycle_topic: str,
    ) -> None:
        self._event_bus = event_bus
        self._state_store = state_store
        self._lifecycle_topic = lifecycle_topic

    async def dispatch(
        self,
        topic: str,
        envelope: ModelEventEnvelope[object],
    ) -> None:
        command = ModelInvocationCommand.model_validate(envelope.payload)
        now = datetime.now(UTC)
        remote_handle = f"local-a2a:{command.task_id}"

        await self._state_store.upsert(
            ModelRemoteTaskState(
                task_id=command.task_id,
                invocation_kind=command.invocation_kind,
                protocol=command.agent_protocol,
                target_ref=command.target_ref,
                remote_task_handle=remote_handle,
                correlation_id=command.correlation_id,
                status=EnumAgentTaskLifecycleType.SUBMITTED,
                last_remote_status="submitted",
                last_emitted_event_type=EnumAgentTaskLifecycleType.SUBMITTED,
                submitted_at=now,
                updated_at=now,
            )
        )

        lifecycle_event = ModelAgentTaskLifecycleEvent(
            task_id=command.task_id,
            correlation_id=command.correlation_id,
            lifecycle_type=EnumAgentTaskLifecycleType.SUBMITTED,
            remote_task_handle=remote_handle,
            artifact={"source_topic": ModelSchemaValue.from_value(topic)},
            occurred_at=now,
            remote_status="submitted",
        )
        lifecycle_envelope = ModelEventEnvelope[dict[str, object]](
            correlation_id=command.correlation_id,
            payload=lifecycle_event.model_dump(mode="json"),
        )
        await self._event_bus.publish(
            self._lifecycle_topic,
            key=str(command.task_id).encode("utf-8"),
            value=lifecycle_envelope.model_dump_json().encode("utf-8"),
        )


async def _ensure_contract_topic_exists(topic: str, bootstrap_servers: str) -> None:
    """Create the contract topic if absent without deleting canonical topics."""
    manager = KafkaTopicManager(bootstrap_servers)
    try:
        await manager.create_topic(topic, partitions=1, replication_factor=1)
        manager.created_topics.clear()
    finally:
        await manager.cleanup()


@pytest.mark.integration
@pytest.mark.kafka
@pytest.mark.asyncio
async def test_remote_agent_invoke_contract_wires_real_kafka_dispatch() -> None:
    """Contract-declared remote-agent invoke topic reaches dispatch via Kafka."""
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    if not bootstrap_servers:
        pytest.skip("KAFKA_BOOTSTRAP_SERVERS is not configured")

    subcontract = load_event_bus_subcontract(CONTRACT_PATH)
    assert subcontract is not None
    assert len(subcontract.subscribe_topics) == 1
    invoke_topic = subcontract.subscribe_topics[0]

    await _ensure_contract_topic_exists(invoke_topic, bootstrap_servers)

    config = ModelKafkaEventBusConfig(
        bootstrap_servers=bootstrap_servers,
        environment="dev",
        timeout_seconds=30,
        max_retry_attempts=2,
        retry_backoff_base=0.5,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=10.0,
    )
    event_bus = EventBusKafka(config=config)
    dispatch_engine = RecordingDispatchEngine()
    wiring = EventBusSubcontractWiring(
        event_bus=event_bus,
        dispatch_engine=dispatch_engine,
        environment="dev",
        node_name=NODE_NAME,
        service=SERVICE_NAME,
        version=NODE_VERSION,
    )

    try:
        await event_bus.start()
        await wiring.wire_subscriptions(subcontract, node_name=NODE_NAME)
        await wait_for_consumer_ready(
            event_bus,
            invoke_topic,
            max_wait=10.0,
        )

        health = await event_bus.health_check()
        assert health["subscriber_count"] == 1
        assert health["consumer_count"] == 1

        expected_group = compute_consumer_group_id(
            ModelNodeIdentity(
                env="dev",
                service=SERVICE_NAME,
                node_name=NODE_NAME,
                version=NODE_VERSION,
            ),
            EnumConsumerGroupPurpose.CONSUME,
        )
        assert (
            expected_group
            == "dev.omnibase-infra.node_remote_agent_invoke_effect.consume.v0.1.0"
        )

        envelope = ModelEventEnvelope[dict[str, object]](
            correlation_id=uuid4(),
            payload={
                "operation": "submit",
                "target_ref": "agent:test-remote-agent",
                "input": {"prompt": "integration smoke"},
            },
        )
        await event_bus.publish(
            invoke_topic,
            key=str(envelope.envelope_id).encode("utf-8"),
            value=envelope.model_dump_json().encode("utf-8"),
        )

        await asyncio.wait_for(dispatch_engine.message_received.wait(), timeout=10.0)

        assert len(dispatch_engine.received) == 1
        received_topic, received_envelope = dispatch_engine.received[0]
        assert received_topic == invoke_topic
        assert received_envelope.payload == envelope.payload
        assert received_envelope.event_type == "omnibase-infra.remote-agent-invoke"
    finally:
        await wiring.cleanup()
        await event_bus.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remote_agent_invoke_runs_end_to_end_with_inmemory_bus_and_state() -> (
    None
):
    """Run P12 locally: command topic -> state store -> lifecycle topic."""
    subcontract = load_event_bus_subcontract(CONTRACT_PATH)
    assert subcontract is not None
    assert len(subcontract.subscribe_topics) == 1
    assert len(subcontract.publish_topics) == 1
    invoke_topic = subcontract.subscribe_topics[0]
    lifecycle_topic = subcontract.publish_topics[0]

    event_bus = EventBusInmemory(environment="local", group="p12-local-run")
    state_store = LocalRemoteTaskStateStore()
    dispatch_engine = LocalRemoteAgentInvokeDispatchEngine(
        event_bus=event_bus,
        state_store=state_store,
        lifecycle_topic=lifecycle_topic,
    )
    wiring = EventBusSubcontractWiring(
        event_bus=event_bus,
        dispatch_engine=dispatch_engine,
        environment="local",
        node_name=NODE_NAME,
        service=SERVICE_NAME,
        version=NODE_VERSION,
    )
    lifecycle_messages: list[ModelEventMessage] = []
    lifecycle_received = asyncio.Event()

    async def capture_lifecycle(message: ModelEventMessage) -> None:
        lifecycle_messages.append(message)
        lifecycle_received.set()

    lifecycle_identity = ModelNodeIdentity(
        env="local",
        service=SERVICE_NAME,
        node_name=f"{NODE_NAME}_test_observer",
        version=NODE_VERSION,
    )

    try:
        await event_bus.start()
        await event_bus.subscribe(
            topic=lifecycle_topic,
            node_identity=lifecycle_identity,
            on_message=capture_lifecycle,
        )
        await wiring.wire_subscriptions(subcontract, node_name=NODE_NAME)

        command = ModelInvocationCommand(
            task_id=uuid4(),
            correlation_id=uuid4(),
            invocation_kind=EnumInvocationKind.AGENT,
            agent_protocol=EnumAgentProtocol.A2A,
            target_ref="agent:local-a2a-smoke",
            payload={
                "prompt": ModelSchemaValue.from_value("local in-memory P12 smoke")
            },
        )
        command_envelope = ModelEventEnvelope[dict[str, object]](
            correlation_id=command.correlation_id,
            payload=command.model_dump(mode="json"),
        )

        await event_bus.publish(
            invoke_topic,
            key=str(command.task_id).encode("utf-8"),
            value=command_envelope.model_dump_json().encode("utf-8"),
        )
        await asyncio.wait_for(lifecycle_received.wait(), timeout=1.0)

        stored = await state_store.get(command.task_id)
        assert stored is not None
        assert stored.target_ref == "agent:local-a2a-smoke"
        assert stored.status is EnumAgentTaskLifecycleType.SUBMITTED
        assert stored.remote_task_handle == f"local-a2a:{command.task_id}"

        assert len(lifecycle_messages) == 1
        raw_lifecycle_envelope = json.loads(lifecycle_messages[0].value.decode("utf-8"))
        lifecycle_envelope = ModelEventEnvelope[object].model_validate(
            raw_lifecycle_envelope
        )
        lifecycle_event = ModelAgentTaskLifecycleEvent.model_validate(
            lifecycle_envelope.payload
        )
        assert lifecycle_event.task_id == command.task_id
        assert lifecycle_event.correlation_id == command.correlation_id
        assert lifecycle_event.lifecycle_type is EnumAgentTaskLifecycleType.SUBMITTED
        assert lifecycle_event.remote_status == "submitted"
    finally:
        await wiring.cleanup()
        await event_bus.close()
